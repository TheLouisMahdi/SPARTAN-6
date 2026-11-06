"""Explicit USER1 probe/CNN backends, persistent XSDB, one owner, no UART fallback.

Programming requires an explicit SVF argument from a separately gated build;
providing a prepared rollback also restores it on normal session close/EOF.
"""
import binascii
import os
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
from bounded_log import BoundedLog

ROOT = Path(__file__).resolve().parents[1]
XSDB = Path(r"C:\Xilinx\Vivado\2023.2\bin\xsdb.bat")
HEADER = struct.Struct("<BBIH")
REPLY = struct.Struct("<4sIHHBBH")
CNN_REPLY = struct.Struct("<4sIHHBBHiiIIQB")


def probe_packet(sequence, payload, command=1):
    if len(payload) > 6144 or command not in (1, 3) or (command == 3 and payload):
        raise ValueError("invalid bounded probe request")
    body = HEADER.pack(1, command, sequence, len(payload))+payload
    return bytes(16)+body+struct.pack("<H", binascii.crc_hqx(body, 0xffff))+b"\0"


def parse_reply(data):
    if len(data) < REPLY.size:
        raise ValueError("truncated JTAG reply")
    signature, sequence, size, crc, status, version, accepted = REPLY.unpack_from(data)
    if signature != b"JTG1" or version != 1:
        raise ValueError("USER1 probe signature/version mismatch")
    return dict(sequence=sequence, size=size, crc=crc, status=status, accepted=accepted)


def recover_echo(captured, packet):
    body = packet[16:-1]
    # After 128 reply bits, a one-bit shift register echoes the request bits.
    recovered = ((int.from_bytes(captured, "little") >> 129) & ((1 << (len(body)*8))-1)).to_bytes(len(body), "little")
    if recovered != body:
        raise ValueError("bit-exact USER1 echo mismatch")
    return recovered


class JtagSession:
    def __init__(self, log_path, *, experimental_probe=False, timeout=5, program_probe_svf=None,
                 expected_usercode=0x4a544731, rollback_svf=None):
        if not experimental_probe:
            raise ValueError("explicit experimental probe selection required")
        if not XSDB.is_file():
            raise FileNotFoundError(XSDB)
        self.owner = threading.get_ident()
        self.timeout = timeout
        self.number = 0
        self.closed = False
        self.programming = program_probe_svf is not None
        self.lines = queue.Queue(maxsize=128)
        self.log = BoundedLog(log_path)
        mode = ["experimental-probe"]
        if program_probe_svf is not None:
            path = Path(program_probe_svf).resolve()
            if not path.is_file():
                self.log.close()
                raise FileNotFoundError(path)
            if expected_usercode not in (0x4a544731, 0x4a544e31, 0x4a545931):
                self.log.close()
                raise ValueError("unsupported experimental image")
            mode = ["program-probe", str(path), str(expected_usercode)]
            if rollback_svf is not None:
                rollback=Path(rollback_svf).resolve()
                if not rollback.is_file():
                    self.log.close()
                    raise FileNotFoundError(rollback)
                mode.append(str(rollback))
        try:
            self.process = subprocess.Popen([os.environ["COMSPEC"], "/d", "/c", str(XSDB),
                str(ROOT/"fpga/jtag_stream.tcl"), *mode], cwd=ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW)
        except BaseException:
            self.log.close()
            raise
        def read():
            try:
                for line in self.process.stdout:
                    self.log.write(line)
                    self.lines.put(line.rstrip("\r\n"))
            except Exception as exc:
                self.lines.put("JTERROR\t0\t"+str(exc).encode("utf-8").hex())
            finally:
                self.lines.put(None)
        self.reader = threading.Thread(target=read, name="xsdb-output", daemon=True)
        self.reader.start()
        try:
            ready = self._receive("JTREADY", startup=True, programming=program_probe_svf is not None)
            self.frequency = int(ready[1])
        except BaseException as exc:
            self.log.write(f"HOST_STARTUP_ERROR {exc!r}\n")
            self.close()
            raise

    def _receive(self, expected, startup=False, programming=False):
        deadline = time.perf_counter()+max(60 if programming else 20 if startup else self.timeout, self.timeout)
        while time.perf_counter() < deadline:
            try:
                line = self.lines.get(timeout=max(.001, deadline-time.perf_counter()))
            except queue.Empty:
                break
            if line is None:
                raise RuntimeError("XSDB exited; see backend log")
            fields = line.split("\t")
            if fields[0] == "JTERROR":
                raise RuntimeError(bytes.fromhex(fields[2]).decode("utf-8", errors="replace"))
            if fields[0] == expected:
                if not startup and fields[1] != str(self.number):
                    raise RuntimeError("XSDB response identity mismatch")
                return fields
        raise TimeoutError("XSDB response timeout; no UART fallback")

    def _request(self, op, expected, *args):
        if self.closed or threading.get_ident() != self.owner:
            raise RuntimeError("JTAG connection must remain on its owner thread")
        self.number += 1
        self.process.stdin.write("\t".join((op, str(self.number), *(str(a) for a in args)))+"\n")
        self.process.stdin.flush()
        return self._receive(expected)

    def set_frequency(self, frequency):
        self.frequency = int(self._request("FREQUENCY", "JTFREQ", frequency)[2])
        return self.frequency

    def reset(self):
        self._request("RESET", "JTRESET")

    def restore_baseline(self):
        previous=self.timeout
        self.timeout=max(previous,30)
        try:self._request("RESTORE", "JTRESTORED")
        finally:self.timeout=previous

    def scan(self, data, capture=True, repeat=1):
        started = time.perf_counter_ns()
        fields = self._request("SCAN", "JTSCAN", int(capture), repeat, data.hex())
        finished = time.perf_counter_ns()
        return {"data": bytes.fromhex(fields[5]), "wall_ms": (finished-started)/1e6,
                "build_ms": int(fields[2])/1000, "execute_ms": int(fields[3])/1000,
                "format_ms": int(fields[4])/1000, "tck": self.frequency,
                "shift_bits": len(data)*8*repeat}

    def exchange_eye(self, packet):
        if len(packet) != 198:
            raise ValueError("compound exchange requires one compact 198-byte eye")
        started = time.perf_counter_ns()
        fields = self._request("EYE", "JTEYE", packet.hex())
        return dict(data=bytes.fromhex(fields[5]), wall_ms=(time.perf_counter_ns()-started)/1e6,
                    upload_ms=int(fields[2])/1000, poll_count=int(fields[3]),
                    exchange_ms=int(fields[4])/1000, upload_provenance="XSDB Tcl upload interval")

    def transact(self, sequence, payload, command=1, capture=False):
        packet = probe_packet(sequence, payload, command)
        started = time.perf_counter_ns()
        upload = self.scan(packet, capture)
        if capture:
            recover_echo(upload["data"], packet)
        expected_crc = binascii.crc_hqx(packet[16:-3], 0xffff)
        deadline = time.perf_counter()+self.timeout
        polls = []
        while time.perf_counter() < deadline:
            result = self.scan(bytes(16))
            polls.append({k: v for k, v in result.items() if k != "data"})
            reply = parse_reply(result["data"])
            if reply["sequence"] == sequence and reply["status"] not in (8, 9):
                if reply["size"] != len(payload) or reply["crc"] != expected_crc:
                    raise ValueError("JTAG receiver length/CRC identity mismatch")
                return dict(reply, upload={k:v for k,v in upload.items() if k!="data"}, polls=polls,
                            wall_ms=(time.perf_counter_ns()-started)/1e6, echo_verified=capture)
        raise TimeoutError("USER1 result missing after bounded polling")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None:
                self.process.stdin.write("CLOSE\t0\n")
                self.process.stdin.flush()
                self.process.wait(timeout=30 if self.programming else 3)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            self.log.write(f"HOST_CLEANUP_ERROR {exc!r}\n")
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=3)
        finally:
            try:
                self.process.stdin.close()
            except (OSError, ValueError) as exc:
                self.log.write(f"HOST_PIPE_CLOSE_ERROR {exc!r}\n")
            self.reader.join(timeout=3)
            self.process.stdout.close()
            self.log.close()


def cnn_packet(sequence, frame_id, capture_ns, eye, pixels, model_id=0x4daa1000):
    if len(pixels)!=3072 or eye not in (0,1):
        raise ValueError("expected exact 3072-byte eye and side 0/1")
    payload=struct.pack("<IQBI",frame_id,capture_ns,eye,model_id)+pixels
    body=HEADER.pack(1,2,sequence,len(payload))+payload
    return bytes(CNN_REPLY.size)+body+struct.pack("<H",binascii.crc_hqx(body,0xffff))+b"\0"


def parse_cnn_reply(data):
    if len(data)!=CNN_REPLY.size:
        raise ValueError("unexpected JTAG CNN reply length")
    signature,sequence,size,crc,status,version,accepted,first,second,cycles,frame,capture,eye=CNN_REPLY.unpack(data)
    if signature!=b"JTN1" or version!=1:
        raise ValueError("JTAG CNN signature/version mismatch")
    return dict(sequence=sequence,size=size,crc=crc,status=status,accepted=accepted,
                native_logits=(first,second),cycles=cycles,frame_id=frame,capture_ns=capture,eye=eye)


class JtagCNNClient:
    """One outstanding eye; frame metadata is checked against actual TDO fields."""
    payload_size = 3089
    expected_cycles = 367739
    make_packet = staticmethod(cnn_packet)
    parse_response = staticmethod(parse_cnn_reply)
    reverse_logits = True

    def _upload_scan(self, packet, capture):
        return self.session.scan(packet, capture)

    def _poll_scan(self):
        return self.session.scan(bytes(CNN_REPLY.size))
    def __init__(self,session,capture_payload=False):
        self.owner=threading.get_ident()
        if getattr(session,"owner",self.owner)!=self.owner:
            raise RuntimeError("JTAG client must use the session owner thread")
        self.session=session
        self.capture_payload=capture_payload
        self.sequence=0xfffffff0
        self.pending=None
        self.timeouts=0
        self.sequence_errors=self.crc_errors=self.status_rejections=0
        self.request_count=0
        self.last_upload_ms=0

    def send_eye(self,pixels,frame_id,capture_ns,eye):
        if threading.get_ident()!=self.owner:raise RuntimeError("JTAG client owner thread required")
        if self.pending is not None:
            raise RuntimeError("one outstanding JTAG eye credit")
        next_sequence=(self.sequence+1)&0xffffffff
        packet=self.make_packet(next_sequence,frame_id,capture_ns,eye,pixels)
        self.sequence=next_sequence
        body=packet[CNN_REPLY.size:-1]
        self.pending=dict(sequence=self.sequence,frame_id=frame_id,capture_ns=capture_ns,eye=eye,
                          crc=binascii.crc_hqx(body[:-2],0xffff),started=time.perf_counter_ns())
        self.request_count+=1
        try:
            measured=self._upload_scan(packet,self.capture_payload)
            if self.capture_payload:
                recovered=((int.from_bytes(measured["data"],"little")>>(CNN_REPLY.size*8+1))&((1<<(len(body)*8))-1)).to_bytes(len(body),"little")
                if recovered!=body:raise ValueError("CNN input echo mismatch")
            self.last_upload_ms=measured["wall_ms"]
            return {k:v for k,v in measured.items() if k!="data"}
        except Exception:
            self.pending=None
            raise

    def receive_eye(self):
        if threading.get_ident()!=self.owner:raise RuntimeError("JTAG client owner thread required")
        if self.pending is None:raise RuntimeError("no outstanding JTAG eye")
        expected=self.pending
        deadline=time.perf_counter()+self.session.timeout
        polls=[]
        try:
            while time.perf_counter()<deadline:
                measured=self._poll_scan()
                polls.append({k:v for k,v in measured.items() if k!="data"})
                reply=self.parse_response(measured["data"])
                if reply["size"]==0 and reply["status"] in (8,9):
                    continue # BOOT/BUSY snapshots have no accepted request identity.
                if reply["sequence"]!=expected["sequence"] or reply["size"]!=self.payload_size:
                    self.sequence_errors+=1
                    raise ValueError("stale, missing or out-of-order CNN response")
                if reply["crc"]!=expected["crc"]:
                    self.crc_errors+=1
                    raise ValueError("CNN input CRC association mismatch")
                if reply["status"]!=0:
                    self.status_rejections+=1
                    self.sequence=(self.sequence-1)&0xffffffff # explicitly rejected, not accepted
                    raise RuntimeError(f"explicit JTAG CNN status {reply['status']} sequence {reply['sequence']}")
                for field in ("frame_id","capture_ns","eye"):
                    if reply[field]!=expected[field]:raise ValueError(f"CNN {field} association mismatch")
                if self.expected_cycles is not None and reply["cycles"]!=self.expected_cycles:raise ValueError("CNN baseline cycle count changed")
                if reply["cycles"]<=0:raise ValueError("missing CNN cycle measurement")
                reply.update(logits=tuple(reversed(reply["native_logits"])) if self.reverse_logits else reply["native_logits"],polls=polls,
                             wall_ms=(time.perf_counter_ns()-expected["started"])/1e6)
                return reply
            self.timeouts+=1
            raise TimeoutError(f"missing JTAG CNN result sequence {expected['sequence']}")
        finally:
            self.pending=None

    def infer_pretrained(self,left,right,frame_id,capture_ns=None):
        from pretrained_eye import input_pixels
        from fpga_host import HardwareResult
        capture_ns=time.perf_counter_ns() if capture_ns is None else capture_ns
        started=time.perf_counter_ns()
        logits=[];cycles=[];upload=0;envelopes=[]
        for eye,image in enumerate((left,right)):
            before=time.perf_counter_ns()
            self.send_eye(input_pixels(image).tobytes(),frame_id,capture_ns,eye)
            upload+=self.last_upload_ms
            result=self.receive_eye()
            after=time.perf_counter_ns()
            logits.append(result["logits"]);cycles.append(result["cycles"])
            envelopes.append((before,after,result["cycles"]*20))
        return HardwareResult(frame_id,4,0,*logits,*cycles,sum(cycles),
            (time.perf_counter_ns()-started)/1e6,upload_ms=upload,kernel_envelopes=tuple(envelopes))
