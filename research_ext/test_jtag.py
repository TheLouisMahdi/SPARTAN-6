"""Offline USER1 framing, backend lifecycle and asynchronous RTL regressions."""
import binascii
import io
from pathlib import Path
import queue
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from jtag_host import JtagSession, JtagCNNClient, CNN_REPLY, cnn_packet, probe_packet, parse_reply, parse_cnn_reply, recover_echo
from fpga_host import crc16


class JtagTests(unittest.TestCase):
    def test_byte_and_bit_order_crc_and_limits(self):
        for n in (0,32,256,1024,3072,6144):
            payload=bytes((i*73+19)%256 for i in range(n))
            packet=probe_packet(0xfffffffe,payload)
            self.assertEqual(packet[16:24],struct.pack("<BBIH",1,1,0xfffffffe,n))
            self.assertEqual(struct.unpack("<H",packet[-3:-1])[0],crc16(packet[16:-3]))
            captured=(int.from_bytes(packet[16:-1],"little")<<129).to_bytes(len(packet),"little")
            self.assertEqual(recover_echo(captured,packet),packet[16:-1])
        with self.assertRaises(ValueError):probe_packet(0,bytes(6145))
        with self.assertRaises(ValueError):probe_packet(0,b"x",command=3)

    def test_reply_signature_identity_and_stale_protection(self):
        data=struct.pack("<4sIHHBBH",b"JTG1",0xffffffff,3072,123,0,1,7)
        self.assertEqual(parse_reply(data)["sequence"],0xffffffff)
        for bad in (data[:10],b"BAD!"+data[4:],data[:13]+b"\x02"+data[14:]):
            with self.assertRaises(ValueError):parse_reply(bad)

    def test_explicit_mode_and_thread_ownership(self):
        with self.assertRaises(ValueError):JtagSession("unused.log")
        session=object.__new__(JtagSession)
        session.closed=False;session.owner=threading.get_ident()+1
        with self.assertRaises(RuntimeError):session._request("SCAN","JTSCAN")

    def test_backend_errors_missing_and_timeout(self):
        session=object.__new__(JtagSession)
        session.lines=queue.Queue();session.timeout=.001;session.number=4
        for value,error in (("JTSCAN\t3",RuntimeError),("JTERROR\t4\t626164",RuntimeError),(None,RuntimeError)):
            session.lines.put(value)
            with self.assertRaises(error):session._receive("JTSCAN")
        with self.assertRaises(TimeoutError):session._receive("JTSCAN")

    def test_backend_cleanup(self):
        class Process:
            def __init__(self):self.stdin=io.StringIO();self.stdout=io.StringIO("JTREADY\t1500000\n")
            def poll(self):return 0
        process=Process()
        with tempfile.TemporaryDirectory() as folder, mock.patch("jtag_host.XSDB") as path, mock.patch("jtag_host.subprocess.Popen",return_value=process):
            path.is_file.return_value=True
            session=JtagSession(Path(folder)/"session.log",experimental_probe=True)
            self.assertEqual(session.frequency,1500000)
            session.close();session.close()
            self.assertTrue(session.log.closed)
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.stdout.closed)
            self.assertFalse(session.reader.is_alive())

    def test_asynchronous_rtl(self):
        from research_ext.jtag_evaluation import simulate
        with tempfile.TemporaryDirectory() as folder:
            simulate(Path(folder)/"probe")

    def test_cnn_metadata_logits_and_sequence_wrap(self):
        class Session:
            timeout=.02
            def scan(self,data,capture=True):
                if len(data)>CNN_REPLY.size:
                    body=data[CNN_REPLY.size:-1]
                    version,command,sequence,length=struct.unpack_from("<BBIH",body)
                    frame,stamp,eye,model=struct.unpack_from("<IQBI",body,8)
                    self.response=CNN_REPLY.pack(b"JTN1",sequence,length,binascii.crc_hqx(body[:-2],0xffff),
                        0,1,1,100+eye,-40,367739,frame,stamp,eye)
                    return {"data":b"","wall_ms":1}
                return {"data":self.response,"wall_ms":1}
        client=JtagCNNClient(Session())
        client.sequence=0xfffffffe
        for frame,stamp,eye in ((0xffffffff,123456789,0),(0,143456789,1)):
            client.send_eye(bytes(3072),frame,stamp,eye)
            with self.assertRaises(RuntimeError):client.send_eye(bytes(3072),frame,stamp,eye)
            result=client.receive_eye()
            self.assertEqual(result["logits"],(-40,100+eye))
            self.assertEqual(result["capture_ns"],stamp)
        self.assertEqual(client.sequence,0)
        sequence=client.sequence
        with self.assertRaises(ValueError):client.send_eye(bytes(10),1,2,0)
        self.assertEqual(client.sequence,sequence)

    def test_cnn_corrupted_association_and_timeout(self):
        class Session:
            timeout=.001
            def scan(self,data,capture=True):return {"data":bytes(41),"wall_ms":1}
        client=JtagCNNClient(Session())
        client.send_eye(bytes(3072),7,8,0)
        with self.assertRaises(ValueError):client.receive_eye()
        self.assertIsNone(client.pending)
        with self.assertRaises(ValueError):parse_cnn_reply(bytes(41))
        with self.assertRaises(ValueError):cnn_packet(1,2,3,2,bytes(3072))

    def test_cnn_missing_result_is_bounded(self):
        class Session:
            timeout=.001
            def scan(self,data,capture=True):
                return {"data":CNN_REPLY.pack(b"JTN1",0,0,0,8,1,0,0,0,0,0,0,0),"wall_ms":0}
        client=JtagCNNClient(Session())
        client.send_eye(bytes(3072),1,2,0)
        with self.assertRaises(TimeoutError):client.receive_eye()
        self.assertEqual(client.timeouts,1)
        self.assertIsNone(client.pending)

    def test_cnn_rtl_logits_invalid_inputs_and_metadata(self):
        from research_ext.jtag_evaluation import simulate_cnn
        with tempfile.TemporaryDirectory() as folder:
            simulate_cnn(Path(folder)/"cnn")

    def test_missing_frame_stream_has_ordered_terminal_records(self):
        from research_ext.jtag_benchmark import measure
        def infer(*args):raise TimeoutError("injected missing hardware result")
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            with self.assertRaises(TimeoutError):measure(output,"missing",infer,[("synthetic",None)],[(0,1)],count=20)
            import json
            rows=[json.loads(line) for line in (output/"missing/ordered_results.jsonl").read_text().splitlines()]
            arrived=[json.loads(line) for line in (output/"missing/arrivals.jsonl").read_text().splitlines()]
            self.assertEqual([r["index"] for r in rows],list(range(len(arrived))))
            self.assertEqual(rows[0]["status"],"MISSING")
            self.assertFalse(any(r["status"]=="VALID" for r in rows))


if __name__=="__main__":unittest.main()
