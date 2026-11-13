"""AX309 UART framing and the fixed TinyEye integer model image."""

from binascii import crc_hqx
from dataclasses import dataclass
import struct
import time
from threading import get_ident

import numpy as np

METHOD_IDS = {"grayscale8": 0, "grayscale2": 1, "sobel2": 2, "aeee2": 3}
MODEL_BYTES = 1208
MAGIC = b"\xa5\x5a"


def crc16(data):
    return crc_hqx(data, 0xFFFF)


def encode_packet(kind, sequence, payload=b""):
    if len(payload) > 2048:
        raise ValueError("Packet payload exceeds 2048 bytes")
    body = struct.pack("<BBIH", 1, kind, sequence, len(payload)) + payload
    return MAGIC + body + struct.pack("<H", crc16(body))


class PacketReader:
    """Incremental parser; reject CRC/length/version failures and search again."""

    def __init__(self):
        self.pending = bytearray()
        self.bad_packets = 0

    def feed(self, data):
        self.pending.extend(data)
        packets = []
        while True:
            start = self.pending.find(MAGIC)
            if start < 0:
                self.pending[:] = self.pending[-1:] if self.pending[-1:] == MAGIC[:1] else b""
                break
            del self.pending[:start]
            if len(self.pending) < 10:
                break
            version, kind, sequence, length = struct.unpack_from("<BBIH", self.pending, 2)
            if version != 1 or length > 2048:
                self.bad_packets += 1
                del self.pending[0]
                continue
            size = 12 + length
            if len(self.pending) < size:
                break
            expected = struct.unpack_from("<H", self.pending, size - 2)[0]
            if crc16(self.pending[2:size - 2]) != expected:
                self.bad_packets += 1
                del self.pending[0]
                continue
            packets.append((kind, sequence, bytes(self.pending[10:size - 2])))
            del self.pending[:size]
        return packets


def model_image(params):
    result = bytearray()
    shapes = ((4, 1, 3, 3), (8, 4, 3, 3), (8, 96), (2, 8))
    for layer, shape in zip(("conv1", "conv2", "fc1", "fc2"), shapes):
        weights = params[layer + "_weight"]
        if weights.dtype != np.int8 or weights.shape != shape:
            raise ValueError(f"Unexpected integer weight shape/type: {layer}")
        bias = params[layer + "_bias"]
        if bias.dtype != np.dtype("int32") or bias.shape != (shape[0],):
            raise ValueError(f"Unexpected integer bias shape/type: {layer}")
        bound = 255 * np.abs(weights.astype(np.int64)).reshape(shape[0], -1).sum(axis=1) + np.abs(bias.astype(np.int64))
        if np.any(bound > np.iinfo(np.int32).max):
            raise ValueError(f"INT32 accumulator bound exceeded: {layer}")
        result.extend(weights.tobytes(order="C"))
    for layer in ("conv1", "conv2", "fc1", "fc2"):
        bias = params[layer + "_bias"]
        if bias.dtype != np.dtype("int32"):
            raise ValueError("Expected INT32 biases")
        result.extend(bias.astype("<i4").tobytes())
    for layer in ("conv1", "conv2", "fc1"):
        multiplier = int(params[layer + "_multiplier"])
        if not 0 < multiplier <= 0xFFFFFFFF:
            raise ValueError("FPGA supports positive uint32 Q24 multipliers")
        result.extend(struct.pack("<I", multiplier))
    if len(result) != MODEL_BYTES:
        raise ValueError("Incorrect model image length")
    return bytes(result)


def eye_payload(method, left, right):
    """Inputs are this method's ordinal codes: uint8 (16,32), never re-encoded."""
    pixels = []
    for eye in (left, right):
        if eye.dtype != np.uint8 or eye.shape != (16, 32):
            raise ValueError("Expected canonical uint8 16x32 encoded eye")
        pixels.append(eye.reshape(-1))
    values = np.concatenate(pixels)
    if method == "grayscale8":
        payload = values.tobytes()
    else:
        if np.any(values > 3):
            raise ValueError("Two-bit codes must be in 0..3")
        four = values.reshape(-1, 4)
        payload = ((four[:, 0] << 6) | (four[:, 1] << 4) | (four[:, 2] << 2) | four[:, 3]).tobytes()
    return bytes([METHOD_IDS[method]]) + payload


@dataclass
class HardwareResult:
    frame_id: int
    method_id: int
    flags: int
    left_logits: tuple
    right_logits: tuple
    left_cycles: int
    right_cycles: int
    pair_cycles: int
    roundtrip_ms: float
    blink_flags: int = 0
    left_blinks: int | None = None
    right_blinks: int | None = None
    upload_ms: float | None = None
    kernel_envelopes: tuple | None = None


class FPGAClient:
    def __init__(self, port, baud=115200, timeout=2.0):
        if not port or not port.strip():
            raise ValueError("A UART port is required")
        import serial
        self.serial = serial.Serial(port, baud, timeout=0.02, write_timeout=timeout)
        self.timeout = timeout
        self.reader = PacketReader()
        self.sequence = 0
        self.owner = get_ident()
        self.request_count = self.timeout_count = self.unmatched_responses = 0
        self.serial.reset_input_buffer()

    def close(self):
        self.serial.close()

    def request(self, kind, payload=b"", sequence=None):
        if get_ident() != self.owner:
            raise RuntimeError("UART requests must remain on the connection owner thread")
        if sequence is None:
            self.sequence = (self.sequence + 1) & 0xFFFFFFFF
            sequence = self.sequence
        started = time.perf_counter_ns()
        self.request_count = getattr(self, "request_count", 0) + 1
        self.serial.write(encode_packet(kind, sequence, payload))
        deadline = started + self.timeout * 1_000_000_000
        while time.perf_counter_ns() < deadline:
            chunk = self.serial.read(max(1, min(self.serial.in_waiting, 2048)))
            for reply_kind, reply_sequence, reply in self.reader.feed(chunk):
                if reply_kind != kind | 0x80 or reply_sequence != sequence:
                    self.unmatched_responses = getattr(self, "unmatched_responses", 0) + 1
                    continue
                if not reply or reply[0] != 0:
                    raise RuntimeError(f"FPGA status error: {reply.hex()}")
                return reply, (time.perf_counter_ns() - started) / 1e6
        self.reader.pending.clear()
        self.timeout_count = getattr(self, "timeout_count", 0) + 1
        raise TimeoutError(f"No matching FPGA reply for command 0x{kind:02X}, sequence {sequence}")

    def ping(self):
        payload, elapsed = self.request(1)
        if len(payload) != 12 or payload[4:8] != b"AEEE":
            raise RuntimeError("Unexpected AX309 hardware signature")
        return {"method_id": payload[1], "model_loaded": bool(payload[2]),
                "timed_blinks": bool(payload[3] & 0x80),
                "binocular_timing": bool(payload[3] & 0x40),
                "fingerprint": struct.unpack_from("<I", payload, 8)[0], "roundtrip_ms": elapsed}

    def load_model(self, predictor):
        if predictor.method == "omz_bgr32":
            status = self.ping()
            if (status["method_id"] != 4 or not status["model_loaded"]
                    or status["fingerprint"] != int(predictor.checkpoint_sha256[:8], 16)):
                raise RuntimeError("Program the matching pretrained CNN bitstream before inference")
            return status
        fingerprint = int(predictor.checkpoint_sha256[:8], 16)
        payload = bytes([METHOD_IDS[predictor.method]]) + struct.pack("<I", fingerprint) + model_image(predictor.params)
        self.request(2, payload)
        status = self.ping()
        if status["method_id"] != METHOD_IDS[predictor.method] or status["fingerprint"] != fingerprint:
            raise RuntimeError("Loaded model fingerprint mismatch")
        return status

    def infer_pretrained(self, left, right, frame_id):
        """Lossless CHW BGR input in three CRC-protected chunks per eye."""
        from pretrained_eye import input_pixels, canonical_logits
        logits, cycles = [], []
        started = time.perf_counter_ns()
        upload_ns = 0
        envelopes = []
        for roi in (left, right):
            data = input_pixels(roi).tobytes()
            upload_started = time.perf_counter_ns()
            for offset in range(0, len(data), 1024):
                self.request(0x20, struct.pack("<H", offset) + data[offset:offset+1024])
            upload_ns += time.perf_counter_ns() - upload_started
            command_start = time.perf_counter_ns()
            payload, _ = self.request(0x21)
            command_end = time.perf_counter_ns()
            if len(payload) != 17:
                raise RuntimeError("Unexpected pretrained CNN reply length")
            native_first, native_second, elapsed, fingerprint = struct.unpack_from("<iiII", payload, 1)
            if fingerprint != 0x4DAA1000:
                raise RuntimeError("Pretrained CNN fingerprint mismatch")
            logits.append(tuple(int(v) for v in canonical_logits((native_first, native_second))))
            cycles.append(elapsed)
            envelopes.append((command_start,command_end,elapsed*20))
        return HardwareResult(frame_id, 4, 0, *logits, *cycles, sum(cycles),
                              (time.perf_counter_ns()-started)/1e6, upload_ms=upload_ns/1e6,
                              kernel_envelopes=tuple(envelopes))

    def configure(self, left_threshold=0, right_threshold=0):
        for threshold in (left_threshold, right_threshold):
            if not -(1 << 31) <= threshold < (1 << 31):
                raise ValueError("Hardware manual thresholds must fit signed INT32")
        self.request(3, struct.pack("<ii", left_threshold, right_threshold))

    def invalidate(self, frame_id):
        self.request(0x11, sequence=frame_id & 0xFFFFFFFF)

    def reset_blinks(self):
        self.request(0x13)

    def eye_timing(self, left, right, timestamp_ms, frame_id):
        """Optional expression coefficients + uint32 capture ms -> FPGA closure timing, RTT ms."""
        from blink_timing import quantize_score, TimingResult
        if not 0 <= timestamp_ms <= 0xffffffff:
            raise ValueError("Capture timestamp must fit uint32 milliseconds")
        scores = [quantize_score(left), quantize_score(right)]
        valid = sum((score is not None) << side for side, score in enumerate(scores))
        request = struct.pack("<IBHH", timestamp_ms, valid, scores[0] or 0, scores[1] or 0)
        reply, elapsed = self.request(0x14, request, frame_id & 0xffffffff)
        if len(reply) != 36 or reply[1] != 1:
            raise RuntimeError("Unexpected binocular timing reply")
        return TimingResult(reply[2], reply[3] & 3, bool(reply[3] & 0x80),
                            *struct.unpack_from("<8I", reply, 4)), elapsed

    def read_model(self):
        return self.read_memory(0,MODEL_BYTES)

    def read_memory(self,start,size):
        image=bytearray()
        for address in range(start,start+size,16):
            reply,_=self.request(5,struct.pack("<H",address))
            if len(reply)!=17:raise RuntimeError("Invalid model readback response")
            image.extend(reply[1:])
        return bytes(image[:size])

    def infer(self, method, left, right, frame_id, capture_ms=None):
        request = eye_payload(method, left, right)
        if capture_ms is not None:
            if not 0 <= capture_ms <= 0xFFFFFFFF:
                raise ValueError("Capture timestamp must fit unsigned 32-bit milliseconds")
            request = request[:1] + struct.pack("<I", capture_ms) + request[1:]
        payload, elapsed = self.request(0x10 if capture_ms is None else 0x12, request, frame_id & 0xFFFFFFFF)
        if len(payload) != (32 if capture_ms is None else 40):
            raise RuntimeError("Invalid hardware result length")
        status, method_id, flags, reserved, lo, lc, ro, rc, left_cycles, right_cycles, pair_cycles = struct.unpack_from("<BBBBiiiiIII", payload)
        if method_id != METHOD_IDS[method]:
            raise RuntimeError("Hardware replied with the wrong encoding method")
        counts = (None, None) if capture_ms is None else struct.unpack_from("<II", payload, 32)
        return HardwareResult(frame_id & 0xFFFFFFFF, method_id, flags, (lo, lc), (ro, rc),
                              left_cycles, right_cycles, pair_cycles, elapsed,
                              reserved if capture_ms is not None else 0, *counts)
