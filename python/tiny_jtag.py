"""Compact Grayscale2 USER1 contract, distinct from the verified BGR contract.

128 packed bytes per eye; native logits are OPEN,CLOSED. The reply associates
the request CRC and metadata but does not carry an independent reply CRC.
"""
import binascii
import struct
import time

from jtag_host import HEADER, CNN_REPLY, JtagCNNClient


def tiny_packet(sequence, frame_id, capture_ns, eye, pixels, model_id, encoder=1):
    if len(pixels) != 128 or eye not in (0, 1) or encoder != 1:
        raise ValueError("expected 128 packed Grayscale2 bytes and eye 0/1")
    payload = struct.pack("<IQBIB", frame_id, capture_ns, eye, model_id, encoder) + pixels
    body = HEADER.pack(2, 2, sequence, len(payload)) + payload
    return bytes(CNN_REPLY.size) + body + struct.pack("<H", binascii.crc_hqx(body, 0xffff)) + b"\0"


def parse_tiny_reply(data):
    if len(data) != CNN_REPLY.size:
        raise ValueError("unexpected compact reply size")
    signature, sequence, size, crc, status, version, accepted, first, second, cycles, frame, capture, eye = CNN_REPLY.unpack(data)
    if signature != b"JTY1" or version != 2:
        raise ValueError("compact firmware signature/version mismatch")
    return dict(sequence=sequence, size=size, crc=crc, status=status, accepted=accepted,
                native_logits=(first, second), cycles=cycles, frame_id=frame, capture_ns=capture, eye=eye)


class JtagTinyClient(JtagCNNClient):
    supports_capture_ns = True
    payload_size = 146
    expected_cycles = None  # Actual counter is reported; physical validation is separate.
    reverse_logits = False
    parse_response = staticmethod(parse_tiny_reply)

    def __init__(self, session, model_fingerprint, capture_payload=False, compound=True):
        self.model_fingerprint = model_fingerprint
        self.compound = compound
        self._combined_response = None
        super().__init__(session, capture_payload)

    def _upload_scan(self, packet, capture):
        if self.compound and not capture and hasattr(self.session, "exchange_eye"):
            measured = self.session.exchange_eye(packet)
            self._combined_response = measured
            return dict(data=b"", wall_ms=measured["upload_ms"], upload_provenance=measured["upload_provenance"])
        return super()._upload_scan(packet, capture)

    def _poll_scan(self):
        if self._combined_response is not None:
            measured, self._combined_response = self._combined_response, None
            return measured
        return super()._poll_scan()

    def make_packet(self, sequence, frame_id, capture_ns, eye, pixels):
        return tiny_packet(sequence, frame_id, capture_ns, eye, pixels, self.model_fingerprint)

    def infer(self, method, left, right, frame_id, capture_ms=None, *, capture_ns=None):
        from fpga_host import HardwareResult, eye_payload
        if method != "grayscale2":
            raise ValueError("compact firmware requires the trained Grayscale2 encoder")
        packed = eye_payload(method, left, right)[1:]
        stamp = capture_ns if capture_ns is not None else (time.perf_counter_ns() if capture_ms is None else int(capture_ms * 1_000_000))
        started = time.perf_counter_ns()
        logits, cycles, envelopes = [], [], []
        upload = 0
        for side in (0, 1):
            before = time.perf_counter_ns()
            self.send_eye(packed[side*128:(side+1)*128], frame_id, stamp, side)
            upload += self.last_upload_ms
            result = self.receive_eye()
            after = time.perf_counter_ns()
            logits.append(result["logits"])
            cycles.append(result["cycles"])
            envelopes.append((before, after, result["cycles"] * 20))
        return HardwareResult(frame_id, 1, 0, *logits, *cycles, sum(cycles),
                              (time.perf_counter_ns()-started)/1e6,
                              upload_ms=upload, kernel_envelopes=tuple(envelopes))
