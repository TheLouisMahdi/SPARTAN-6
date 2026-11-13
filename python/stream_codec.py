"""Experimental offline codec reference. Not a negotiated AX309 wire protocol.

One record describes one 3072-byte CHW BGR eye. CRC covers metadata and data.
References are separate per eye and advance only after a validated record.
"""
import struct
import zlib

RAW, ZERO, XOR_ZERO, HYBRID = range(4)
HEADER = struct.Struct("<BBBBIQIIHH")
EYE_BYTES = 3072
OVERHEAD = HEADER.size + 4


def pack_runs(data, repeat=False):
    out = bytearray()
    i = 0
    while i < len(data):
        end = i + 1
        if repeat or data[i] == 0:
            while end < min(len(data), i + 128) and data[end] == data[i]:
                end += 1
        if (repeat and end - i >= 3) or (not repeat and data[i] == 0):
            out.append(128 | (end - i - 1))
            if repeat:
                out.append(data[i])
            i = end
        else:
            start = i
            i += 1
            while i < min(len(data), start + 128):
                if not repeat and data[i] == 0:
                    break
                if repeat and i + 2 < len(data) and data[i] == data[i+1] == data[i+2]:
                    break
                i += 1
            out.append(i - start - 1)
            out.extend(data[start:i])
    return bytes(out)


def unpack_runs(data, repeat=False):
    out = bytearray()
    i = operations = 0
    while i < len(data):
        token = data[i]
        count = (token & 127) + 1
        i += 1
        operations += 1
        if len(out) + count > EYE_BYTES:
            raise ValueError("decoded length overflow")
        if token & 128:
            if repeat:
                if i == len(data):
                    raise ValueError("truncated repeat")
                value = data[i]
                i += 1
            else:
                value = 0
            out.extend(bytes([value]) * count)
        else:
            if i + count > len(data):
                raise ValueError("truncated literal")
            out.extend(data[i:i+count])
            i += count
        operations += count
    if len(out) != EYE_BYTES:
        raise ValueError("decoded length mismatch")
    return bytes(out), operations


def encode(data, eye, frame_id, capture_ns, batch_id, mode, reference=None, key=False):
    """Return (record, candidate size before RAW fallback). Reference=(id, bytes)."""
    if len(data) != EYE_BYTES or eye not in (0, 1):
        raise ValueError("expected one exact eye tensor and eye 0/1")
    if mode not in (RAW, ZERO, XOR_ZERO, HYBRID):
        raise ValueError("unsupported codec")
    ref_id = 0
    selected = mode
    if key or (mode == XOR_ZERO and reference is None):
        selected = RAW
    if selected == XOR_ZERO:
        ref_id, previous = reference
        if len(previous) != EYE_BYTES or frame_id != ((ref_id + 1) & 0xffffffff):
            raise ValueError("encoder reference identity mismatch")
        payload = pack_runs(bytes(a ^ b for a, b in zip(data, previous)))
    elif selected == RAW:
        payload = data
    else:
        payload = pack_runs(data, repeat=selected == HYBRID)
    candidate_size = len(payload)
    if len(payload) >= EYE_BYTES:
        selected, payload, ref_id = RAW, data, 0
    flags = int(selected != XOR_ZERO)  # explicit independent keyframe
    header = HEADER.pack(1, eye, selected, flags, frame_id, capture_ns,
                         batch_id, ref_id, EYE_BYTES, len(payload))
    body = header + payload
    return body + struct.pack("<I", zlib.crc32(body)), candidate_size


class Decoder:
    def __init__(self):
        self.references = {}

    def reset(self):
        self.references.clear()

    def decode(self, record):
        """Reject corruption before any reference commit; errors require fresh keys."""
        try:
            return self._decode(record)
        except (ValueError, struct.error):
            self.reset()
            raise ValueError("invalid record; explicit keyframes required") from None

    def _decode(self, record):
        if len(record) < OVERHEAD or len(record) > OVERHEAD + EYE_BYTES:
            raise ValueError("record length")
        version, eye, codec, flags, frame, capture, batch, ref, raw_size, size = HEADER.unpack_from(record)
        if version != 1 or eye not in (0, 1) or codec not in range(4):
            raise ValueError("version/eye/codec")
        if raw_size != EYE_BYTES or len(record) != OVERHEAD + size:
            raise ValueError("length")
        if zlib.crc32(record[:-4]) != struct.unpack_from("<I", record, len(record)-4)[0]:
            raise ValueError("CRC")
        if flags != int(codec != XOR_ZERO) or (codec != XOR_ZERO and ref != 0):
            raise ValueError("keyframe flags")
        previous = self.references.get(eye)
        if previous:
            distance = (frame - previous[0]) & 0xffffffff
            if not 0 < distance < 0x80000000:
                raise ValueError("duplicate or old identity")
        payload = record[HEADER.size:-4]
        if codec == RAW:
            if size != EYE_BYTES:
                raise ValueError("RAW length")
            decoded, operations = payload, EYE_BYTES
        else:
            decoded, operations = unpack_runs(payload, repeat=codec == HYBRID)
        if codec == XOR_ZERO:
            if previous is None or previous[0] != ref or frame != ((ref + 1) & 0xffffffff):
                raise ValueError("missing or stale reference")
            decoded = bytes(a ^ b for a, b in zip(decoded, previous[1]))
            operations += EYE_BYTES
        self.references[eye] = (frame, decoded)
        return {"eye": eye, "frame_id": frame, "capture_ns": capture,
                "batch_id": batch, "codec": codec, "data": decoded,
                "decode_operations": operations}
