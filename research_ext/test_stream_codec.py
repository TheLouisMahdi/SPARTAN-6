"""Offline lossless/reference recovery checks; never opens a device."""
import struct
import sys
from pathlib import Path
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from stream_codec import Decoder, encode, pack_runs, unpack_runs, RAW, ZERO, XOR_ZERO, HYBRID, EYE_BYTES
from research_ext.streaming_evaluation import model_stream


class StreamCodecTests(unittest.TestCase):
    def test_model_bounded_and_batch_deadline(self):
        result = model_stream(20, 4)
        self.assertLessEqual(result["max_occupancy"], 9)
        self.assertEqual(result["arrived"], result["completed"]+result["rejected"])
        self.assertEqual(result["unbounded_batch_fill_ms"], 60)
        self.assertEqual(result["trace"][0]["batch_wait_ms"], 20)
        self.assertGreater(result["rejected"], 0)

    def test_fidelity_and_raw_bound(self):
        import random
        rng = random.Random(1701)
        fixtures = [bytes(EYE_BYTES), bytes([255])*EYE_BYTES,
                    bytes(range(256))*12, rng.randbytes(EYE_BYTES)]
        for mode in (RAW, ZERO, XOR_ZERO, HYBRID):
            decoder = Decoder()
            reference = None
            for frame, data in enumerate(fixtures + fixtures[::-1]):
                record, _ = encode(data, 0, frame, frame*20_000_000, frame//4, mode, reference)
                result = decoder.decode(record)
                self.assertEqual(result["data"], data)
                self.assertEqual(result["frame_id"], frame)
                self.assertEqual(result["capture_ns"], frame*20_000_000)
                self.assertLessEqual(len(record), EYE_BYTES + 32)
                reference = (frame, data)

    def test_temporal_separate_eyes_and_wrap(self):
        decoder = Decoder()
        refs = [None, None]
        for frame in (0xfffffffe, 0xffffffff, 0, 1):
            for eye in (0, 1):
                data = bytes([eye+1])*EYE_BYTES
                record, _ = encode(data, eye, frame, 123, 0, XOR_ZERO, refs[eye])
                result = decoder.decode(record)
                self.assertEqual(data, result["data"])
                if refs[eye]:
                    self.assertLess(len(record), 100)
                refs[eye] = (frame, data)

    def test_missing_stale_gap_duplicate_and_recovery(self):
        data = bytes(EYE_BYTES)
        key, _ = encode(data, 0, 4, 0, 0, RAW)
        delta, _ = encode(data, 0, 5, 0, 0, XOR_ZERO, (4, data))
        decoder = Decoder()
        with self.assertRaises(ValueError):
            decoder.decode(delta)
        decoder.decode(key)
        with self.assertRaises(ValueError):
            decoder.decode(key)
        decoder.decode(key)
        gap, _ = encode(data, 0, 7, 0, 0, XOR_ZERO, (6, data))
        with self.assertRaises(ValueError):
            decoder.decode(gap)
        recovery, _ = encode(data, 0, 8, 0, 0, RAW)
        self.assertEqual(decoder.decode(recovery)["data"], data)

    def test_crc_truncation_metadata_and_no_partial_commit(self):
        good, _ = encode(bytes(EYE_BYTES), 1, 1, 0, 0, ZERO)
        damaged = bytearray(good)
        damaged[-1] ^= 1
        bad_length = bytearray(good)
        bad_length[26:28] = struct.pack("<H", 55)
        bad_length[-4:] = struct.pack("<I", zlib.crc32(bad_length[:-4]))
        for record in (good[:-1], bytes(damaged), bytes(bad_length), b"", good+b"x"):
            decoder = Decoder()
            with self.assertRaises(ValueError):
                decoder.decode(record)
            self.assertEqual(decoder.references, {})

    def test_token_bounds_and_adversarial_runs(self):
        for repeat in (False, True):
            for count in (1, 2, 3, 127, 128, 129, 255):
                data = (bytes([0])*count + bytes([7])*count)* (3072//(count*2)+1)
                data = data[:3072]
                self.assertEqual(unpack_runs(pack_runs(data, repeat), repeat)[0], data)
            for payload in (b"\x00", b"\xff"*25, b"\x7f\x01", b"\xff\x01"*25):
                with self.assertRaises(ValueError):
                    unpack_runs(payload, repeat)


if __name__ == "__main__":
    unittest.main()
