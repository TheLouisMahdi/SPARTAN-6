"""Pretrained deployment contracts; no network download, camera or fitting."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from fpga_host import FPGAClient
from live_pipeline import EyePipeline, live_cli
from pretrained_eye import PretrainedEye, read_model, input_pixels, prepare, DIRECTORY


class PretrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PretrainedEye()

    def test_rejects_changed_official_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "model.onnx").write_bytes(b"wrong weights")
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_model(temporary)

    def test_explicit_prepare_repairs_corruption_only_after_checksum_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fpga/generated").mkdir(parents=True)
            directory = root / "models"
            directory.mkdir()
            (directory / "model.onnx").write_bytes(b"corrupt cache")
            (directory / "LICENSE").write_bytes((DIRECTORY / "LICENSE").read_bytes())
            with patch("pretrained_eye.ROOT",root), patch("pretrained_eye.urllib.request.urlopen",return_value=io.BytesIO(b"bad download")):
                with self.assertRaisesRegex(ValueError,"SHA384"):
                    prepare(directory)
            self.assertEqual((directory / "model.onnx").read_bytes(),b"corrupt cache")
            raw = read_model()
            with patch("pretrained_eye.ROOT",root), patch("pretrained_eye.urllib.request.urlopen",return_value=io.BytesIO(raw)):
                prepare(directory)
            self.assertEqual(read_model(directory),raw)

    def test_color_order_and_normalization_contract(self):
        roi = np.full((32,32,3), [0,127,255], np.uint8)
        pixels = input_pixels(roi)
        self.assertEqual(pixels[:,0,0].tolist(), [0,127,255])
        for invalid in (np.zeros((16,32),np.uint8), np.zeros((32,32,3),np.float32)):
            with self.assertRaises(ValueError):
                input_pixels(invalid)

    def test_parallel_cpu_nets_match_and_keep_fractional_scores(self):
        roi = np.full((32,32,3),127,np.uint8)
        expected = self.model.predict(roi)
        self.assertTrue(np.any(expected != expected.astype(np.int32)))
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.model.predict, roi) for _ in range(4)]
            for future in futures:
                np.testing.assert_array_equal(future.result(), expected)
        pipeline = EyePipeline(self.model)
        try:
            result = pipeline.predict(7,1,{"left":roi,"right":roi})
            np.testing.assert_array_equal(result.eyes["left"]["logits"], expected)
            self.assertEqual(result.eyes["left"]["state"], "OPEN")
            missing = pipeline.predict(8,2,{"left":roi})
            self.assertEqual(missing.eyes["right"]["state"], "UNAVAILABLE")
        finally:
            pipeline.close()

    def test_fixed_conversion_tracks_original_decision(self):
        for value in (0,127,255):
            roi = np.full((32,32,3),value,np.uint8)
            floating = self.model.predict(roi)
            fixed = self.model.predict(roi,fixed=True)
            self.assertEqual(np.argmax(floating),np.argmax(fixed))
            np.testing.assert_allclose(fixed/16384,floating,atol=.004,rtol=0)

    def test_uart_upload_is_lossless_and_checks_firmware_identity(self):
        client = FPGAClient.__new__(FPGAClient)
        requests = []
        def request(kind,payload=b"",sequence=None):
            requests.append((kind,payload))
            return ((b"\0" if kind==0x20 else b"\0"+struct.pack("<iiII",-4,8,100,0x4DAA1000)),0)
        client.request = request
        roi = np.arange(3072,dtype=np.uint8).reshape(32,32,3)
        result = client.infer_pretrained(roi,roi,9)
        self.assertEqual(result.left_logits,(-4,8))
        self.assertEqual(result.pair_cycles,200)
        self.assertEqual([struct.unpack_from("<H",p)[0] for k,p in requests[:3]],[0,1024,2048])
        self.assertEqual(b"".join(p[2:] for k,p in requests[:3]),input_pixels(roi).tobytes())
        with patch.object(client,"ping",return_value={"method_id":0,"model_loaded":False,"fingerprint":0}):
            with self.assertRaisesRegex(RuntimeError,"matching pretrained"):
                client.load_model(self.model)

    def test_default_is_ready_without_devices(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(live_cli(["--status"]),0)
        self.assertIn("open-closed-eye-0001",output.getvalue())


if __name__ == "__main__":
    unittest.main()
