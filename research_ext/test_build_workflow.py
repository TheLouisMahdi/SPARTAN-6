"""Artifact validation only; no ISE process, JTAG or camera is opened."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import rtl_evidence


class BuildWorkflowTests(unittest.TestCase):
    def test_p1_can_be_selected_when_the_default_top_is_p2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "p1"
            result = rtl_evidence.build_variant(output, 1, ise=root / "uninstalled")
            self.assertEqual(result["status"], "NOT_RUN_MISSING_DEPENDENCY")
            self.assertIn("parameter integer PE_COUNT=1", (output / "rtl/eye_top.v").read_text())
            self.assertIn("parameter integer PRETRAINED=0", (output / "rtl/eye_top.v").read_text())
            self.assertIn("reg model_loaded=PRETRAINED", (output / "rtl/command_controller.v").read_text())
            self.assertTrue((output / "rtl/crc16_function.vh").is_file())

    def test_program_rejects_changed_bit_before_any_tool_is_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "eye_top.bit").write_bytes(b"changed")
            (output / "result.json").write_text(json.dumps({"status": "PASS", "timing_met": True,
                "bit_sha256": hashlib.sha256(b"original").hexdigest()}))
            with patch("rtl_evidence.subprocess.run", side_effect=AssertionError("No tool should run")):
                with self.assertRaisesRegex(ValueError, "BIT hash"):
                    rtl_evidence.program_variant(output)

    def test_program_rejects_stale_rtl_before_any_tool_is_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            payload = b"bit fixture"
            (output / "eye_top.bit").write_bytes(payload)
            (output / "result.json").write_text(json.dumps({"status": "PASS", "timing_met": True,
                "bit_sha256": hashlib.sha256(payload).hexdigest(), "source_sha256": {"rtl/eye_top.v": "stale"}}))
            with patch("rtl_evidence.subprocess.run", side_effect=AssertionError("No tool should run")):
                with self.assertRaisesRegex(ValueError, "sources changed"):
                    rtl_evidence.program_variant(output)


if __name__ == "__main__":
    unittest.main()
