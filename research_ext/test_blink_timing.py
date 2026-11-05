"""Observed-duration and UART contract tests, independent of faces and model accuracy."""
from dataclasses import asdict
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from blink_timing import BinocularTimer, quantize_score
from fpga_host import FPGAClient
from verify_timing import golden_rows


class TimingTests(unittest.TestCase):
    def test_fast_one_closed_sample_has_33ms_duration(self):
        timer=BinocularTimer()
        timer.update(0,0,0);timer.update(1,1,33)
        result=timer.update(0,0,66)
        self.assertTrue(result.event)
        self.assertEqual((result.count,result.last_left_ms,result.last_right_ms,result.last_overlap_ms),(1,33,33,33))

    def test_current_elapsed_and_slow_duration_are_retained(self):
        timer=BinocularTimer();timer.update(0,0,0);timer.update(1,1,33)
        for t in range(133,2034,100):result=timer.update(1,1,t)
        self.assertEqual(result.left_current_ms,2000)
        result=timer.update(0,0,2066)
        self.assertEqual((result.count,result.long_count,result.last_duration_ms),(1,1,2033))

    def test_asymmetric_eyes_report_durations_and_overlap(self):
        timer=BinocularTimer()
        for t,l,r in ((0,0,0),(33,1,0),(66,1,1),(99,0,1),(132,0,0)):
            result=timer.update(l,r,t)
        self.assertEqual((result.count,result.last_left_ms,result.last_right_ms,result.last_overlap_ms,result.last_duration_ms),(1,66,66,33,99))

    def test_all_declared_boundary_cases(self):
        self.assertEqual(len({row['case'] for row in golden_rows()}),18)

    def test_repeated_time_rejected_and_missing_score_invalid(self):
        timer=BinocularTimer();timer.update(0,0,1)
        with self.assertRaises(ValueError):timer.update(1,1,1)
        result=timer.update(None,None,33)
        self.assertEqual(result.valid_mask,0)
        self.assertIsNone(quantize_score(float('nan')))

    def test_exact_uart_encoding_and_result(self):
        expected=BinocularTimer().update(0,0,33)
        words=[expected.count,expected.long_count,expected.left_current_ms,expected.right_current_ms,
               expected.last_left_ms,expected.last_right_ms,expected.last_overlap_ms,expected.last_duration_ms]
        reply=struct.pack('<4B8I',0,1,expected.closed_mask,expected.valid_mask,*words)
        client=object.__new__(FPGAClient)
        with patch.object(client,'request',return_value=(reply,2.0)) as request:
            actual,rtt=client.eye_timing(0,0,33,7)
        self.assertEqual(asdict(actual),asdict(expected))
        self.assertEqual(rtt,2.0)
        request.assert_called_once_with(0x14,struct.pack('<IBHH',33,3,0,0),7)

    def test_timed_cnn_reply_decodes_its_extended_payload(self):
        import numpy as np
        reply=struct.pack('<BBBBiiiiIIIII',0,0,0,3,10,-10,20,-20,101,102,203,7,8)
        client=object.__new__(FPGAClient)
        roi=np.zeros((16,32),np.uint8)
        with patch.object(client,'request',return_value=(reply,2.5)) as request:
            result=client.infer('grayscale8',roi,roi,9,capture_ms=123)
        self.assertEqual((result.left_logits,result.right_logits),((10,-10),(20,-20)))
        self.assertEqual((result.left_blinks,result.right_blinks,result.blink_flags),(7,8,3))
        self.assertEqual(request.call_args.args[0],0x12)


if __name__=='__main__':unittest.main()
