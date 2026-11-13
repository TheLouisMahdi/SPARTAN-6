"""Offline compact transport/startup regressions; all device calls are mocked."""
import binascii
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import demo
from jtag_host import CNN_REPLY
from tiny_jtag import JtagTinyClient, tiny_packet, parse_tiny_reply


class CompactTests(unittest.TestCase):
    def test_bounded_logging_keeps_order_and_reports_backpressure(self):
        from bounded_log import BoundedLog
        class Stream:
            def __init__(self):self.started=threading.Event();self.release=threading.Event();self.records=[]
            def write(self,text):
                self.started.set()
                if not self.release.wait(2):raise TimeoutError("synthetic disk stall")
                self.records.append(text)
            def close(self):pass
        stream=Stream();log=BoundedLog(stream=stream,capacity=2)
        try:
            log.write("first")
            self.assertTrue(stream.started.wait(1))
            log.write("second");log.write("third")
            with self.assertRaises(BufferError):log.write("rejected")
        finally:
            stream.release.set();log.close()
        self.assertEqual(stream.records,["first","second","third"])
        self.assertFalse(log.worker.is_alive())
    def test_compound_exchange_keeps_host_identity_validation(self):
        class Session:
            timeout=.01
            def scan(self,*args): raise AssertionError("unexpected extra host turnaround")
            def exchange_eye(self,packet):
                seq=struct.unpack_from("<I",packet,43)[0]
                frame,stamp,side=struct.unpack_from("<IQB",packet,49)
                reply=CNN_REPLY.pack(b"JTY1",seq,146,binascii.crc_hqx(packet[41:-3],0xffff),0,2,1,-23,42,101000,frame,stamp,side)
                return dict(data=reply,wall_ms=3,upload_ms=.3,poll_count=3,exchange_ms=2.8,upload_provenance="DETERMINISTIC_UNIT_TEST")
        client=JtagTinyClient(Session(),0x353bd6ca)
        client.send_eye(bytes(128),77,1234567890123,1)
        result=client.receive_eye()
        self.assertEqual(result["logits"],(-23,42))
        self.assertEqual(result["capture_ns"],1234567890123)
        self.assertEqual(result["polls"][0]["poll_count"],3)
    def test_stream_startup_failure_cannot_open_camera_or_landmarker(self):
        from live_pipeline import live_cli
        slot=SimpleNamespace(ready=True,method="grayscale2",record={"architecture":"tiny-eye-v1"},checkpoint_sha256="353bd6ca"+"0"*56)
        with tempfile.TemporaryDirectory() as folder, mock.patch("live_pipeline.ModelSlot",return_value=slot), \
                mock.patch("cv2.VideoCapture") as camera,mock.patch("capture.capture_frames") as capture, \
                mock.patch("capture.vision.FaceLandmarker.create_from_options") as landmarker, \
                mock.patch("cv2.namedWindow"),mock.patch("cv2.destroyAllWindows"),mock.patch("cv2.waitKey",return_value=-1),mock.patch("cv2.getWindowProperty",return_value=1):
            def fail(): raise RuntimeError("synthetic startup failure")
            code=live_cli(["--network","tiny","--backend","fpga","--pipeline","overlap","--report",str(Path(folder)/"session.json")],client_factory=fail)
            self.assertEqual(code,1)
            camera.assert_not_called();capture.assert_not_called();landmarker.assert_not_called()

    def test_report_separates_kernel_cycles_and_excludes_programming_time(self):
        from pipeline_report import summarize
        ns=1_000_000_000
        events=[dict(kind="capture",frame_id=1,capture_ns=ns),
            dict(kind="prepared",frame_id=1,capture_ns=ns,prep_start_ns=ns,prep_end_ns=ns+100_000_000),
            dict(kind="inference",frame_id=1,cycles=[10,20,33],fpga_start_ns=ns+100_000_000,fpga_end_ns=ns+200_000_000,uart_roundtrip_ms=100,uart_upload_ms=None,reason=None),
            dict(kind="published",frame_id=1,capture_ns=ns,result_start_ns=ns+200_000_000,result_end_ns=ns+300_000_000,reason=None,
                 eyes={"left":{"state":"OPEN"},"right":{"state":"CLOSED"}},blink_count=0,blink_event=False)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);(path/"trace.jsonl").write_text("\n".join(json.dumps(e) for e in events)+"\n")
            summary=summarize(path,dict(status="PASS",source="DETERMINISTIC_UNIT_TEST",mode="overlap",start_ns=0,end_ns=99*ns))
            self.assertAlmostEqual(summary["measurement_seconds"],.3)
            self.assertAlmostEqual(summary["fpga_kernel_pair"]["mean_ms"],30/50000)
            self.assertAlmostEqual(summary["fpga_controller_pair"]["mean_ms"],33/50000)
    def test_preview_renders_synthetic_frame_without_devices(self):
        import numpy as np
        from live_pipeline import render_stream_preview
        frame=np.zeros((240,640,3),np.uint8)
        prepared=SimpleNamespace(frame=frame,frame_id=7,capture_ns=1000000000,
            boxes={"left":(60,180,100,200)},prep_start_ns=1000000000,prep_end_ns=1001000000)
        snapshot=SimpleNamespace(inference=SimpleNamespace(prepared=prepared,hardware=None,start_ns=1001000000,end_ns=1002000000),
            eyes={"left":{"state":"OPEN"},"right":{"state":"UNAVAILABLE"}},timing=SimpleNamespace(count=0),reason="SYNTHETIC_OFFLINE_RENDER")
        counts={"capture":1,"fpga_completed":0,"dropped_preparation":0}
        with mock.patch("cv2.VideoCapture",side_effect=AssertionError("camera forbidden")),mock.patch("cv2.imshow",side_effect=AssertionError("window forbidden")):
            rendered=render_stream_preview(snapshot,1010000000,counts,1000000000,"overlap")
        self.assertEqual(rendered.shape,frame.shape)
        self.assertTrue(np.any(rendered))
        self.assertFalse(np.any(frame))
    def test_compact_startup_checks_coefficient_readback(self):
        import numpy as np
        from fpga_host import HardwareResult
        uart, session, backend=mock.Mock(),mock.Mock(),mock.Mock()
        uart.read_model.return_value=b"coefficients"
        backend.infer.return_value=HardwareResult(0,1,0,(-23,42),(-23,42),101000,101000,202000,1)
        model=SimpleNamespace(method="grayscale2",params={})
        client=demo.DualClient(uart,session,backend)
        with mock.patch("fpga_host.model_image",return_value=b"coefficients"),mock.patch("integer_model.infer",return_value=np.array([[-23,42]])):
            client.load_model(model)
            uart.read_model.return_value=b"changed"
            with self.assertRaisesRegex(RuntimeError,"readback mismatch"):
                client.load_model(model)
    def test_demo_startup_checks_hardware_arithmetic_and_closes_both_owners(self):
        import numpy as np
        from fpga_host import HardwareResult
        uart, session, backend = mock.Mock(), mock.Mock(), mock.Mock()
        model=SimpleNamespace(method="omz_bgr32",predict=lambda roi,fixed:np.array([-23,42]))
        backend.infer_pretrained.return_value=HardwareResult(0,4,0,(-23,42),(-23,42),367739,367739,735478,1)
        client=demo.DualClient(uart,session,backend)
        client.load_model(model)
        backend.infer_pretrained.return_value.left_logits=(42,-23)
        with self.assertRaisesRegex(RuntimeError,"before camera"):
            client.load_model(model)
        uart.close.side_effect=OSError("synthetic disconnect")
        with self.assertRaises(OSError): client.close()
        session.close.assert_called_once()

    def test_dsp_requantizer_math_and_cycle_equivalence(self):
        import os
        import random
        import subprocess
        rng = random.Random(611)
        values = [(a,m) for a in (-2147483648,-1,0,1,127,255,256,2147483647)
                  for m in (0,1,32768,8388608,16777216,0xffffffff)]
        values += [(rng.randrange(-2147483648,2147483648),rng.randrange(2**32)) for _ in range(512)]
        binary = ROOT / "assets/tools/iverilog/ucrt64/bin"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            lines = []
            for a,m in values:
                expected = 0 if a<=0 else min(255,(a*m+0x800000)>>24)
                lines.append(f"{a&0xffffffff:08x}{m:08x}{expected:02x}")
            (path/"rq_vectors.hex").write_text("\n".join(lines)+"\n")
            env = dict(os.environ,PATH=str(binary)+os.pathsep+os.environ["PATH"])
            commands = [[str(binary/"iverilog.exe"),"-g2001",f"-DRQ_CASES={len(values)}","-s","tb_requantize_equivalence","-o","rq.vvp",
                         str(ROOT/"fpga/rtl/requantize_q24.v"),str(ROOT/"fpga/experimental/requantize_q24_dsp.v"),str(ROOT/"research_ext/tb_requantize_equivalence.v")],
                        [str(binary/"vvp.exe"),"rq.vvp"]]
            for command in commands:
                output = subprocess.check_output(command,cwd=path,env=env,stderr=subprocess.STDOUT,text=True,timeout=30)
            self.assertIn("REQUANTIZE_EQUIVALENCE_PASS",output)
            self.assertNotIn("FAIL",output)

    def test_bounded_scheduler_routes_compact_codes_on_one_owner(self):
        import numpy as np
        from fpga_host import HardwareResult
        from stream_pipeline import StreamingPipeline
        now = [1000000000000]
        def clock():
            now[0] += 1000000
            return now[0]
        model = SimpleNamespace(method="grayscale2", checkpoint_sha256="353bd6ca"+"0"*56, params={})
        events, calls = [], []
        class Client:
            def __init__(self): self.owner = threading.get_ident()
            def load_model(self, model): return dict(method_id=1, model_loaded=True, fingerprint=0x353bd6ca)
            def read_model(self): return b"coefficients"
            def configure(self, *args): pass
            def close(self): pass
            def infer(self, method, left, right, frame, stamp):
                self_assert = (threading.get_ident()==self.owner and method=="grayscale2" and left.shape==(16,32)
                               and set(np.unique(left))=={0,1,2,3} and np.array_equal(left,right))
                if not self_assert: raise AssertionError("compact owner/encoding contract")
                calls.append((frame,stamp))
                return HardwareResult(frame,1,0,(-23,42),(-23,42),120173,120173,240346,.1)
        roi = np.tile(np.array([0,64,128,255],np.uint8), (16,8))
        def source(stop):
            for frame in range(1,4): yield frame,clock(),roi
        def prepare(frame, ns): return dict(left=frame,right=frame),{}
        with mock.patch("fpga_host.model_image",return_value=b"coefficients"):
            pipeline=StreamingPipeline(source,lambda:prepare,Client,model,mode="sequential",emit=events.append,clock=clock)
            pipeline.start()
            self.assertTrue(pipeline.done.wait(3))
            pipeline.close();pipeline.raise_error()
        self.assertEqual([frame for frame,stamp in calls],[1,2,3])
        self.assertEqual([e["frame_id"] for e in events if e["kind"]=="published"],[1,2,3])

    def test_compact_contract_and_native_order(self):
        packet = tiny_packet(0xffffffff, 77, 1234567890123, 1, bytes([0x1b])*128, 0x353bd6ca)
        self.assertEqual(len(packet), 198)
        self.assertEqual(packet[41:49], struct.pack("<BBIH", 2, 2, 0xffffffff, 146))
        self.assertEqual(packet[66], 1)
        self.assertEqual(packet[67:195], bytes([0x1b])*128)
        self.assertEqual(struct.unpack("<H", packet[-3:-1])[0], binascii.crc_hqx(packet[41:-3], 0xffff))
        reply = CNN_REPLY.pack(b"JTY1", 2, 146, 9, 0, 2, 1, -100, 20, 123, 77, 1234, 1)
        self.assertEqual(parse_tiny_reply(reply)["native_logits"], (-100, 20))
        with self.assertRaises(ValueError):
            parse_tiny_reply(b"JTN1"+reply[4:])
        with self.assertRaises(ValueError):
            tiny_packet(0, 0, 0, 0, bytes(129), 1)

    def test_client_wrap_metadata_and_credit(self):
        class Session:
            timeout = .01
            def scan(self, data, capture=True):
                if len(data) == 198:
                    sequence = struct.unpack_from("<I", data, 43)[0]
                    frame, stamp, side = struct.unpack_from("<IQB", data, 49)
                    self.reply = CNN_REPLY.pack(b"JTY1", sequence, 146, binascii.crc_hqx(data[41:-3], 0xffff),
                        0, 2, 1, -23, 42, 120173, frame, stamp, side)
                return dict(data=self.reply, wall_ms=1)
        client = JtagTinyClient(Session(), 0x353bd6ca)
        client.sequence = 0xffffffff
        client.send_eye(bytes(128), 88, 1234, 0)
        with self.assertRaises(RuntimeError):
            client.send_eye(bytes(128), 89, 1234, 1)
        result = client.receive_eye()
        self.assertEqual(result["sequence"], 0)
        self.assertEqual(result["logits"], (-23, 42))
        self.assertEqual(result["frame_id"], 88)

    def test_check_missing_catalog_never_opens_devices(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("demo.open_client", side_effect=AssertionError("device opened")):
            self.assertEqual(demo.main(["--check", "--deployment", str(Path(folder)/"missing.json")]), 2)

    def test_deployment_hashes_and_pending_physical_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            file = base / "artifact"
            file.write_bytes(b"frozen")
            ref = dict(path="artifact", sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            entry = dict(model="tiny", transport="jtag", usercode=0x4a545931, physical_verified=False,
                         **{key: ref for key in ("bit", "svf", "rollback_svf", "model_file", "physical_evidence")})
            catalog = base / "catalog.json"
            catalog.write_text(json.dumps(dict(schema_version=1, images=[entry])))
            report = demo.preflight(catalog, ports=["TEST_PORT"])
            self.assertEqual(report["status"], "NOT_READY")
            self.assertEqual(report["candidates"][0]["status"], "NOT_VERIFIED")
            file.write_bytes(b"changed")
            self.assertEqual(demo.preflight(catalog, ports=[])["candidates"][0]["status"], "NOT_READY")


if __name__ == "__main__":
    unittest.main()
