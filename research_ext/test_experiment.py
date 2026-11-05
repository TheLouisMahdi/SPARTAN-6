"""Synthetic software contracts; these tests do not verify a physical FPGA."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import Event, Thread, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from encoders import METHODS
from evidence import classify_clip, encoders
from experiment import ComparisonPipeline, Session, regenerate, ROOT
from fpga_host import FPGAClient, HardwareResult, METHOD_IDS, PacketReader, encode_packet
from integer_model import infer
from live_pipeline import ModelSlot, PairResult, live_cli
from blink_timing import BinocularTimer
from ocular_metrics import observations
from rtl_evidence import synthetic_parameters, simulate


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.params = synthetic_parameters()
        np.savez(self.root / "weights.npz", **self.params)
        self.manifest = self.root / "manifest.json"
        self.record = dict(schema_version=1, state="trained", architecture="tiny-eye-v1", input=[1, 16, 32],
            classes=["OPEN", "CLOSED"], encoder="aeee2", encoder_spec=METHODS["aeee2"].SPEC,
            checkpoint="weights.npz", sha256=hashlib.sha256((self.root/"weights.npz").read_bytes()).hexdigest())
        self.manifest.write_text(json.dumps(self.record))
        self.model = ModelSlot(self.manifest)
        self.rois = dict(left=np.arange(512, dtype=np.uint8).reshape(16, 32), right=np.full((16, 32), 180, np.uint8))

    def pipeline(self, mode="compare", client=None):
        pipeline = ComparisonPipeline(self.model, mode, client)
        self.addCleanup(pipeline.close)
        return pipeline

    def client(self, modify=0):
        def response(method, left, right, frame_id):
            logits = infer(np.stack([left, right])[:, None] * np.uint8(85), self.params)
            logits = logits.astype(np.int64)
            logits[0, 0] += modify
            return HardwareResult(frame_id, METHOD_IDS[method], 0, tuple(logits[0]), tuple(logits[1]), 100, 100, 210, 8)
        return SimpleNamespace(infer=response)

    def session(self, mode="compare"):
        session = Session(self.root / "session", self.model, mode, 0, "TEST_ONLY", 115200)
        self.addCleanup(lambda: [s.close() for s in session.streams.values()])
        return session

    def rows(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_active_encoders_and_shared_exact_integer_input(self):
        self.assertEqual(set(encoders()), {"aeee2", "grayscale2"})
        for name in ("aeee2", "grayscale2"):
            self.model.method = name
            pipeline = self.pipeline(client=self.client())
            with patch.object(METHODS[name], "encode", wraps=METHODS[name].encode) as encode:
                result = pipeline.predict(1, 1_000_000, self.rois)
            self.assertEqual(encode.call_count, 2)
            np.testing.assert_array_equal(result.backends["cpu"]["logits"], result.backends["fpga"]["logits"])

    def test_cpu_worker_overlaps_fpga_on_caller_thread(self):
        entered, release = Event(), Event()
        owner = get_ident()
        pipeline = self.pipeline(client=self.client())
        original_cpu, original_fpga = pipeline._cpu, pipeline.client.infer
        def cpu(codes):
            self.assertNotEqual(owner, get_ident())
            entered.set()
            self.assertTrue(release.wait(2))
            return original_cpu(codes)
        def fpga(*args):
            self.assertEqual(owner, get_ident())
            self.assertTrue(entered.wait(2))
            release.set()
            return original_fpga(*args)
        pipeline._cpu, pipeline.client.infer = cpu, fpga
        pipeline.predict(1, 1_000_000, self.rois)

    def test_connection_owner_rejects_cross_thread_requests(self):
        client = object.__new__(FPGAClient)
        client.owner = get_ident()
        errors = []
        def invoke():
            try:
                client.request(1)
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = Thread(target=invoke)
        thread.start()
        thread.join(2)
        self.assertIn("owner", errors[0])

    def test_uart_ignores_stale_sequence_and_wrong_command(self):
        client = object.__new__(FPGAClient)
        client.owner, client.sequence, client.timeout = get_ident(), 0, 1
        client.reader = PacketReader()
        client.serial = Mock(in_waiting=0)
        client.serial.read.return_value = (encode_packet(0x81, 99, b"\0wrong") +
            encode_packet(0x90, 1, b"\0wrong") + encode_packet(0x81, 1, b"\0correct"))
        reply, elapsed = client.request(1)
        self.assertEqual(reply, b"\0correct")
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(client.serial.write.call_count, 1)
        self.assertEqual(client.unmatched_responses, 2)
        self.assertEqual(client.request_count, 1)

    def test_uart_timeout_clears_partial_packet(self):
        client = object.__new__(FPGAClient)
        client.owner, client.sequence, client.timeout = get_ident(), 0, 1
        client.reader = PacketReader()
        client.reader.pending.extend(b"partial")
        client.serial = Mock(in_waiting=0)
        with patch("fpga_host.time.perf_counter_ns", side_effect=[0, 2_000_000_000]):
            with self.assertRaises(TimeoutError):
                client.request(1)
        self.assertEqual(client.reader.pending, b"")
        self.assertEqual(client.timeout_count, 1)

    def test_simulator_timeout_is_a_failure_not_a_missing_dependency(self):
        output = self.root / "simulation_timeout"
        with patch("rtl_evidence.subprocess.run", side_effect=[SimpleNamespace(returncode=0),
                subprocess.TimeoutExpired("vvp", 600)]):
            result = simulate(output)
        self.assertEqual(result["status"], "FAIL_TIMEOUT")

    def test_duplicate_out_of_order_and_wrong_response(self):
        client = self.client()
        pipeline = self.pipeline(client=client)
        pipeline.predict(3, 3_000_000, self.rois)
        for frame_id, timestamp in ((3, 4_000_000), (2, 4_000_000), (4, 3_000_000)):
            with self.assertRaises(ValueError):
                pipeline.predict(frame_id, timestamp, self.rois)
        client.infer = lambda *args: HardwareResult(99, 3, 0, (1, 2), (1, 2), 1, 1, 2, 1)
        result = pipeline.predict(4, 4_000_000, self.rois)
        self.assertIsNone(result.backends["fpga"]["logits"])
        self.assertIn("Mismatched", result.backends["fpga"]["error"])

    def test_timeout_and_missing_eye_have_no_fabricated_logits(self):
        client = SimpleNamespace(infer=Mock(side_effect=TimeoutError("timeout")))
        pipeline = self.pipeline(client=client)
        result = pipeline.predict(1, 1_000_000, self.rois)
        self.assertIsNone(result.backends["fpga"]["logits"])
        self.assertIsNotNone(result.backends["cpu"]["logits"])
        result = pipeline.predict(2, 2_000_000, {"left": self.rois["left"]})
        self.assertEqual(client.infer.call_count, 1)
        self.assertTrue(all(d["logits"] is None for d in result.backends.values()))

    def test_comparison_csv_detects_integer_mismatch_and_regenerates(self):
        session = self.session()
        timestamp = session.origin + 1_000_000
        session.capture(1, timestamp)
        result = self.pipeline(client=self.client(modify=7)).predict(1, timestamp, self.rois)
        session.record(result, [0, 0], timestamp + 2_000_000)
        session.finish("complete")
        summary = json.loads((session.path / "summary.json").read_text())
        self.assertEqual(summary["comparison"]["exact_output_mismatches"], 1)
        self.assertEqual(summary["comparison"]["max_abs_logit_diff"], 7)
        before = [(session.path / name).read_bytes() for name in ("summary.json", "report.html")]
        regenerate(session.path)
        self.assertEqual(before, [(session.path / name).read_bytes() for name in ("summary.json", "report.html")])
        self.assertEqual(summary["fatigue_level"], "UNKNOWN")

    def test_stale_rejected_data_excluded_but_raw_logits_retained(self):
        session = self.session()
        pipeline = self.pipeline(client=self.client())
        for frame_id, ms in enumerate((1, 51, 101), 1):
            timestamp = session.origin + ms*1_000_000
            session.capture(frame_id, timestamp)
            result = pipeline.predict(frame_id, timestamp, self.rois)
            session.record(result, [None, None], timestamp+300_000_000)
        session.finish("complete")
        rows = self.rows(session.path / "fpga.csv")
        self.assertTrue(all(r["validity"] == "stale" and r["left_state"] == "UNAVAILABLE" for r in rows))
        self.assertTrue(all(r["left_open_logit"] != "" for r in rows))
        self.assertEqual(regenerate(session.path)["backends"]["fpga"]["ocular"]["eyes"]["left"]["observed_ms"], 0)
        self.assertEqual(self.rows(session.path/"comparison.csv")[0]["max_abs_logit_diff"], "")

    def test_dropped_frames_and_independent_backend_files(self):
        for mode in ("cpu", "fpga"):
            session = Session(self.root / mode, self.model, mode, 0, "TEST", 115200)
            self.addCleanup(lambda s=session: [v.close() for v in s.streams.values()])
            for frame_id in range(1, 5):
                session.capture(frame_id, session.origin+frame_id*10_000_000)
            result = self.pipeline(mode, None if mode == "cpu" else self.client()).predict(3, session.origin+30_000_000, self.rois)
            session.record(result, [0, 0], result.capture_ns+1_000_000)
            session.finish("complete")
            rows = self.rows(session.path/(mode+".csv"))
            self.assertEqual([int(r["frame_id"]) for r in rows], [1, 2, 3, 4])
            self.assertEqual([r["validity"] for r in rows], ["dropped", "dropped", "valid", "dropped"])
            self.assertEqual(rows[0]["inference_ms"], "")
            self.assertFalse((session.path / (("fpga" if mode == "cpu" else "cpu")+".csv")).exists())
            summary = regenerate(session.path)["backends"][mode]
            self.assertEqual((summary["captured_frames"], summary["processed_frames"], summary["dropped_frames"]), (4, 1, 3))

    def test_missing_model_and_encoder_mismatch_stop_before_devices(self):
        for args in (["--encoder", "grayscale2", "--manifest", str(self.manifest)],
                     ["--encoder", "aeee2", "--manifest", str(self.root/"missing.json")]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                live_cli(args)
            self.assertEqual(error.exception.code, 1)
        self.record.update(state="untrained", checkpoint=None, sha256=None)
        self.manifest.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "MODEL_NOT_READY"):
            ComparisonPipeline(ModelSlot(self.manifest), "cpu")

    def test_checkpoint_hash_mismatch_rejected(self):
        self.record["sha256"] = "0"*64
        self.manifest.write_text(json.dumps(self.record))
        with self.assertRaisesRegex(ValueError, "hash"):
            ModelSlot(self.manifest)

    def test_tied_logits_remain_comparable_without_ocular_state(self):
        session = self.session()
        timestamp = session.origin+1_000_000
        data = {name: dict(logits=np.ones((2, 2), np.int32), inference_ms=1) for name in ("cpu", "fpga")}
        result = PairResult(1, timestamp, {}, 1, backends=data)
        session.record(result, [None, None], timestamp+1_000_000)
        session.finish("complete")
        self.assertEqual(self.rows(session.path/"cpu.csv")[0]["left_state"], "UNAVAILABLE")
        self.assertEqual(regenerate(session.path)["comparison"]["matched_frames"], 1)

    def test_clip_classifier_requires_whole_clip_provenance(self):
        artifact = self.root / "fatigue.json"
        fields = [s+suffix for s in ("left", "right") for suffix in ("_closed_fraction", "_blink_rate")]
        model = dict(status="PASS", classes=[0, 5, 10], aggregation="whole-clip valid-time indicators",
            features=fields, train_mean=[0]*4, train_std=[1]*4, centroids=[[0]*4, [1]*4, [2]*4],
            splits={s: dict(status="PASS", n_clips=3, n_subjects=3) for s in ("val", "test")})
        artifact.write_text(json.dumps(model))
        row = dict(dataset="UTA-RLDD", label_scope="video", subject_id="TEST", source_sha256="TEST",
            annotation_sha256="TEST", replay_summary_sha256="TEST", **dict.fromkeys(fields, 1))
        self.assertEqual(classify_clip(row, artifact)["level"], "LOW_VIGILANCE")
        row["label_scope"] = "window"
        self.assertEqual(classify_clip(row, artifact)["level"], "UNKNOWN")

    def test_missing_or_unvalidated_fatigue_artifact_is_unknown(self):
        self.assertEqual(classify_clip({})["level"], "UNKNOWN")
        artifact = self.root / "fatigue.json"
        artifact.write_text(json.dumps(dict(status="PASS")))
        self.assertEqual(classify_clip({}, artifact)["level"], "UNKNOWN")

    def test_session_csv_reuses_evidence_ocular_reader(self):
        session = self.session("cpu")
        timestamp = session.origin+1_000_000
        result = self.pipeline("cpu").predict(1, timestamp, self.rois)
        session.record(result, [None, None], timestamp+300_000_000)
        session.finish("complete")
        samples = observations(session.path/"cpu.csv")
        self.assertEqual(samples[0]["left_state"], "UNAVAILABLE")

    def test_live_session_uses_existing_capture_and_closes_resources(self):
        import time
        def capture(pending, errors, stop, stopped, camera, on_capture):
            timestamp = time.monotonic_ns()
            on_capture(1, timestamp)
            pending.put((1, timestamp, np.zeros((240, 640, 3), np.uint8)))
            stop.wait(3)
            stopped.set()
        task = Mock()
        task.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        with patch("capture.capture_frames", capture), \
             patch("capture.vision.FaceLandmarker.create_from_options", return_value=task), \
             patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), redirect_stdout(io.StringIO()):
            code = live_cli(["--encoder", "aeee2", "--manifest", str(self.manifest),
                "--output", str(self.root/"live"), "--report", str(self.root/"live.json")])
        self.assertEqual(code, 0)
        row = self.rows(self.root/"live/cpu.csv")[0]
        self.assertEqual((row["validity"], row["left_open_logit"]), ("invalid", ""))
        task.close.assert_called_once()

    def test_delayed_timing_reply_is_stale_in_legacy_csv(self):
        import time
        def capture(pending, errors, stop, stopped, camera):
            pending.put((1, time.monotonic_ns(), np.zeros((240, 640, 3), np.uint8)))
            stop.wait(3)
            stopped.set()
        def delayed(timer, result, client):
            timing = BinocularTimer().update(0, 0, 1)
            return timing, [None, None], result.capture_ns+300_000_000
        task = Mock()
        task.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        with patch("capture.capture_frames", capture), \
             patch("capture.vision.FaceLandmarker.create_from_options", return_value=task), \
             patch("live_pipeline.synchronize_timing", delayed), \
             patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), redirect_stdout(io.StringIO()):
            code = live_cli(["--manifest", str(self.manifest), "--frames", str(self.root/"legacy.csv"),
                "--report", str(self.root/"legacy.json")])
        row = self.rows(self.root/"legacy.csv")[0]
        self.assertEqual(code, 0)
        self.assertEqual((row["validity"], row["left_state"], row["right_state"]), ("stale", "UNAVAILABLE", "UNAVAILABLE"))
        self.assertEqual(json.loads((self.root/"legacy.json").read_text())["valid_eye_pairs"], 0)

    @unittest.skipUnless(os.name == "nt", "Windows batch contract")
    def test_launchers_propagate_dispatch_success_and_failure(self):
        for name, args in (("run_cpu.bat", ["aeee2"]), ("run_fpga.bat", ["grayscale2", "COM26"]),
                           ("run_cpu.bat", ["pretrained"]), ("run_fpga.bat", ["pretrained", "COM26"]), ("test_all.bat", ["3"])):
            shutil.copy2(ROOT/name, self.root/name)
            for code in (0, 37):
                (self.root/"run.cmd").write_text(f"@echo off\nexit /b {code}\n")
                process = subprocess.run([os.environ["COMSPEC"], "/d", "/c", str(self.root/name), *args], capture_output=True)
                self.assertEqual(process.returncode, code)
        for name in ("run_cpu.bat", "run_fpga.bat"):
            process = subprocess.run([os.environ["COMSPEC"], "/d", "/c", str(self.root/name), "wrong"], capture_output=True)
            self.assertEqual(process.returncode, 2)


class OfflineExperimentTests(unittest.TestCase):
    """Synthetic files and mocked transport exercise real pretrained weights, not dataset accuracy."""
    @classmethod
    def setUpClass(cls):
        from pretrained_eye import PretrainedEye
        cls.model = PretrainedEye(precision="float")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def images(self, count=3, grayscale=False):
        import cv2
        from public_data import CEW_URL
        raw = self.root/"raw"
        (raw/"verified_open").mkdir(parents=True)
        (raw/"verified_closed").mkdir()
        for index in range(count):
            folder = "verified_open" if index % 2 == 0 else "verified_closed"
            shape = (24,24) if grayscale else (24,24,3)
            cv2.imwrite(str(raw/folder/f"fixture_{index}.png"), np.full(shape, index*60+10, np.uint8))
        provenance = dict(approved_local_use=True, source_url=CEW_URL, rights_note="SYNTHETIC_TEST_ONLY; no real dataset results",
            label_provenance="Synthetic fixture labels for software contract only", label_directories={
                "verified_open":dict(label="OPEN", source_type="eye_patch"),
                "verified_closed":dict(label="CLOSED", source_type="eye_patch")})
        (raw/"provenance.json").write_text(json.dumps(provenance))
        return raw

    def manifest(self, count=3, grayscale=False):
        from public_data import register_cew
        raw = self.images(count, grayscale)
        manifest = self.root/"registered/manifest.csv"
        register_cew(raw, manifest)
        return manifest

    def run_cew(self, manifest, mode="cpu", output="run", client=None):
        from experiment import experiment_cli
        args = ["cew", "--manifest", str(manifest), "--mode", mode, "--output", str(self.root/output)]
        with redirect_stdout(io.StringIO()), patch("fpga_host.FPGAClient", return_value=client) if client else patch("fpga_host.FPGAClient", side_effect=AssertionError("No board requested")):
            code = experiment_cli(args)
        return code, self.root/output

    def rows(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def transport(self, error=None):
        model = self.model
        status = dict(method_id=4, model_loaded=True, fingerprint=int(model.checkpoint_sha256[:8],16), binocular_timing=True)
        def infer(left, right, frame_id):
            if error:
                raise error
            logits = [model.predict(roi,fixed=True) for roi in (left,right)]
            return HardwareResult(frame_id,4,0,tuple(logits[0]),tuple(logits[1]),100,110,210,2500,upload_ms=7)
        client = Mock()
        client.load_model.return_value = status
        client.infer_pretrained.side_effect = infer
        client.reader.bad_packets = 0
        return client

    def video(self):
        import cv2
        from data_tools import sha256
        videos = self.root/"videos"
        videos.mkdir()
        path = videos/"fixture.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),10,(64,48))
        self.assertTrue(writer.isOpened())
        for index in range(4):
            writer.write(np.full((48,64,3),30+index*30,np.uint8))
        writer.release()
        with (videos/"videos.csv").open("w",newline="") as stream:
            fields = ("subject_id","session_id","video_id","path","sha256","label_scope","source_label")
            out = csv.DictWriter(stream,fields)
            out.writeheader()
            out.writerow(dict(subject_id="SYNTHETIC",session_id="fixture",video_id="fixture",path=path.name,
                sha256=sha256(path),label_scope="video",source_label="10"))
        registry = self.root/"registry.json"
        registry.write_text(json.dumps(dict(schema_version=1,datasets=[dict(dataset="UTA-RLDD",version="SYNTHETIC_TEST_ONLY",
            source_url="https://sites.google.com/view/utarldd/home",license="Synthetic fixture only",rights_note="No participant data",
            approved_local_use=True,root=str(videos))])))
        return registry, path

    def test_registration_preserves_paths_and_leaves_unknown_identity_empty(self):
        from public_data import read_cew_manifest
        manifest = self.manifest()
        rows = read_cew_manifest(manifest)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r["subject_id"] == "" for r in rows))
        self.assertTrue(all(Path(r["source_path"]).is_file() for r in rows))
        with self.assertRaises(FileExistsError):
            from public_data import register_cew
            register_cew(self.root/"raw",manifest)

    def test_registration_rejects_unverified_labels_and_exact_duplicates(self):
        from public_data import register_cew
        raw = self.images()
        provenance = json.loads((raw/"provenance.json").read_text())
        del provenance["label_directories"]
        (raw/"provenance.json").write_text(json.dumps(provenance))
        with self.assertRaisesRegex(ValueError,"labels are never inferred"):
            register_cew(raw,self.root/"out/manifest.csv")
        provenance["label_directories"] = {"verified_open":dict(label="OPEN",source_type="eye_patch"),"verified_closed":dict(label="CLOSED",source_type="eye_patch")}
        (raw/"provenance.json").write_text(json.dumps(provenance))
        source = next((raw/"verified_open").glob("*.png"))
        shutil.copy2(source,raw/"verified_closed/duplicate.png")
        with self.assertRaisesRegex(ValueError,"duplicate"):
            register_cew(raw,self.root/"out/manifest.csv")

    def test_cew_cpu_odd_patch_count_logs_three_real_samples_without_temporal_metrics(self):
        manifest = self.manifest(grayscale=True)
        code,output = self.run_cew(manifest)
        self.assertEqual(code,0)
        rows = self.rows(output/"samples.csv")
        self.assertEqual(len(rows),3)
        self.assertTrue(all(r["source_color_mode"] == "grayscale_replicated" for r in rows))
        self.assertTrue(all(r["cpu_prediction"] and r["cpu_fp32_prediction"] for r in rows))
        summary = json.loads((output/"summary.json").read_text())
        self.assertEqual(summary["accuracy"]["overall"]["cpu_fixed"]["count"],3)
        self.assertEqual(summary["samples"]["registered_source_samples"],3)
        self.assertNotIn("ocular",summary["backends"]["cpu"])
        for name in ("fpga.csv","comparison.csv","ocular.json"):
            self.assertFalse((output/name).exists())

    def test_fixed_only_cpu_benchmark_skips_the_float_pass(self):
        from experiment import experiment_cli
        manifest = self.manifest(2)
        output = self.root/"fixed_only"
        with redirect_stdout(io.StringIO()):
            code = experiment_cli(["cew","--manifest",str(manifest),"--cpu-precision","fixed","--output",str(output)])
        self.assertEqual(code,0)
        self.assertEqual(self.rows(output/"cpu.csv")[0]["cpu_fp32_ms"],"")
        self.assertEqual(self.rows(output/"samples.csv")[0]["cpu_fp32_prediction"],"")
        self.assertEqual(json.loads((output/"session.json").read_text())["cpu_precision"],"fixed-point")

    def test_corrupted_registered_image_and_changed_source_are_not_silently_discarded(self):
        manifest = self.manifest(2)
        rows = self.rows(manifest)
        Path(rows[0]["source_path"]).write_bytes(b"corrupt")
        code,output = self.run_cew(manifest)
        self.assertEqual(code,0)
        logged = self.rows(output/"samples.csv")
        self.assertEqual(len(logged),2)
        self.assertTrue(any("SHA256" in r["error_reason"] for r in logged))
        self.assertEqual(json.loads((output/"summary.json").read_text())["samples"]["decoded_source_samples"],1)

    def test_corrupted_and_unsupported_images_cannot_produce_a_pass_with_zero_inferences(self):
        import cv2
        from public_data import register_cew
        raw = self.images(2)
        next((raw/"verified_open").glob("*.png")).write_bytes(b"not an image")
        cv2.imwrite(str(next((raw/"verified_closed").glob("*.png"))),np.zeros((24,24,4),np.uint8))
        manifest = self.root/"registered/manifest.csv"
        register_cew(raw,manifest)
        code,output = self.run_cew(manifest)
        self.assertEqual(code,1)
        self.assertEqual(len(self.rows(output/"samples.csv")),2)
        self.assertEqual(json.loads((output/"session.json").read_text())["status"],"INCOMPLETE")
        self.assertEqual(self.rows(output/"cpu.csv"),[])

    def test_output_failure_retains_logs_and_marks_session_incomplete(self):
        from experiment import write_json as original
        manifest = self.manifest(2)
        def write(path,value):
            if path.name == "summary.json":
                raise OSError("Synthetic output failure")
            return original(path,value)
        with patch("experiment.write_json",side_effect=write):
            code,output = self.run_cew(manifest)
        self.assertEqual(code,1)
        self.assertEqual(json.loads((output/"session.json").read_text())["status"],"INCOMPLETE")
        self.assertEqual(len(self.rows(output/"cpu.csv")),1)

    def test_confusion_matrix_orientation_and_zero_support(self):
        from experiment import cew_accuracy
        rows = [dict(label=a,cpu_prediction=b,cpu_fp32_prediction=b,source_type="eye_patch")
            for a,b in (("OPEN","OPEN"),("OPEN","CLOSED"),("CLOSED","CLOSED"))]
        metric = cew_accuracy(rows)["overall"]["cpu_fixed"]
        self.assertEqual(metric["confusion_matrix"],[[1,1],[0,1]])
        self.assertAlmostEqual(metric["accuracy"],2/3)
        self.assertAlmostEqual(metric["macro_f1"],2/3)
        self.assertEqual(cew_accuracy([])["overall"],{})

    def test_pretrained_comparison_uses_identical_inputs_and_actual_fp32_scores(self):
        from pretrained_eye import input_pixels
        roi = np.arange(24*24*3,dtype=np.uint8).reshape(24,24,3)
        client = self.transport()
        pipeline = ComparisonPipeline(self.model,"compare",client)
        self.addCleanup(pipeline.close)
        result = pipeline.predict(1,0,dict(left=roi,right=roi.copy()))
        a,b = result.backends["cpu"],result.backends["fpga"]
        np.testing.assert_array_equal(a["logits"],b["logits"])
        self.assertEqual(a["input_left_sha256"],hashlib.sha256(input_pixels(roi).tobytes()).hexdigest())
        np.testing.assert_array_equal(a["fp32_logits"][0],self.model.predict(roi))

    def test_cew_mocked_compare_has_independent_logs_without_live_staleness(self):
        manifest = self.manifest(2)
        code,output = self.run_cew(manifest,"compare",client=self.transport())
        self.assertEqual(code,0)
        summary = json.loads((output/"summary.json").read_text())
        self.assertEqual(summary["comparison"]["exact_output_mismatches"],0)
        self.assertEqual(summary["comparison"]["matched_frames"],1)
        self.assertEqual(self.rows(output/"fpga.csv")[0]["uart_roundtrip_ms"],"2500")
        for name in ("cpu.csv","fpga.csv","comparison.csv"):
            self.assertEqual(len(self.rows(output/name)),1)
        self.assertIsNone(summary["session"]["fpga_configuration"]["pe_count"])

    def test_fpga_timeout_retains_cpu_diagnostics_and_incomplete_status(self):
        manifest = self.manifest(2)
        code,output = self.run_cew(manifest,"compare",client=self.transport(TimeoutError("serial timeout")))
        self.assertEqual(code,1)
        summary = json.loads((output/"summary.json").read_text())
        self.assertEqual(summary["session"]["status"],"INCOMPLETE")
        self.assertEqual(summary["backends"]["fpga"]["timeout_count"],1)
        self.assertEqual(self.rows(output/"comparison.csv")[0]["max_abs_logit_diff"],"")
        self.assertTrue(self.rows(output/"samples.csv")[0]["cpu_prediction"])

    def test_recovered_bad_packet_or_unmatched_response_stops_the_experiment(self):
        manifest = self.manifest(4)
        for index,kind in enumerate(("crc","sequence")):
            client = self.transport()
            client.reader.bad_packets = int(kind == "crc")
            client.unmatched_responses = int(kind == "sequence")
            code,output = self.run_cew(manifest,"compare",output=f"integrity_{index}",client=client)
            self.assertEqual(code,1)
            self.assertEqual(json.loads((output/"session.json").read_text())["status"],"INCOMPLETE")
            self.assertEqual(client.infer_pretrained.call_count,1)

    def test_model_mismatch_never_falls_back_to_cpu(self):
        manifest = self.manifest(2)
        client = self.transport()
        client.load_model.side_effect = RuntimeError("Program the matching pretrained CNN bitstream before inference")
        code,output = self.run_cew(manifest,"fpga",client=client)
        self.assertEqual(code,1)
        self.assertEqual(json.loads((output/"session.json").read_text())["status"],"FAIL")
        self.assertFalse((output/"cpu.csv").exists())
        client.infer_pretrained.assert_not_called()

    def test_missing_dataset_stops_before_serial_and_writes_status(self):
        from experiment import experiment_cli
        with patch("fpga_host.FPGAClient",side_effect=AssertionError("Must not open UART")),redirect_stdout(io.StringIO()):
            self.assertEqual(experiment_cli(["cew","--manifest",str(self.root/"missing.csv"),"--mode","fpga","--output",str(self.root/"missing")]),1)
        self.assertEqual(json.loads((self.root/"missing/session.json").read_text())["status"],"DATA_NOT_AVAILABLE")
        self.assertTrue((self.root/"missing/report.html").is_file())

    def test_real_video_pts_replay_uses_cnn_and_never_invents_eye_truth(self):
        from experiment import experiment_cli
        registry,path = self.video()
        localizer = Mock()
        def eyes(frame,localizer,timestamp):
            return dict(left=frame[:24,:24].copy(),right=frame[:24,24:48].copy())
        output = self.root/"video_run"
        with patch("experiment.create_localizer",return_value=localizer),patch("experiment.locate_eyes",side_effect=eyes),redirect_stdout(io.StringIO()):
            code = experiment_cli(["video","--registry",str(registry),"--video-id","fixture","--output",str(output)])
        self.assertEqual(code,0)
        rows = self.rows(output/"samples.csv")
        self.assertEqual(len(rows),8)
        self.assertEqual([float(rows[i]["media_timestamp_ms"]) for i in range(0,8,2)],[0,100,200,300])
        self.assertTrue(all(r["label"] == "" for r in rows))
        self.assertTrue(all(r["cpu_prediction"] for r in rows))
        ocular = json.loads((output/"ocular.json").read_text())["cpu"]
        self.assertEqual(ocular["windows_seconds"]["60"][0]["indicator"],"UNKNOWN")
        self.assertEqual(json.loads((output/"summary.json").read_text())["processed_video_count"],1)
        self.assertEqual(json.loads((output/"session.json").read_text())["clip_self_report_level"],"DROWSY")
        localizer.close.assert_called_once()

    def test_video_decode_interruption_is_incomplete_and_missing_eyes_break_coverage(self):
        from experiment import experiment_cli
        registry,path = self.video()
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True,np.zeros((48,64,3),np.uint8)),(False,None)]
        output = self.root/"partial"
        with patch("experiment.create_localizer",return_value=Mock()),patch("experiment.locate_eyes",return_value={}),\
             patch("cv2.VideoCapture",return_value=capture),redirect_stdout(io.StringIO()):
            code = experiment_cli(["video","--registry",str(registry),"--video-id","fixture","--output",str(output)])
        self.assertEqual(code,1)
        summary = json.loads((output/"summary.json").read_text())
        self.assertEqual(summary["session"]["status"],"INCOMPLETE")
        self.assertEqual(summary["processed_video_count"],0)
        self.assertEqual(self.rows(output/"cpu.csv")[0]["validity"],"invalid")
        capture.release.assert_called_once()

    def test_video_keyboard_interruption_releases_resources(self):
        from experiment import experiment_cli
        registry,path = self.video()
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = KeyboardInterrupt
        localizer = Mock()
        output = self.root/"interrupted"
        with patch("experiment.create_localizer",return_value=localizer),patch("cv2.VideoCapture",return_value=capture),redirect_stdout(io.StringIO()):
            code = experiment_cli(["video","--registry",str(registry),"--video-id","fixture","--output",str(output)])
        self.assertEqual(code,130)
        self.assertEqual(json.loads((output/"session.json").read_text())["status"],"INCOMPLETE")
        capture.release.assert_called_once()
        localizer.close.assert_called_once()

    def test_offline_report_regeneration_is_byte_identical(self):
        code,output = self.run_cew(self.manifest(2))
        self.assertEqual(code,0)
        before = [(output/name).read_bytes() for name in ("summary.json","report.html")]
        regenerate(output)
        self.assertEqual(before,[(output/name).read_bytes() for name in ("summary.json","report.html")])

    def test_indicator_thresholds_partial_windows_and_missing_data(self):
        from ocular_metrics import ocular_indicator,temporal_report
        self.assertEqual(ocular_indicator(1,60000,.01,100),"LOW")
        self.assertEqual(ocular_indicator(1,60000,.15,100),"MODERATE")
        self.assertEqual(ocular_indicator(1,60000,.35,100),"HIGH")
        self.assertEqual(ocular_indicator(1,60000,.01,2000),"HIGH")
        self.assertEqual(ocular_indicator(.79,60000,.5,3000),"UNKNOWN")
        self.assertEqual(ocular_indicator(1,29000,.5,3000),"UNKNOWN")
        self.assertEqual(ocular_indicator(1,60000,.5,3000,full_window=False),"UNKNOWN")
        rows = [dict(t_ms=i*100,left_state="OPEN",right_state="OPEN") for i in range(601)]
        for i in range(10,26):
            rows[i].update(left_state="CLOSED",right_state="CLOSED")
        rows[25].update(left_state="UNAVAILABLE",right_state="UNAVAILABLE")
        for i in range(26,42):
            rows[i].update(left_state="CLOSED",right_state="CLOSED")
        window = temporal_report(rows)["windows_seconds"]["60"][0]
        self.assertLess(window["bilateral"]["longest_observed_closure_ms"],2000)
        self.assertEqual(window["indicator"],"MODERATE")


if __name__ == "__main__":
    unittest.main()
