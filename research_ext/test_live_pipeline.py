"""Offline contract checks with synthetic tensors; no camera, training or board."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from threading import Barrier, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from blink_timing import BinocularTimer
from encoders import METHODS
from fpga_host import HardwareResult
from integer_model import infer, requantize
from live_pipeline import EyePipeline, ModelSlot, PairResult, live_cli, model_cli, read_parameters, synchronize_timing, update_timing


def scalar_logits(pixels, params):
    """Independent Python-integer oracle: no vectorized inference helpers."""
    def quantize(value, multiplier):
        return min(255, (max(value, 0) * multiplier + (1 << 23)) >> 24)

    values = [pixels.tolist()]
    for name in ("conv1", "conv2"):
        weights = params[name + "_weight"].tolist()
        biases = params[name + "_bias"].tolist()
        multiplier = int(params[name + "_multiplier"])
        height, width = (len(values[0]) - 2) // 2, (len(values[0][0]) - 2) // 2
        pooled = []
        for channel, filters in enumerate(weights):
            plane = []
            for y in range(height):
                row = []
                for x in range(width):
                    cells = []
                    for dy in (0, 1):
                        for dx in (0, 1):
                            accumulator = biases[channel]
                            for source, kernel in zip(values, filters):
                                for ky in range(3):
                                    for kx in range(3):
                                        accumulator += source[2*y+dy+ky][2*x+dx+kx] * kernel[ky][kx]
                            cells.append(quantize(accumulator, multiplier))
                    row.append(max(cells))
                plane.append(row)
            pooled.append(plane)
        values = pooled
    vector = [value for plane in values for row in plane for value in row]
    for name in ("fc1", "fc2"):
        accumulators = [int(bias) + sum(value * int(weight) for value, weight in zip(vector, weights))
                        for weights, bias in zip(params[name + "_weight"], params[name + "_bias"])]
        if name == "fc1":
            vector = [quantize(value, int(params[name + "_multiplier"])) for value in accumulators]
    return accumulators


class TimingLink:
    """Deterministic transport stand-in; no serial device or hardware claim."""
    def __init__(self):
        self.timer = BinocularTimer()
        self.samples = []

    def eye_timing(self, left, right, timestamp_ms, frame_id):
        self.samples.append((left, right, timestamp_ms, frame_id))
        return self.timer.update(left, right, timestamp_ms), 0

    def invalidate(self, frame_id):
        self.timer.invalidate()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "eye_model.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 1, "state": "untrained", "architecture": "tiny-eye-v1",
            "input": [1, 16, 32], "classes": ["OPEN", "CLOSED"], "encoder": "grayscale8",
            "checkpoint": None, "sha256": None}), encoding="utf-8")

    def activate(self):
        rng = np.random.default_rng(19)
        params = {}
        for name, shape in zip(("conv1", "conv2", "fc1", "fc2"),
                               ((4, 1, 3, 3), (8, 4, 3, 3), (8, 96), (2, 8))):
            params[name + "_weight"] = rng.integers(-8, 9, shape, dtype=np.int8)
            params[name + "_bias"] = rng.integers(-30, 31, shape[0], dtype=np.int32)
            if name != "fc2":
                params[name + "_multiplier"] = np.array(1 << 20, np.int64)
        checkpoint = self.root / "synthetic.npz"
        np.savez(checkpoint, **params)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(model_cli(["--manifest", str(self.manifest), "--parameters", str(checkpoint),
                                        "--encoder", "grayscale8"]), 0)
        return ModelSlot(self.manifest), checkpoint

    def test_empty_slot_never_infers(self):
        pipeline = EyePipeline(ModelSlot(self.manifest))
        self.addCleanup(pipeline.close)
        with patch("live_pipeline.infer", side_effect=AssertionError("Must not infer")):
            result = pipeline.predict(9, 1000000, {"left": np.zeros((16, 32), np.uint8)})
        self.assertEqual((result.frame_id, result.capture_ns), (9, 1000000))
        self.assertIsNone(pipeline.pool)
        self.assertTrue(all(eye["state"] == "UNAVAILABLE" and eye["logits"] is None
                            for eye in result.eyes.values()))

    def test_one_and_two_workers_match_integer_reference(self):
        model, _ = self.activate()
        rois = {"left": np.arange(512, dtype=np.uint8).reshape(16, 32),
                "right": np.full((16, 32), 201, np.uint8)}
        expected = infer(np.stack(list(rois.values()))[:, None], model.params)
        for workers in (1, 2):
            pipeline = EyePipeline(model, workers)
            try:
                result = pipeline.predict(7, 44000000, rois)
                np.testing.assert_array_equal([result.eyes[side]["logits"] for side in rois], expected)
                self.assertEqual((result.frame_id, result.capture_ns), (7, 44000000))
            finally:
                pipeline.close()
        self.assertFalse(model.params["conv1_weight"].flags.writeable)

    def test_all_encoders_match_independent_scalar_arithmetic(self):
        _, checkpoint = self.activate()
        rng = np.random.default_rng(81)
        rois = {"left": rng.integers(0, 256, (16, 32), dtype=np.uint8),
                "right": np.arange(512, dtype=np.uint8).reshape(16, 32)}
        for method, encoder in METHODS.items():
            with self.subTest(encoder=method):
                with redirect_stdout(io.StringIO()):
                    model_cli(["--manifest", str(self.manifest), "--parameters", str(checkpoint), "--encoder", method])
                model = ModelSlot(self.manifest)
                pipeline = EyePipeline(model, workers=2)
                try:
                    result = pipeline.predict(19, 70000000, rois)
                    for side, roi in rois.items():
                        codes = encoder.encode(roi)
                        pixels = codes if method == "grayscale8" else codes * np.uint8(85)
                        self.assertEqual(result.eyes[side]["logits"], scalar_logits(pixels, model.params))
                finally:
                    pipeline.close()

    def test_extreme_integer_values_match_scalar_arithmetic(self):
        model, checkpoint = self.activate()
        params = {name: value.copy() for name, value in model.params.items()}
        params["conv1_weight"][:] = 0
        params["conv1_bias"][:] = [-2147483647, -1, 1, 2147483647]
        params["conv1_multiplier"] = np.array(0xffffffff, np.uint32)
        params["conv2_multiplier"] = np.array(1 << 23, np.int64)
        params["fc1_multiplier"] = np.array(0xffffffff, np.uint32)
        params["fc1_bias"][0] = 1048576
        params["fc2_weight"][0, 0] = -128
        np.savez(checkpoint, **params)
        checked = read_parameters(checkpoint)
        for value in (0, 255):
            pixels = np.full((16, 32), value, np.uint8)
            np.testing.assert_array_equal(infer(pixels[None, None], checked)[0], scalar_logits(pixels, checked))

    def test_q24_rounding_and_saturation_at_integer_limits(self):
        values = np.array([-2147483647, -1, 0, 1, 2, 255, 510, 2147483647], dtype=np.int32)
        for multiplier in (1, (1 << 23) - 1, 1 << 23, 0xffffffff):
            expected = [min(255, (max(int(value), 0) * multiplier + (1 << 23)) >> 24) - 128
                        for value in values]
            np.testing.assert_array_equal(requantize(values, multiplier), expected)

    def test_two_independent_eye_jobs_can_overlap(self):
        model, _ = self.activate()
        pipeline = EyePipeline(model, workers=2)
        self.addCleanup(pipeline.close)
        barrier, threads = Barrier(2), []

        def worker(pixels, params):
            threads.append(get_ident())
            barrier.wait(timeout=3)
            return np.array([[5, -5]], dtype=np.int32)

        with patch("live_pipeline.infer", side_effect=worker):
            result = pipeline.predict(1, 1000000, {side: np.zeros((16, 32), np.uint8) for side in ("left", "right")})
        self.assertEqual(len(set(threads)), 2)
        self.assertEqual(result.eyes["left"]["state"], "OPEN")

    def test_missing_eye_does_not_reuse_previous_logits(self):
        model, _ = self.activate()
        pipeline = EyePipeline(model)
        self.addCleanup(pipeline.close)
        roi = np.zeros((16, 32), np.uint8)
        pipeline.predict(1, 1000000, {"left": roi, "right": roi})
        result = pipeline.predict(2, 2000000, {"left": roi})
        self.assertEqual(result.eyes["right"], {"state": "UNAVAILABLE", "logits": None, "margin": None})

    def test_tied_logits_are_unavailable(self):
        model, _ = self.activate()
        pipeline = EyePipeline(model)
        self.addCleanup(pipeline.close)
        with patch("live_pipeline.infer", return_value=np.array([[0, 0]], np.int32)):
            result = pipeline.predict(1, 1000000, {"left": np.zeros((16, 32), np.uint8)})
        self.assertEqual(result.eyes["left"]["state"], "UNAVAILABLE")

    def test_modified_checkpoint_is_rejected(self):
        _, checkpoint = self.activate()
        checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            ModelSlot(self.manifest)

    def test_hash_and_parameters_use_the_same_file_snapshot(self):
        expected, checkpoint = self.activate()
        original_read = Path.read_bytes
        reads = []

        def replace_after_read(path):
            payload = original_read(path)
            if path == checkpoint:
                reads.append(path)
                changed = {name: value.copy() for name, value in expected.params.items()}
                changed["conv1_weight"][:] = 0
                np.savez(checkpoint, **changed)
            return payload

        with patch.object(Path, "read_bytes", replace_after_read):
            loaded = ModelSlot(self.manifest)
        np.testing.assert_array_equal(loaded.params["conv1_weight"], expected.params["conv1_weight"])
        self.assertEqual(len(reads), 1)

    def test_failed_activation_keeps_the_previous_manifest(self):
        _, checkpoint = self.activate()
        previous = self.manifest.read_bytes()
        with patch("live_pipeline.ModelSlot", side_effect=ValueError("Checkpoint changed during activation")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                model_cli(["--manifest", str(self.manifest), "--parameters", str(checkpoint), "--encoder", "aeee2"])
        self.assertEqual(self.manifest.read_bytes(), previous)

    def test_invalid_manifest_types_are_rejected_cleanly(self):
        original = json.loads(self.manifest.read_text(encoding="utf-8"))
        for record in ([], dict(original, encoder=[])):
            self.manifest.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ValueError):
                ModelSlot(self.manifest)

    def test_invalid_tensor_shape_is_rejected(self):
        model, checkpoint = self.activate()
        params = dict(model.params)
        params["conv1_weight"] = np.zeros((4, 1, 5, 5), np.int8)
        np.savez(checkpoint, **params)
        with self.assertRaisesRegex(ValueError, "weight shape"):
            read_parameters(checkpoint)

    def test_wrong_archive_type_and_accumulator_overflow_are_rejected(self):
        single = self.root / "single.npy"
        np.save(single, np.zeros((16, 32), np.uint8))
        with self.assertRaisesRegex(ValueError, "NPZ"):
            read_parameters(single)
        model, checkpoint = self.activate()
        params = dict(model.params)
        params["fc1_bias"] = np.full(8, np.iinfo(np.int32).max, np.int32)
        np.savez(checkpoint, **params)
        with self.assertRaisesRegex(ValueError, "accumulator bound"):
            read_parameters(checkpoint)

    def test_fractional_multiplier_is_rejected(self):
        model, checkpoint = self.activate()
        params = dict(model.params)
        params["conv1_multiplier"] = np.array(1.5)
        np.savez(checkpoint, **params)
        with self.assertRaisesRegex(ValueError, "integer scalars"):
            read_parameters(checkpoint)

    def test_empty_fpga_mode_stops_before_device_initialization(self):
        with patch("live_pipeline.FPGAClient", side_effect=AssertionError("Must not open UART")):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    live_cli(["--manifest", str(self.manifest), "--backend", "fpga"])
        self.assertEqual(raised.exception.code, 1)

    def test_status_never_opens_devices(self):
        with patch("live_pipeline.FPGAClient", side_effect=AssertionError("Must not open UART")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(live_cli(["--manifest", str(self.manifest), "--status"]), 0)

    def test_fpga_path_does_not_duplicate_cnn_unless_verifying(self):
        model, _ = self.activate()
        rois = {side: np.full((16, 32), 130, np.uint8) for side in ("left", "right")}
        logits = infer(np.stack(list(rois.values()))[:, None], model.params)
        client = Mock()
        client.infer.return_value = HardwareResult(15, 0, 0, tuple(logits[0]), tuple(logits[1]), 1, 1, 2, 1)
        pipeline = EyePipeline(model, client=client)
        with patch.object(pipeline, "_infer_eye", side_effect=AssertionError("Duplicate CPU CNN")):
            result = pipeline.predict(15, 1000000, rois)
        self.assertIsNone(pipeline.pool)
        np.testing.assert_array_equal([result.eyes[side]["logits"] for side in rois], logits)
        self.assertEqual(client.infer.call_count, 1)
        self.assertEqual(result.hardware_cycles, (1, 1, 2))
        verifying = EyePipeline(model, client=client, verify=True)
        with patch.object(verifying, "_infer_eye", return_value=np.array([1234567, -1234567], np.int32)):
            with self.assertRaisesRegex(RuntimeError, "logits differ"):
                verifying.predict(16, 2000000, rois)

    @staticmethod
    def observation(ms, left="OPEN", right="OPEN"):
        return PairResult(ms, (1000 + ms) * 1000000,
                          {"left": {"state": left}, "right": {"state": right}}, 0)

    def test_bilateral_cycle_keeps_capture_duration(self):
        timer = BinocularTimer()
        for ms, state in ((0, "OPEN"), (33, "CLOSED"), (66, "OPEN")):
            observation = self.observation(ms, state, state)
            result, _ = update_timing(timer, observation, observation.capture_ns + 10000000)
        self.assertEqual((result.count, result.last_duration_ms), (1, 33))

    def test_stale_closure_and_wink_do_not_count(self):
        for stale in (False, True):
            timer = BinocularTimer()
            first = self.observation(0)
            update_timing(timer, first, first.capture_ns)
            closed = self.observation(33, "CLOSED", "CLOSED" if stale else "OPEN")
            result, scores = update_timing(timer, closed, closed.capture_ns + (300000000 if stale else 0))
            if stale:
                self.assertEqual(scores, [None, None])
            reopened = self.observation(66)
            result, _ = update_timing(timer, reopened, reopened.capture_ns)
            self.assertEqual(result.count, 0)

    def test_stale_reopening_is_invalid_before_uart_dispatch(self):
        timer, client = BinocularTimer(), TimingLink()
        for ms, state, age in ((0, "OPEN", 0), (33, "CLOSED", 0), (66, "OPEN", 300000000)):
            observation = self.observation(ms, state, state)
            now = observation.capture_ns + age
            with patch("live_pipeline.time.monotonic_ns", return_value=now):
                timing, scores, _ = synchronize_timing(timer, observation, client)
        self.assertEqual(scores, [None, None])
        self.assertEqual(client.samples[-1][:2], (None, None))
        self.assertEqual((timing.count, timer.count, client.timer.count), (0, 0, 0))

    def test_delayed_uart_reply_hides_live_state_and_preserves_accepted_history(self):
        timer, client = BinocularTimer(), TimingLink()
        for ms, state in ((0, "OPEN"), (33, "CLOSED")):
            observation = self.observation(ms, state, state)
            with patch("live_pipeline.time.monotonic_ns", return_value=observation.capture_ns):
                synchronize_timing(timer, observation, client)
        reopened = self.observation(66)
        finished_ns = reopened.capture_ns + 300000000
        with patch("live_pipeline.time.monotonic_ns", side_effect=[reopened.capture_ns, finished_ns]):
            timing, scores, published_ns = synchronize_timing(timer, reopened, client)
        self.assertEqual(scores, [None, None])
        self.assertEqual((timing.count, timer.count, client.timer.count), (1, 1, 1))
        self.assertEqual((timer.valid_mask, client.timer.valid_mask), (0, 0))
        self.assertEqual((timing.last_duration_ms, published_ns), (33, finished_ns))
        for ms, state in ((99, "CLOSED"), (132, "OPEN")):
            observation = self.observation(ms, state, state)
            with patch("live_pipeline.time.monotonic_ns", return_value=observation.capture_ns):
                timing, _, _ = synchronize_timing(timer, observation, client)
        self.assertEqual(timing.count, 1)

    def test_live_empty_model_report_and_csv_without_camera_or_model_inference(self):
        import csv
        from ocular_metrics import observations

        def fake_capture(pending, errors, stop, stopped, camera):
            pending.put((1, time.monotonic_ns(), np.zeros((240, 640, 3), np.uint8)))
            stop.wait(timeout=3)
            stopped.set()

        task = Mock()
        task.detect_for_video.return_value = SimpleNamespace(face_landmarks=[])
        report, frames = self.root / "report.json", self.root / "frames.csv"
        with patch("capture.capture_frames", fake_capture), \
             patch("capture.vision.FaceLandmarker.create_from_options", return_value=task), \
             patch("cv2.namedWindow"), patch("cv2.imshow"), patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"), redirect_stdout(io.StringIO()):
            self.assertEqual(live_cli(["--manifest", str(self.manifest), "--report", str(report), "--frames", str(frames)]), 0)
        saved = json.loads(report.read_text())
        self.assertEqual((saved["model_state"], saved["frames"], saved["valid_eye_pairs"]), ("untrained", 1, 0))
        self.assertIsNone(saved["closures"]["total"])
        with frames.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual((rows[0]["left_state"], rows[0]["right_state"]), ("UNAVAILABLE", "UNAVAILABLE"))
        self.assertIsNone(observations(frames)[0]["left_open_fraction"])
        task.close.assert_called_once()

    def test_live_camera_failure_writes_failure_report_and_releases_landmarker(self):
        def fake_capture(pending, errors, stop, stopped, camera):
            errors.put("Synthetic capture failure")
            stopped.set()

        task = Mock()
        report = self.root / "failure.json"
        with patch("capture.capture_frames", fake_capture), \
             patch("capture.vision.FaceLandmarker.create_from_options", return_value=task), \
             patch("cv2.namedWindow"), patch("cv2.destroyAllWindows"), redirect_stdout(io.StringIO()):
            self.assertEqual(live_cli(["--manifest", str(self.manifest), "--report", str(report)]), 1)
        saved = json.loads(report.read_text())
        self.assertEqual(saved["status"], "FAIL")
        self.assertEqual(saved["error"], "Synthetic capture failure")
        self.assertIsNone(saved["after_read_mean_ms"])
        task.close.assert_called_once()


class StreamingTests(unittest.TestCase):
    """Deterministic worker contracts; these fake transports are never physical evidence."""
    def make(self, values=(1,2,3), mode='sequential', block=False, failure=None, missing=None, clock=None):
        from threading import Event, Lock
        from stream_pipeline import StreamingPipeline
        started,release,next_prepared=Event(),Event(),Event()
        now=[1000000000000]; guard=Lock()
        def ticks():
            with guard: now[0]+=1000000; return now[0]
        clock=clock or ticks
        events=[]; holder={}; closed=Event()
        model=SimpleNamespace(checkpoint_sha256='4daa1000'+'0'*56,ready=True,method='omz_bgr32',predict=lambda roi,fixed=True:np.array([int(roi[0,0,0]),255-int(roi[0,0,0])]))
        class Client:
            def __init__(self): self.owner=get_ident();self.reader=SimpleNamespace(bad_packets=0);self.timeout_count=self.unmatched_responses=0
            def check(self):
                if get_ident()!=self.owner: raise AssertionError('UART used outside its owner')
            def load_model(self,model):
                self.check();return dict(method_id=4,model_loaded=True,fingerprint=0 if failure=='firmware' else 0x4daa1000)
            def configure(self,*args): self.check()
            def invalidate(self,*args): self.check()
            def infer_pretrained(self,left,right,frame_id):
                self.check();started.set()
                if block and frame_id==1:
                    if not release.wait(2): raise TimeoutError('Test synchronization deadline')
                if failure=='timeout': self.timeout_count+=1;raise TimeoutError('Synthetic UART timeout')
                if failure=='crc': self.reader.bad_packets+=1
                logits=[tuple(model.predict(roi)) for roi in (left,right)]
                return HardwareResult(frame_id+1 if failure=='id' else frame_id,4,0,*logits,367739,367739,735478,.1)
            def close(self): self.check();closed.set()
        def source(stop):
            for i,value in enumerate(values,1):
                if i==2 and block:
                    if not started.wait(2): raise RuntimeError('FPGA did not start')
                if failure=='camera': raise RuntimeError('Synthetic camera disconnect')
                yield i,clock(),np.full((4,4,3),value,np.uint8)
        def prepare(frame,ns):
            if failure=='mediapipe': raise RuntimeError('Synthetic MediaPipe failure')
            if int(frame[0,0,0])==2: next_prepared.set()
            return ({} if int(frame[0,0,0])==missing else dict(left=frame,right=frame)),{}
        pipe=StreamingPipeline(source,lambda:prepare,Client,model,mode=mode,verify=True,emit=events.append,clock=clock)
        return pipe,events,started,release,next_prepared,closed,now

    def finish(self,pipe):
        self.assertTrue(pipe.done.wait(4),'Pipeline did not finish')
        pipe.close();pipe.raise_error()
        self.assertFalse(any(t.is_alive() for t in pipe.threads))

    def test_stream_order_identity_readonly_and_cpu_equality(self):
        pipe,events,*_=self.make()
        pipe.start();self.finish(pipe)
        rows=[e for e in events if e['kind']=='published']
        self.assertEqual([e['frame_id'] for e in rows],[1,2,3])
        self.assertEqual([e['raw_logits'][0] for e in rows],[[1,254],[2,253],[3,252]])
        self.assertEqual(len({e['capture_ns'] for e in rows}),3)
        self.assertFalse(pipe.display.get().inference.prepared.frame.flags.writeable)

    def test_real_worker_overlap_latest_policy_and_bounded_queues(self):
        pipe,events,started,release,prepared,closed,_=self.make(tuple(range(1,9)),mode='overlap',block=True)
        try:
            pipe.start();self.assertTrue(started.wait(2));self.assertTrue(prepared.wait(2))
            self.assertTrue(pipe.prep_done.wait(2))
            self.assertEqual(pipe.prepared.qsize(),1);self.assertEqual(pipe.inferred.maxsize,2)
            self.assertEqual(pipe.counts['dropped_preparation'],6)
            self.assertFalse(release.is_set())
        finally: release.set();self.finish(pipe)
        self.assertEqual([e['frame_id'] for e in events if e['kind']=='published'],[1,8])
        self.assertTrue(closed.is_set())

    def test_missing_eyes_never_reuse_previous_logits(self):
        pipe,events,*_=self.make(missing=2)
        pipe.start();self.finish(pipe)
        row=next(e for e in events if e['kind']=='published' and e['frame_id']==2)
        self.assertIsNone(row['raw_logits']);self.assertEqual(row['eyes']['left']['state'],'UNAVAILABLE')

    def test_failures_propagate_without_cpu_fallback(self):
        for failure in ('timeout','crc','firmware','id','camera','mediapipe'):
            with self.subTest(failure=failure):
                pipe,events,*rest=self.make(failure=failure)
                pipe.start();self.assertTrue(pipe.done.wait(3));pipe.close()
                with self.assertRaises(RuntimeError):pipe.raise_error()
                self.assertEqual(pipe.counts['fpga_completed'],0 if failure in ('timeout','firmware','id','camera','mediapipe') else 1)
                self.assertTrue(rest[-2].is_set())
                self.assertFalse(any(t.is_alive() for t in pipe.threads))

    def test_temporal_sequence_matches_existing_timer(self):
        values=(255,255,0,0,255,255);now=[1000000000000]
        pipe,events,*_=self.make(values,clock=lambda:now[0])
        def source(stop):
            for i,value in enumerate(values):
                now[0]=1000000000000+i*60000000
                yield i+1,now[0],np.full((4,4,3),value,np.uint8)
        pipe.source_factory=source
        pipe.start();self.finish(pipe)
        ref=BinocularTimer();expected=[]
        for i,value in enumerate(values):expected.append(ref.update(float(value==0),float(value==0),(1000000000000+i*60000000)//1000000).count)
        self.assertEqual([e['blink_count'] for e in events if e['kind']=='published'],expected)
        self.assertEqual(expected[-1],1)

    def test_stale_result_cannot_publish_old_state_or_blink(self):
        pipe,events,started,release,prepared,closed,now=self.make((1,),block=True)
        try:
            pipe.start();self.assertTrue(started.wait(2));now[0]+=300000000
        finally: release.set();self.finish(pipe)
        row=next(e for e in events if e['kind']=='published')
        self.assertEqual(row['reason'],'stale_result');self.assertEqual(row['eyes']['left']['state'],'UNAVAILABLE');self.assertFalse(row['blink_event'])

    def test_duplicate_frame_rejected(self):
        pipe,events,*_=self.make()
        def source(stop):
            yield 1,pipe.clock(),np.zeros((4,4,3),np.uint8)
            yield 1,pipe.clock(),np.zeros((4,4,3),np.uint8)
        pipe.source_factory=source
        pipe.start();self.assertTrue(pipe.done.wait(3));pipe.close()
        with self.assertRaisesRegex(RuntimeError,'duplicate_frame'):pipe.raise_error()

    def test_repeat_shutdown_no_worker_gui_calls(self):
        from threading import Event
        for _ in range(3):
            pipe,*_=self.make()
            with patch('cv2.imshow',side_effect=AssertionError('Worker GUI call')),patch('cv2.waitKey',side_effect=AssertionError('Worker GUI call')):
                pipe.start();self.finish(pipe)
            published=pipe.counts['published'];self.assertTrue(pipe.done.is_set());self.assertEqual(pipe.counts['published'],published)

    def test_metric_intersection_and_empty_hardware_not_a_speedup(self):
        from pipeline_report import intersection,compare
        self.assertEqual(intersection(0,10,5,15),5);self.assertEqual(intersection(0,3,4,5),0)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths=[]
            for mode in ('sequential','overlap'):
                folder=root/mode;folder.mkdir();paths.append(folder)
                (folder/'summary.json').write_text(json.dumps(dict(pipeline_mode=mode,status='PASS',processed_fps=0,scope='DETERMINISTIC_UNIT_TEST')))
                (folder/'session.json').write_text(json.dumps(dict(provenance='DETERMINISTIC_UNIT_TEST',source_sha256={})))
            result=compare(paths,root/'comparison')
            self.assertEqual(result['status'],'NOT_AVAILABLE');self.assertIsNone(result['throughput_ratio'])

    def test_prepare_and_source_resources_close_on_success_and_failure(self):
        from threading import Event
        for failure in (None,'mediapipe'):
            with self.subTest(failure=failure):
                pipe,*_=self.make(failure=failure)
                source_closed,prepare_closed=Event(),Event()
                original_source=pipe.source_factory; original_prepare=pipe.prepare_factory()
                def source(stop):
                    try: yield from original_source(stop)
                    finally: source_closed.set()
                class Prepare:
                    def __call__(self,frame,ns):return original_prepare(frame,ns)
                    def close(self):prepare_closed.set()
                pipe.source_factory=source;pipe.prepare_factory=Prepare
                pipe.start();self.assertTrue(pipe.done.wait(3));pipe.close()
                if failure:
                    with self.assertRaises(RuntimeError):pipe.raise_error()
                else:pipe.raise_error()
                self.assertTrue(source_closed.is_set());self.assertTrue(prepare_closed.is_set())

    def test_future_capture_is_not_inferred(self):
        pipe,events,*_=self.make((1,))
        def source(stop):yield 1,pipe.clock()+500000000,np.zeros((4,4,3),np.uint8)
        pipe.source_factory=source;pipe.start();self.finish(pipe)
        row=next(e for e in events if e['kind']=='published')
        self.assertEqual(row['reason'],'stale_result');self.assertIsNone(row['raw_logits'])
        self.assertEqual(pipe.counts['submitted'],0)

    def test_result_backpressure_preserves_all_accepted_results(self):
        from threading import Event
        pipe,events,*_=self.make(tuple(range(1,6)),mode='overlap')
        seen={i:Event() for i in range(1,6)};result_started,release=Event(),Event()
        original_factory=pipe.client_factory
        def factory():
            client=original_factory()
            def infer(left,right,frame_id):
                client.check();value=int(left[0,0,0]);seen[frame_id].set()
                return HardwareResult(frame_id,4,0,(value,255-value),(value,255-value),367739,367739,735478,.1)
            client.infer_pretrained=infer
            return client
        def source(stop):
            for i in range(1,6):
                if i>1 and not seen[i-1].wait(2):raise RuntimeError('Source/FPGA synchronization timeout')
                yield i,pipe.clock(),np.full((4,4,3),i,np.uint8)
        def predict(roi,fixed=True):
            result_started.set()
            if not release.wait(2):raise RuntimeError('Result synchronization timeout')
            value=int(roi[0,0,0]);return np.array([value,255-value])
        pipe.source_factory=source;pipe.client_factory=factory;pipe.model.predict=predict
        try:
            pipe.start();self.assertTrue(result_started.wait(2));self.assertTrue(seen[4].wait(2))
            self.assertEqual(pipe.inferred.qsize(),2)
        finally:release.set();self.finish(pipe)
        self.assertEqual([e['frame_id'] for e in events if e['kind']=='published'],list(range(1,6)))


if __name__ == "__main__":
    unittest.main()
