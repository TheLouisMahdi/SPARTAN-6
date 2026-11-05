"""Pretrained or explicit TinyEye inference with capture-timed bilateral events.

Default: pinned pretrained CNN. An explicitly selected empty TinyEye slot produces
UNAVAILABLE, never an OPEN/CLOSED guess or a MediaPipe expression-score fallback.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np

from blink_timing import BinocularTimer, MAXIMUM_GAP_MS
from encoders import METHODS
from encoders.common import validate_roi
from fpga_host import FPGAClient, model_image
from integer_model import infer

ROOT = Path(__file__).resolve().parents[1]
MODEL_SLOT = ROOT / "assets/models/eye_model.json"
SIDES = ("left", "right")
LAYERS = ("conv1", "conv2", "fc1", "fc2")


def read_parameters(source):
    """Validate the existing INT8 UART contract before starting any device."""
    payload = source if isinstance(source, bytes) else Path(source).read_bytes()
    archive = np.load(BytesIO(payload), allow_pickle=False)
    if isinstance(archive, np.ndarray):
        raise ValueError("Checkpoint must be an NPZ archive, not a single NPY array")
    with archive:
        params = {name: archive[name] for name in archive.files}
    expected = {name + suffix for name in LAYERS for suffix in ("_weight", "_bias")}
    expected.update(name + "_multiplier" for name in LAYERS[:3])
    if set(params) != expected:
        raise ValueError("Checkpoint must contain the four layers' weights/biases and three multipliers")
    for name in LAYERS[:3]:
        multiplier = params[name + "_multiplier"]
        if multiplier.ndim != 0 or multiplier.dtype.kind not in "iu":
            raise ValueError("Q24 multipliers must be integer scalars")
    model_image(params)  # Shapes, dtypes, INT32 bounds and uint32 Q24 multipliers.
    for value in params.values():
        value.flags.writeable = False
    return params


class ModelSlot:
    """One explicitly activated checkpoint; architecture is a deployment contract."""

    def __init__(self, path=MODEL_SLOT):
        self.path = Path(path)
        self.record = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if (not isinstance(self.record, dict) or self.record.get("schema_version") != 1
                or self.record.get("architecture") != "tiny-eye-v1"
                or self.record.get("input") != [1, 16, 32]
                or self.record.get("classes") != ["OPEN", "CLOSED"]):
            raise ValueError("Model manifest does not match the current FPGA tensor contract")
        self.method = self.record.get("encoder")
        if not isinstance(self.method, str) or self.method not in METHODS:
            raise ValueError("Unknown model encoder")
        self.params = None
        self.checkpoint_sha256 = None
        if self.record.get("state") == "untrained":
            if self.record.get("checkpoint") is not None or self.record.get("sha256") is not None:
                raise ValueError("An untrained model slot must not reference a checkpoint")
            return
        if self.record.get("state") != "trained":
            raise ValueError("Model state must be untrained or trained")
        checkpoint = self.record.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError("Trained model requires an explicit checkpoint path")
        if self.record.get("encoder_spec") != METHODS[self.method].SPEC:
            raise ValueError("Checkpoint encoder specification differs from the installed encoder")
        checkpoint = self.path.parent / checkpoint
        payload = checkpoint.read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
        if self.checkpoint_sha256 != self.record.get("sha256"):
            raise ValueError("Checkpoint hash differs from the activated model")
        self.params = read_parameters(payload)

    @property
    def ready(self):
        return self.params is not None


@dataclass(frozen=True)
class PairResult:
    frame_id: int
    capture_ns: int
    eyes: dict
    processing_ms: float
    hardware_cycles: tuple | None = None


class EyePipeline:
    """Persistent eye workers; each result belongs to one original camera frame.

    NumPy performs the integer dot products. Two workers process independent
    eyes, not two concurrent requests on the same UART connection. Whether this
    is faster than one worker must be measured on the target laptop.
    """

    def __init__(self, model, workers=2, client=None, verify=False):
        if workers not in (1, 2):
            raise ValueError("Eye workers must be 1 or 2")
        if client is not None and not model.ready:
            raise ValueError("FPGA inference requires an explicitly trained model")
        self.model, self.client, self.verify = model, client, verify
        self.pool = ThreadPoolExecutor(max_workers=workers) if model.ready and client is None else None

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def _codes(self, roi):
        validate_roi(roi)
        return METHODS[self.model.method].encode(roi)

    def _infer_eye(self, roi):
        if self.model.method == "omz_bgr32":
            return self.model.predict(roi)
        codes = self._codes(roi)
        pixels = codes if self.model.method == "grayscale8" else codes * np.uint8(85)
        return infer(pixels[None, None], self.model.params)[0]

    def predict(self, frame_id, capture_ns, rois):
        started = time.perf_counter_ns()
        eyes = {side: {"state": "UNAVAILABLE", "logits": None, "margin": None} for side in SIDES}
        available = {side: rois[side] for side in SIDES if rois.get(side) is not None}
        scores = {}
        hardware_cycles = None
        if self.model.ready and self.client is not None:
            if len(available) == 2:
                if self.model.method == "omz_bgr32":
                    hardware = self.client.infer_pretrained(*(available[side] for side in SIDES), frame_id)
                else:
                    codes = [self._codes(available[side]) for side in SIDES]
                    hardware = self.client.infer(self.model.method, *codes, frame_id)
                hardware_cycles = (hardware.left_cycles, hardware.right_cycles, hardware.pair_cycles)
                scores = dict(zip(SIDES, (hardware.left_logits, hardware.right_logits)))
                if self.verify:
                    for side in SIDES:
                        reference = (self.model.predict(available[side], fixed=True) if self.model.method == "omz_bgr32"
                                     else self._infer_eye(available[side]))
                        if not np.array_equal(scores[side], reference):
                            raise RuntimeError("FPGA logits differ from the independent integer reference")
            else:
                self.client.invalidate(frame_id)
        elif self.model.ready:
            jobs = {side: self.pool.submit(self._infer_eye, roi) for side, roi in available.items()}
            scores = {side: job.result() for side, job in jobs.items()}
        for side, logits in scores.items():
            convert = float if self.model.method == "omz_bgr32" and self.client is None else int
            logits = [convert(value) for value in logits]
            margin = logits[1] - logits[0]
            # Equal logits give no evidence for either state. Scores are not probabilities.
            state = "CLOSED" if margin > 0 else "OPEN" if margin < 0 else "UNAVAILABLE"
            eyes[side] = {"state": state, "logits": logits, "margin": margin}
        return PairResult(frame_id, capture_ns, eyes, (time.perf_counter_ns() - started) / 1e6, hardware_cycles)


def update_timing(timer, result, now_ns, max_age_ms=MAXIMUM_GAP_MS):
    """Freshness is checked at acceptance; unavailable inputs cancel partial cycles."""
    fresh = 0 <= now_ns - result.capture_ns <= max_age_ms * 1_000_000
    scores = [None if not fresh or result.eyes[side]["state"] == "UNAVAILABLE"
              else float(result.eyes[side]["state"] == "CLOSED") for side in SIDES]
    timing = timer.update(*scores, (result.capture_ns // 1_000_000) & 0xffffffff)
    return timing, scores


def synchronize_timing(timer, result, client=None):
    """Mirror capture-time events; hide expired live states after a late reply.

    Invalidation cancels continuity, not completed history. A fresh observation
    accepted before UART dispatch can already have committed an event on the
    FPGA, so a delayed acknowledgement must not be presented as a rollback.
    """
    timing, scores = update_timing(timer, result, time.monotonic_ns())
    if client is not None:
        hardware, _ = client.eye_timing(*scores, (result.capture_ns // 1_000_000) & 0xffffffff, result.frame_id)
        if hardware != timing:
            raise RuntimeError("FPGA binocular timing differs from the capture-timed reference")
        timing = hardware
    finished_ns = time.monotonic_ns()
    if finished_ns - result.capture_ns > MAXIMUM_GAP_MS * 1_000_000:
        timer.invalidate()
        if client is not None:
            client.invalidate(result.frame_id)
        scores = [None, None]
    return timing, scores, finished_ns


def model_cli(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the empty model slot or explicitly activate future INT8 weights.")
    parser.add_argument("--manifest", type=Path, default=MODEL_SLOT)
    parser.add_argument("--parameters", type=Path, help="Explicit trained int8_parameters.npz; does not train or measure accuracy")
    parser.add_argument("--encoder", choices=tuple(METHODS), help="Required with --parameters; must match its training preprocessing")
    args = parser.parse_args(argv)
    if bool(args.parameters) != bool(args.encoder):
        parser.error("--parameters and --encoder must be supplied together")
    try:
        if args.parameters:
            payload = args.parameters.read_bytes()
            read_parameters(payload)
            record = {"schema_version": 1, "state": "trained", "architecture": "tiny-eye-v1",
                      "input": [1, 16, 32], "classes": ["OPEN", "CLOSED"], "encoder": args.encoder,
                      "encoder_spec": METHODS[args.encoder].SPEC,
                      "checkpoint": str(args.parameters.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                      "accuracy": "not established by activation"}
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.manifest.with_name(args.manifest.name + ".tmp")
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            try:
                model = ModelSlot(temporary)
                temporary.replace(args.manifest)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            model = ModelSlot(args.manifest)
        print(json.dumps({"state": model.record["state"], "ready": model.ready,
                          "architecture": model.record["architecture"], "encoder": model.method,
                          "checkpoint_sha256": model.checkpoint_sha256}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"MODEL ERROR: {exc}\n")


def live_cli(argv=None, forced_backend=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Explicit legacy TinyEye manifest; cannot be combined with --network pretrained")
    parser.add_argument("--network", choices=("pretrained", "tiny"), help="Default: pretrained; explicit --manifest selects TinyEye")
    parser.add_argument("--precision", choices=("float", "fixed"), default="float", help="CPU pretrained arithmetic; FPGA always uses fixed")
    parser.add_argument("--baud", type=int, help="Default: 1000000 for pretrained firmware, 115200 for TinyEye")
    parser.add_argument("--backend", choices=("cpu", "fpga"), default=forced_backend or "cpu")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--camera", type=int, default=1, help="Integrated Camera on this laptop; choose explicitly if devices change")
    parser.add_argument("--port", default="COM26", help="Opened only with --backend fpga")
    parser.add_argument("--verify", action="store_true", help="Also recompute FPGA logits on CPU; verification overhead is included")
    parser.add_argument("--seconds", type=float, default=0, help="0: run until Q/Esc")
    parser.add_argument("--status", action="store_true", help="Show readiness without opening camera or UART")
    parser.add_argument("--report", type=Path, default=ROOT / "assets/logs/live/latest.json")
    parser.add_argument("--frames", type=Path, help="New prediction CSV for offline ocular analysis; never ground-truth labels")
    args = parser.parse_args(argv)
    if args.camera < 0 or not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("Camera must be nonnegative; seconds must be finite and nonnegative")
    if forced_backend and args.backend != forced_backend:
        parser.error(f"This command requires --backend {forced_backend}")
    if args.verify and args.backend != "fpga":
        parser.error("--verify requires the FPGA backend")
    if args.manifest and args.network == "pretrained":
        parser.error("--manifest refers to TinyEye; omit it for the pretrained CNN")
    pretrained = args.network == "pretrained" or (args.network is None and args.manifest is None)
    if args.baud is not None and args.baud <= 0:
        parser.error("Baud rate must be positive")
    try:
        if pretrained:
            from pretrained_eye import PretrainedEye
            model = PretrainedEye(precision=args.precision)
        else:
            model = ModelSlot(args.manifest or MODEL_SLOT)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"MODEL ERROR: {exc}\n")
    print(f"MODEL: {model.record['architecture']}; {'ready' if model.ready else 'NOT TRAINED; eye states unavailable'}", flush=True)
    if args.status:
        print(f"BACKEND: {args.backend}; workers={args.workers}; camera/UART not opened")
        return 0
    if args.backend == "fpga" and not model.ready:
        parser.exit(1, "Train and explicitly activate a model before FPGA inference. No devices opened.\n")

    # Heavy device imports and initialization happen only after the readiness checks.
    import cv2
    from queue import Empty, Queue
    from threading import Event, Thread
    from capture import EYE_CORNERS, MODEL_PATH, capture_frames, extract_eye_roi, mp, vision, text

    pending, errors = Queue(maxsize=1), Queue(maxsize=1)
    stop, stopped = Event(), Event()
    camera_worker = landmarker = client = pipeline = None
    timer = BinocularTimer()
    frame_stream = writer = None
    frame_origin_ms = None
    stats = {"schema_version": 1, "backend": args.backend, "workers": args.workers,
             "model_state": model.record["state"], "checkpoint_sha256": model.checkpoint_sha256,
             "encoder": model.method, "camera": args.camera, "verify_cnn_on_cpu": args.verify,
             "architecture": model.record["architecture"],
             "cpu_precision": (args.precision if args.backend == "cpu" else "fixed" if args.verify else None) if pretrained else "int8",
             "baud": args.baud or (1000000 if pretrained else 115200),
             "frames": 0, "valid_eye_pairs": 0, "skipped_capture_frames": 0,
             "hardware_cnn_pairs": 0, "last_hardware_cycles": None,
             "hardware_clock_hz_from_project": 50000000,
             "pair_processing_sum_ms": 0.0, "pair_processing_max_ms": 0.0,
             "after_read_sum_ms": 0.0, "after_read_max_ms": 0.0,
             "images_saved": False, "training_performed": False, "calibration_performed": False,
             "accuracy": "not measured; prediction CSV is not ground truth",
             "timing": "capture timestamps; freshness at acceptance; expired display states are hidden",
             "aperture": "unavailable; binary logits do not measure eye-opening fraction"}
    previous_frame_id = 0
    window = "Eye pipeline | Q / Esc to quit"
    exit_code = 0
    try:
        if args.frames:
            args.frames.parent.mkdir(parents=True, exist_ok=True)
            frame_stream = args.frames.open("x", encoding="utf-8", newline="")
            writer = csv.DictWriter(frame_stream, fieldnames=("frame_id", "timestamp_ms", "left_state", "right_state",
                "left_open_fraction", "right_open_fraction", "both_event", "both_count", "long_count", "last_duration_ms", "after_read_ms",
                "fpga_left_cycles", "fpga_right_cycles", "fpga_pair_cycles"))
            writer.writeheader()
        if args.backend == "fpga":
            client = FPGAClient(args.port, baud=stats["baud"]) if pretrained or args.baud else FPGAClient(args.port)
            status = client.ping()
            if not status.get("binocular_timing"):
                raise RuntimeError("FPGA firmware does not support the existing binocular timer")
            client.load_model(model)
            client.configure(0, 0)
            client.reset_blinks()
        pipeline = EyePipeline(model, args.workers, client, args.verify)
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        camera_worker = Thread(target=capture_frames, args=(pending, errors, stop, stopped, args.camera), daemon=True)
        camera_worker.start()
        started = last_capture = time.monotonic()
        previous_ms = -1
        while not args.seconds or time.monotonic() - started < args.seconds:
            if stopped.is_set():
                raise RuntimeError(errors.get_nowait() if not errors.empty() else "Camera stopped")
            try:
                frame_id, captured_ns, frame = pending.get(timeout=0.05)
            except Empty:
                if time.monotonic() - last_capture > 5:
                    raise RuntimeError("Camera stalled for five seconds")
                if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q"), 27):
                    break
                continue
            last_capture = time.monotonic()
            captured_ms = captured_ns // 1_000_000
            if captured_ms <= previous_ms:
                continue
            previous_ms = captured_ms
            rois, boxes = {}, {}
            if time.monotonic_ns() - captured_ns <= MAXIMUM_GAP_MS * 1_000_000:
                landmarks = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), captured_ms)
                if landmarks.face_landmarks:
                    for side in SIDES:
                        roi, box = extract_eye_roi(frame, landmarks.face_landmarks[0], EYE_CORNERS[side])
                        if roi is not None:
                            if pretrained:
                                # Keep original BGR data; never reconstruct color from the legacy gray thumbnail.
                                x0, y0, x1, y1 = box
                                roi = frame[y0:y1, x0:x1].copy()
                            rois[side], boxes[side] = roi, box
            # Landmark processing itself can consume the age budget.
            if time.monotonic_ns() - captured_ns > MAXIMUM_GAP_MS * 1_000_000:
                rois.clear()
            result = pipeline.predict(frame_id, captured_ns, rois)
            if stopped.is_set():
                raise RuntimeError("Camera stopped while processing the frame")
            timing, scores, now_ns = synchronize_timing(timer, result, client)
            age_ms = (now_ns - captured_ns) / 1e6
            states = ["UNAVAILABLE" if not timing.valid_mask & (1 << index)
                      else "CLOSED" if timing.closed_mask & (1 << index) else "OPEN" for index in range(2)]
            stats["frames"] += 1
            stats["valid_eye_pairs"] += timing.valid_mask == 3
            stats["skipped_capture_frames"] += max(0, frame_id - previous_frame_id - 1)
            previous_frame_id = frame_id
            stats["pair_processing_sum_ms"] += result.processing_ms
            stats["pair_processing_max_ms"] = max(stats["pair_processing_max_ms"], result.processing_ms)
            stats["after_read_sum_ms"] += age_ms
            stats["after_read_max_ms"] = max(stats["after_read_max_ms"], age_ms)
            if result.hardware_cycles is not None:
                stats["hardware_cnn_pairs"] += 1
                stats["last_hardware_cycles"] = dict(zip(("left", "right", "pair"), result.hardware_cycles))
            if frame_origin_ms is None:
                frame_origin_ms = captured_ms
            if writer:
                writer.writerow(dict(frame_id=frame_id, timestamp_ms=captured_ms - frame_origin_ms,
                    left_state=states[0], right_state=states[1], left_open_fraction="", right_open_fraction="",
                    both_event=int(timing.event), both_count=timing.count, long_count=timing.long_count,
                    last_duration_ms=timing.last_duration_ms, after_read_ms=age_ms,
                    **dict(zip(("fpga_left_cycles", "fpga_right_cycles", "fpga_pair_cycles"), result.hardware_cycles or ("", "", "")))))
            display = frame.copy()
            for index, side in enumerate(SIDES):
                state = "UNAVAILABLE" if scores[index] is None else "CLOSED" if timing.closed_mask & (1 << index) else "OPEN"
                text(display, f"{side.upper()}: {state}", 10, 55 + index * 25)
                if side in boxes:
                    x0, y0, x1, y1 = boxes[side]
                    cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 220), 1)
            title = f"{args.backend.upper()} {model.record['architecture']}" if model.ready else "MODEL NOT TRAINED | ROI preview only"
            text(display, title, 10, 25)
            totals = (str(timing.count), str(timing.long_count)) if stats["valid_eye_pairs"] else ("--", "--")
            last_duration = str(timing.last_duration_ms) if timing.count else "--"
            text(display, f"BOTH-eye cycles: {totals[0]} | long: {totals[1]} | last: {last_duration} ms", 10, 110)
            text(display, f"Pair processing: {result.processing_ms:.1f} ms | age since read: {(now_ns-captured_ns)/1e6:.1f} ms", 10, 135)
            current = [str(value) if scores[index] is not None else "--"
                       for index, value in enumerate((timing.left_current_ms, timing.right_current_ms))]
            text(display, f"Current closure L/R: {current[0]}/{current[1]} ms", 10, 160)
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xff in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        stats["error"] = str(exc)
        print(f"LIVE ERROR: {exc}", flush=True)
        exit_code = 1
    finally:
        stop.set()
        if camera_worker:
            camera_worker.join(timeout=2)
        if pipeline:
            pipeline.close()
        if frame_stream:
            frame_stream.close()
        if landmarker:
            landmarker.close()
        if client:
            try:
                client.invalidate(0)
            except Exception as exc:
                print(f"FPGA shutdown error: {exc}", flush=True)
                exit_code = 1
            finally:
                client.close()
        cv2.destroyAllWindows()
        stats["closures"] = {"total": timer.count if stats["valid_eye_pairs"] else None,
                             "long": timer.long_count if stats["valid_eye_pairs"] else None,
                             "last_duration_ms": timer.last_duration_ms if timer.count else None}
        for prefix in ("pair_processing", "after_read"):
            stats[prefix + "_mean_ms"] = stats.pop(prefix + "_sum_ms") / stats["frames"] if stats["frames"] else None
        stats["status"] = "PASS" if exit_code == 0 else "FAIL"
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_name(args.report.name + ".tmp")
            temporary.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.report)
        except OSError as exc:
            print(f"REPORT ERROR: {exc}", flush=True)
            exit_code = 1
        print("LIVE SUMMARY " + json.dumps(stats), flush=True)
    return exit_code
