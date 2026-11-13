"""Unified entry point for FPGA, host, pretrained and offline eye inference."""

import sys
sys.dont_write_bytecode = True
import argparse
import json
import time
import cv2
import numpy as np
import hashlib
import csv
import math
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from uuid import uuid4
from data_tools import sha256
from encoders import METHODS
from eye_state import ManualThresholds, BlinkDetector, BlinkEventLog, update_blinks, ScoreCalibration, CalibrationUI, LiveObservation, decide, StableEyeStates, BilateralBlinkDetector
from integer_model import infer
from capture import EYE_CORNERS, MODEL_PATH, capture_frames, extract_eye_roi, mp, vision, ROOT, CODE_ROOT, ASSETS
from fpga_host import FPGAClient
from blink_timing import BinocularTimer
from dataset import read_roi

# Live Inference

WINDOW = "AEEE live eye inference | LAPTOP INT8 | Q / ESC to quit"
METHOD_KEYS = dict(zip(map(ord, "1234"), METHODS))
COLORS = {"OPEN": (90, 225, 120), "CLOSED": (75, 150, 255), "UNAVAILABLE": (150, 160, 175)}
PANEL_HEIGHT = 940
SLIDERS = {"left": (85, 470), "right": (660, 1045)}


class ThresholdUI:
    def __init__(self, settings, limit=100000):
        self.settings = settings
        self.limit = limit
        self.step = max(1, limit // 1000)
        self.dragging = None
        self.save_requested = False

    def set_from_x(self, side, x):
        start, end = SLIDERS[side]
        fraction = min(1, max(0, (x - start) / (end - start)))
        value = round(((2 * fraction - 1) * self.limit) / self.step) * self.step
        self.settings.set(side, value)

    def mouse(self, event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 738 <= y <= 777:
                for side, (start, end) in SLIDERS.items():
                    if start <= x <= end:
                        self.dragging = side
                        self.set_from_x(side, x)
                        return
                    if start - 60 <= x <= start - 20:
                        self.settings.set(side, max(-self.limit, self.settings.value(side) - self.step))
                        return
                    if end + 20 <= x <= end + 60:
                        self.settings.set(side, min(self.limit, self.settings.value(side) + self.step))
                        return
            if 816 <= y <= 856:
                if 24 <= x <= 225:
                    self.settings.reset()
                elif 245 <= x <= 446:
                    self.save_requested = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            self.set_from_x(self.dragging, x)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging is not None:
                self.set_from_x(self.dragging, x)
            self.dragging = None


class EyePredictor:
    """Load a frozen fit once; do not select/tune using webcam predictions."""

    def __init__(self, method="aeee2", seed=None, run_dir=ASSETS / "runs/phase4_v1",
                 bundle_path=ASSETS / "data/encoded/v1/bundle.json", compare_float=False):
        self.method = method
        if seed is None:
            # Selection uses frozen training/validation records only, never held-out results.
            candidates = [json.loads(path.read_text(encoding="utf-8"))
                          for path in sorted((Path(run_dir) / method).glob("seed_*/fit.json"))]
            if not candidates:
                raise ValueError(f"No frozen fits available for {method}")
            chosen = max(candidates, key=lambda fit: (fit["selected_validation_macro_f1"], -fit["seed"]))
            seed = chosen["seed"]
        self.seed = seed
        self.directory = Path(run_dir).resolve() / method / f"seed_{seed}"
        fit = json.loads((self.directory / "fit.json").read_text(encoding="utf-8"))
        config = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
        bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
        if fit["method"] != method or fit["seed"] != seed:
            raise ValueError("Frozen fit does not match requested method/seed")
        if sha256(Path(bundle_path)) != config["bundle_sha256"]:
            raise ValueError("Encoder bundle differs from the training run")
        if bundle["methods"][method] != METHODS[method].SPEC:
            raise ValueError("Encoder specification differs from the frozen training bundle")
        for filename in (f"encoders/{method}.py", "encoders/common.py", "encoders/__init__.py"):
            if sha256(CODE_ROOT / filename) != bundle["source_code_sha256"][filename]:
                raise ValueError(f"Encoder source differs from trained model: {filename}")
        if sha256(CODE_ROOT / "integer_model.py") != config["source_sha256"]["integer_model.py"]:
            raise ValueError("Integer arithmetic differs from the frozen reference")
        checkpoint = self.directory / "int8_parameters.npz"
        if sha256(checkpoint) != fit["int8_sha256"]:
            raise ValueError("INT8 checkpoint hash differs from the frozen fit")
        self.checkpoint_sha256 = fit["int8_sha256"]
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.params = {key: archive[key] for key in archive.files}
        self.identifier = f"{method} / seed {seed} / {METHODS[method].SPEC['version']}"
        self.validation_f1 = fit["selected_validation_macro_f1"]
        self.float_model = None
        if compare_float:
            import torch
            from research_model import TinyEye
            checkpoint = self.directory / "float32.pt"
            if sha256(checkpoint) != fit["float32_sha256"]:
                raise ValueError("FP32 checkpoint hash differs from frozen fit")
            if sha256(CODE_ROOT / "research_model.py") != config["source_sha256"]["research_model.py"]:
                raise ValueError("Float architecture differs from frozen run")
            torch.set_num_threads(2)
            self.float_model = TinyEye()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.float_model.load_state_dict(saved["state_dict"])
            self.float_model.eval()

    def predict_rois(self, rois):
        """Infer current valid eyes only. An invalid eye never gets an old result."""
        started = time.perf_counter_ns()
        predictions, mapped, sides = {}, [], []
        for side, roi in rois.items():
            if roi is None:
                continue
            view = roi.view()
            view.flags.writeable = False
            codes = METHODS[self.method].encode(view)
            pixels = codes if self.method == "grayscale8" else codes * np.uint8(85)
            predictions[side] = {"pixels": pixels, "roi": roi}
            mapped.append(pixels)
            sides.append(side)
        encoded_at = time.perf_counter_ns()
        if sides:
            batch = np.stack(mapped)[:, None, :, :]
            logits = infer(batch, self.params)
            for side, scores in zip(sides, logits):
                predictions[side].update(state=("OPEN", "CLOSED")[int(scores.argmax())],
                                         logits=[int(score) for score in scores])
        finished = time.perf_counter_ns()
        if sides and self.float_model is not None:
            import torch
            with torch.inference_mode():
                float_logits = self.float_model(torch.from_numpy(batch).float() / 255).numpy()
            for side, scores in zip(sides, float_logits):
                predictions[side].update(float_state=("OPEN", "CLOSED")[int(scores.argmax())],
                                         float_logits=[float(score) for score in scores])
        return predictions, {"encode_ms": (encoded_at - started) / 1e6,
                             "cnn_ms": (finished - encoded_at) / 1e6,
                             "float_ms": (time.perf_counter_ns() - finished) / 1e6}


def text(image, message, x, y, size=0.55, color=(220, 225, 235)):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def render_preview(frame, boxes, predictions, predictor, frame_id, timings, age_ms, fps,
                   threshold_ui=None, execution_label=None, bilateral_blinks=None):
    panel = np.full((PANEL_HEIGHT, 1120, 3), (24, 20, 18), np.uint8)
    text(panel, "AEEE / EYE STATE LAB", 24, 38, 0.9, (240, 225, 180))
    text(panel, execution_label or "LAPTOP INT8 REFERENCE | FPGA inference: not connected", 24, 70, 0.55)
    text(panel, predictor.identifier + f" | baseline val F1 {predictor.validation_f1:.3f}", 24, 100, 0.53, (140, 210, 250))
    annotated = frame.copy() if frame is not None else np.zeros((360, 640, 3), np.uint8)
    for side, (x0, y0, x1, y1) in boxes.items():
        state = predictions.get(side, {}).get("state", "UNAVAILABLE")
        color = COLORS[state]
        cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), color, 2)
        text(annotated, f"{side}: {state}", x0, max(20, y0 - 8), 0.6, color)
    height, width = annotated.shape[:2]
    scale = min(640 / width, 360 / height)
    preview = cv2.resize(annotated, (round(width * scale), round(height * scale)))
    ph, pw = preview.shape[:2]
    panel[126:126 + ph, 24:24 + pw] = preview
    text(panel, "Camera is unmirrored. Left/right refer to the subject.", 24, 512, 0.49)
    progress = f"Frame {frame_id} | {fps:.1f} FPS"
    if bilateral_blinks is not None:
        progress += f" | BOTH-eye blinks: {bilateral_blinks}"
    text(panel, progress, 24, 548, 0.6)
    text(panel, f"After-read age {age_ms:.1f} ms (not sensor latency)", 24, 580, 0.54)
    text(panel, f"Locate {timings.get('locate_ms', 0):.1f} ms | Encode {timings.get('encode_ms', 0):.2f} ms",
         24, 612, 0.54)
    text(panel, f"INT8 CNN {timings.get('cnn_ms', 0):.2f} ms for current valid eye(s)", 24, 644, 0.54)
    for index, side in enumerate(EYE_CORNERS):
        x, y = 700, 130 + index * 250
        result = predictions.get(side)
        state = result["state"] if result else "UNAVAILABLE"
        text(panel, f"SUBJECT-{side.upper()}", x, y + 16, 0.65)
        state_label = state
        if bilateral_blinks is not None:
            state_label += " (confirmed)"
        if result and "blink_count" in result and bilateral_blinks is None:
            state_label += f" | blinks {result['blink_count']}"
        text(panel, state_label, x, y + 51, 0.65 if result and "blink_count" in result else 0.85, COLORS[state])
        text(panel, "RAW 32x16          ENCODED INPUT", x, y + 78, 0.42)
        for offset, field in ((0, "roi"), (184, "pixels")):
            tile = np.zeros((80, 160), np.uint8) if result is None else cv2.resize(
                result[field], (160, 80), interpolation=cv2.INTER_NEAREST)
            panel[y + 90:y + 170, x + offset:x + offset + 160] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        scores = str(result["logits"]) if result else "--"
        text(panel, f"Raw logits [OPEN, CLOSED]: {scores}", x, y + 199, 0.41)
        if result and "float_state" in result:
            float_state = result["float_state"]
            agreement = "AGREE" if result.get("model_state", state) == float_state else "DIFFER"
            text(panel, f"FP32 raw: {float_state} | numerical {agreement}", x, y + 223,
                 0.44, COLORS[float_state])
        if result and "threshold" in result:
            text(panel, f"Raw {result['model_state']} | margin {result['margin']:+d} | T {result['threshold']:+d}",
                 x, y + 245, 0.39)
    text(panel, "1 Grayscale8 | 2 Grayscale2 | 3 Sobel2 | 4 AEEE | Q / ESC exit", 24, 680, 0.55)
    if threshold_ui is not None:
        controls = threshold_ui.settings
        for side, (start, end) in SLIDERS.items():
            value = controls.value(side)
            text(panel, f"SUBJECT-{side.upper()} threshold: {value:+,}", start - 60, 717, 0.61)
            cv2.line(panel, (start, 758), (end, 758), (130, 145, 160), 5)
            center = (start + end) // 2
            cv2.line(panel, (center, 747), (center, 769), (190, 200, 215), 1)
            position = start + round((min(threshold_ui.limit, max(-threshold_ui.limit, value)) / threshold_ui.limit + 1) / 2 * (end - start))
            cv2.circle(panel, (position, 758), 11, (120, 210, 250), -1)
            for x, label in ((start - 60, "-"), (end + 20, "+")):
                cv2.rectangle(panel, (x, 738), (x + 40, 777), (65, 80, 95), -1)
                text(panel, label, x + 12, 765, 0.7)
            text(panel, "easier CLOSED", start, 800, 0.43)
            text(panel, "easier OPEN", end - 94, 800, 0.43)
        for x, label in ((24, "RESET BOTH (R)"), (245, "SAVE (S)")):
            cv2.rectangle(panel, (x, 816), (x + 201, 856), (65, 80, 95), -1)
            text(panel, label, x + 13, 843, 0.59)
        text(panel, "Lower threshold = easier CLOSED. Zero restores the model's original rule.", 24, 885, 0.52)
        text(panel, controls.status, 24, 917, 0.49, (130, 190, 225))
    else:
        text(panel, "Experimental model output; scores are not calibrated confidence or ground truth.",
             24, 917, 0.49, (130, 165, 220))
    return panel


def host_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), default="aeee2")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=None,
                        help="Default: best frozen validation macro F1 within the selected method")
    parser.add_argument("--run-dir", type=Path, default=ASSETS / "runs/phase4_v1")
    parser.add_argument("--bundle", type=Path, default=ASSETS / "data/encoded/v1/bundle.json")
    parser.add_argument("--seconds", type=float, default=0, help="Stop after this duration; 0 runs until Q/ESC")
    parser.add_argument("--headless", action="store_true", help="Run capture/inference without preview windows")
    parser.add_argument("--compare-float", action="store_true", help="Show matching frozen FP32 prediction for diagnosis")
    parser.add_argument("--stop-file", type=Path, help="Exit cleanly if this file appears")
    parser.add_argument("--report", type=Path, help="Optional metadata-only JSON; never saves camera images")
    parser.add_argument("--events", type=Path, help="Optional new blink prediction CSV")
    parser.add_argument("--session-id", default=uuid4().hex)
    parser.add_argument("--threshold-settings", type=Path,
                        help="Default: manual_thresholds.json inside the selected run directory")
    parser.add_argument("--left-threshold", type=int, help="Override the initial subject-left threshold")
    parser.add_argument("--right-threshold", type=int, help="Override the initial subject-right threshold")
    parser.add_argument("--threshold-range", type=int, default=100000, help="Slider range is +/- this integer margin")
    args = parser.parse_args(argv)
    if args.seconds < 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be finite and nonnegative")
    if not 1 <= args.threshold_range <= 4294967295:
        parser.error("--threshold-range must be between 1 and 4294967295")
    pending, errors = Queue(maxsize=1), Queue(maxsize=1)
    stop, stopped = Event(), Event()
    worker, landmarker = None, None
    blink_detectors = {side: BlinkDetector() for side in EYE_CORNERS}
    event_log = None
    stats = {"frames": 0, "frames_with_face": 0, "eyes_inferred": 0,
             "prediction_counts": {"OPEN": 0, "CLOSED": 0},
             "raw_model_prediction_counts": {"OPEN": 0, "CLOSED": 0},
             "execution": "laptop NumPy INT8 reference; no FPGA inference",
             "camera": 0, "images_saved": False, "models_used": [],
             "float_compared_eyes": 0, "float_int8_disagreements": 0,
             "manual_threshold_history": []}
    exit_code = 0
    try:
        if args.events:
            event_log = BlinkEventLog(args.events, args.session_id)
        stats["session_id"] = args.session_id
        predictor = EyePredictor(args.method, args.seed, args.run_dir, args.bundle, args.compare_float)
        thresholds = ManualThresholds(args.threshold_settings or args.run_dir / "manual_thresholds.json")
        thresholds.activate(predictor.checkpoint_sha256, predictor.identifier)
        for side in ("left", "right"):
            override = getattr(args, f"{side}_threshold")
            if override is not None:
                thresholds.set(side, override)
        threshold_ui = ThresholdUI(thresholds, args.threshold_range)
        last_thresholds = None
        stats["models_used"].append(predictor.identifier)
        print(f"MODEL {predictor.identifier}\nMODE laptop INT8 reference; no dataset/image recording", flush=True)
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1120, PANEL_HEIGHT)
            cv2.setMouseCallback(WINDOW, threshold_ui.mouse)
        worker = Thread(target=capture_frames, args=(pending, errors, stop, stopped),
                        name="live-eye-capture", daemon=True)
        worker.start()
        started = last_capture = last_log = time.monotonic()
        last_timestamp_ms = -1
        while not args.seconds or time.monotonic() - started < args.seconds:
            if args.stop_file and args.stop_file.exists():
                break
            if stopped.is_set():
                raise RuntimeError(errors.get_nowait() if not errors.empty() else "Camera stopped")
            snapshot = thresholds.snapshot()
            if snapshot != last_thresholds:
                for detector in blink_detectors.values():
                    detector.invalidate()
                stats["manual_threshold_history"].append(dict(snapshot, processed_frames=stats["frames"]))
                last_thresholds = snapshot
                print("MANUAL THRESHOLDS " + json.dumps(snapshot), flush=True)
            frame_id, boxes, predictions, timings = 0, {}, {}, {}
            frame = None
            age_ms = 0.0
            try:
                frame_id, capture_ns, frame = pending.get(timeout=0.05)
            except Empty:
                if time.monotonic() - last_capture > 5:
                    raise RuntimeError("No camera frames for five seconds")
            else:
                last_capture = time.monotonic()
                age_ms = (time.monotonic_ns() - capture_ns) / 1e6
                # Never classify a frame that has sat in a driver/application stall.
                if age_ms > 1000:
                    raise RuntimeError("Camera frame is over one second old")
                locate_start = time.perf_counter_ns()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = max(capture_ns // 1_000_000, last_timestamp_ms + 1)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                last_timestamp_ms = timestamp_ms
                rois = {side: None for side in EYE_CORNERS}
                if result.face_landmarks:
                    stats["frames_with_face"] += 1
                    for side, corners in EYE_CORNERS.items():
                        roi, box = extract_eye_roi(frame, result.face_landmarks[0], corners)
                        rois[side] = roi
                        if box is not None:
                            boxes[side] = box
                locate_ms = (time.perf_counter_ns() - locate_start) / 1e6
                predictions, timings = predictor.predict_rois(rois)
                predictions = thresholds.apply(predictions)
                predictions = update_blinks(blink_detectors, predictions, capture_ns // 1_000_000)
                if event_log:
                    event_log.write({side: item["blink_event"] for side, item in predictions.items()},
                                    blink_detectors, capture_ns // 1_000_000)
                timings["locate_ms"] = locate_ms
                age_ms = (time.monotonic_ns() - capture_ns) / 1e6
                if stopped.is_set() or age_ms > 1000:
                    raise RuntimeError("Capture stopped or result became stale during processing")
                stats["frames"] += 1
                stats["eyes_inferred"] += len(predictions)
                for prediction in predictions.values():
                    stats["prediction_counts"][prediction["state"]] += 1
                    stats["raw_model_prediction_counts"][prediction["model_state"]] += 1
                    if "float_state" in prediction:
                        stats["float_compared_eyes"] += 1
                        stats["float_int8_disagreements"] += prediction["model_state"] != prediction["float_state"]
            elapsed = time.monotonic() - started
            if time.monotonic() - last_log >= 2:
                states = ", ".join(f"{side}={predictions.get(side, {}).get('state', 'UNAVAILABLE')}"
                                   + (f"/FP32:{predictions[side]['float_state']}" if side in predictions and "float_state" in predictions[side] else "")
                                   for side in EYE_CORNERS)
                print(f"LIVE frame={frame_id} {states} after_read_ms={age_ms:.1f}", flush=True)
                last_log = time.monotonic()
            if not args.headless:
                panel = render_preview(frame, boxes, predictions, predictor, frame_id, timings,
                                       age_ms, stats["frames"] / max(elapsed, 0.001), threshold_ui)
                cv2.imshow(WINDOW, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("r"), ord("R")):
                    thresholds.reset()
                if key in (ord("s"), ord("S")):
                    threshold_ui.save_requested = True
                if threshold_ui.save_requested:
                    threshold_ui.save_requested = False
                    try:
                        thresholds.save()
                    except (OSError, ValueError) as exc:
                        thresholds.status = "SAVE FAILED: " + str(exc)[:100]
                        print(thresholds.status, flush=True)
                if key in METHOD_KEYS and METHOD_KEYS[key] != predictor.method:
                    predictor = EyePredictor(METHOD_KEYS[key], args.seed, args.run_dir, args.bundle, args.compare_float)
                    thresholds.activate(predictor.checkpoint_sha256, predictor.identifier)
                    threshold_ui.dragging = None
                    stats["models_used"].append(predictor.identifier)
                    print(f"MODEL {predictor.identifier}", flush=True)
        stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        stats["error"] = str(exc)
        print(f"Live inference stopped: {exc}", flush=True)
        exit_code = 1
    finally:
        stats["blinks"] = {side: detector.count for side, detector in blink_detectors.items()}
        if event_log:
            stats["event_time_origin_capture_ms"] = event_log.origin_ms
            try:
                event_log.close()
            except OSError as exc:
                stats["event_log_error"] = str(exc)
                exit_code = 1
        stop.set()
        cv2.destroyAllWindows()
        if worker is not None:
            worker.join(timeout=2)
            if worker.is_alive():
                stats["shutdown_warning"] = "Native camera driver did not return within two seconds"
                exit_code = 1
        if landmarker is not None:
            landmarker.close()
        print("SUMMARY " + json.dumps(stats), flush=True)
        if args.report:
            try:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                print(f"Cannot write report: {exc}", flush=True)
                exit_code = 1
    return exit_code


# Fpga Live

FPGA_WINDOW="AEEE FPGA inference | SPARTAN-6 INT8 | Q / ESC to quit"


def fpga_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--run-dir", type=Path, required=True, help="Frozen TinyEye training run")
    parser.add_argument("--method",choices=tuple(METHODS),default="grayscale8")
    parser.add_argument("--seed",type=int,default=43)
    parser.add_argument("--seconds",type=float,default=0)
    parser.add_argument("--headless",action="store_true")
    parser.add_argument("--calibrate",action="store_true",help="Operator-labelled normal/wide-open/closed score check")
    parser.add_argument("--inspect-dir",type=Path,help="New local folder for up to 120 unlabelled eye-pair diagnostics")
    parser.add_argument("--left-threshold",type=int,help="Initial live override; not saved automatically")
    parser.add_argument("--right-threshold",type=int,help="Initial live override; not saved automatically")
    parser.add_argument("--stop-file",type=Path)
    parser.add_argument("--report",type=Path,default=ASSETS/"logs/live_report.json")
    parser.add_argument("--events",type=Path,help="Optional new blink prediction CSV")
    parser.add_argument("--session-id",default=uuid4().hex)
    args = parser.parse_args()
    if args.camera<0:parser.error("--camera must be nonnegative")
    if args.seconds<0 or not np.isfinite(args.seconds):parser.error("--seconds must be finite and nonnegative")
    if args.calibrate and args.headless:parser.error("--calibrate requires the interactive preview")
    if args.inspect_dir and args.headless:parser.error("--inspect-dir requires the interactive preview")
    pending,errors = Queue(maxsize=1),Queue(maxsize=1)
    stop,stopped = Event(),Event()
    client=landmarker=worker=None
    calibration_ui=None
    observation=None
    blink_detectors={side:BlinkDetector() for side in EYE_CORNERS}
    bilateral=BilateralBlinkDetector()
    stable_states=StableEyeStates()
    event_log=None
    stats={"execution":"physical Spartan-6 INT8 CNN over UART", "port":args.port,"baud":115200,"camera":args.camera,
           "frames":0,"hardware_pairs":0,"verified_pairs":0,"mismatches":0,"images_saved":bool(args.inspect_dir),
           "threshold_history":[],"models_used":[],"roundtrip_ms":[],"core_cycles":[]}
    stats["blink_scope"]="host bilateral, 2 consecutive joint observations; raw per-eye FPGA events are diagnostics"
    stats["live_state_confirmation_frames"]=2
    exit_code=0
    try:
        if args.events:event_log=BlinkEventLog(args.events,args.session_id)
        stats["session_id"]=args.session_id
        predictor=EyePredictor(args.method,args.seed,args.run_dir)
        thresholds=ManualThresholds(args.run_dir/"manual_thresholds.json")
        thresholds.activate(predictor.checkpoint_sha256,predictor.identifier)
        for side in ("left","right"):
            override=getattr(args,side+"_threshold")
            if override is not None:thresholds.set(side,override)
        controls=ThresholdUI(thresholds)
        if args.inspect_dir:observation=LiveObservation(args.inspect_dir)
        if args.calibrate:
            score_path=ASSETS/"logs"/("score_check_"+str(time.time_ns())+".json")
            calibration_ui=CalibrationUI(ScoreCalibration(predictor.checkpoint_sha256,score_path))
        client=FPGAClient(args.port)
        print("HARDWARE "+json.dumps(client.ping()),flush=True)
        hardware_status=client.load_model(predictor)
        if not hardware_status["timed_blinks"]:
            raise RuntimeError("This FPGA bitstream lacks capture-timed blink events; rebuild/program the current RTL first")
        client.reset_blinks()
        stats["models_used"].append(predictor.identifier)
        applied=None
        options=vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,num_faces=1,output_face_blendshapes=False,
            output_facial_transformation_matrixes=False)
        landmarker=vision.FaceLandmarker.create_from_options(options)
        if not args.headless:
            cv2.namedWindow(FPGA_WINDOW,cv2.WINDOW_NORMAL)
            cv2.resizeWindow(FPGA_WINDOW,1120,PANEL_HEIGHT)
            cv2.setMouseCallback(FPGA_WINDOW,controls.mouse)
        worker=Thread(target=capture_frames,args=(pending,errors,stop,stopped,args.camera),daemon=True)
        worker.start()
        started=last_capture=last_log=time.monotonic()
        timestamp=-1
        board_valid=False
        while not args.seconds or time.monotonic()-started<args.seconds:
            if args.stop_file and args.stop_file.exists(): break
            if stopped.is_set(): raise RuntimeError(errors.get_nowait() if not errors.empty() else "Camera stopped")
            snapshot=thresholds.snapshot()
            if snapshot!=applied:
                client.configure(thresholds.value("left"),thresholds.value("right"))
                for detector in blink_detectors.values():detector.invalidate()
                bilateral.invalidate();stable_states.invalidate()
                applied=snapshot
                stats["threshold_history"].append(dict(snapshot,hardware_pairs=stats["hardware_pairs"]))
            frame=None
            predictions,boxes,timings={}, {}, {}
            frame_id=0
            age_ms=0
            try:
                frame_id,captured_ns,frame=pending.get(timeout=0.05)
            except Empty:
                if time.monotonic()-last_capture>5: raise RuntimeError("Camera stalled")
            else:
                last_capture=time.monotonic()
                if (time.monotonic_ns()-captured_ns)/1e6>1000: raise RuntimeError("Stale camera frame")
                locating=time.perf_counter_ns()
                timestamp=max(timestamp+1,captured_ns//1_000_000)
                result=landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),timestamp)
                rois={}
                if result.face_landmarks:
                    for side,corners in EYE_CORNERS.items():
                        roi,box=extract_eye_roi(frame,result.face_landmarks[0],corners)
                        if roi is not None: rois[side]=roi;boxes[side]=box
                timings["locate_ms"]=(time.perf_counter_ns()-locating)/1e6
                if len(rois)==2:
                    encoding=time.perf_counter_ns()
                    codes=[METHODS[predictor.method].encode(rois[side]) for side in EYE_CORNERS]
                    pixels=np.stack(codes)
                    if predictor.method!="grayscale8":pixels=pixels*np.uint8(85)
                    timings["encode_ms"]=(time.perf_counter_ns()-encoding)/1e6
                    hardware=client.infer(predictor.method,*codes,frame_id,
                                          capture_ms=(captured_ns//1_000_000)&0xFFFFFFFF)
                    actual=np.array([hardware.left_logits,hardware.right_logits],dtype=np.int64)
                    expected=infer(pixels[:,None],predictor.params)
                    stats["hardware_pairs"]+=1
                    if not np.array_equal(actual,expected):
                        stats["mismatches"]+=1
                        raise RuntimeError("FPGA logits differ from the independent integer reference")
                    stats["verified_pairs"]+=1
                    for index,side in enumerate(EYE_CORNERS):
                        raw="CLOSED" if hardware.flags&(1<<index) else "OPEN"
                        if raw!=("OPEN","CLOSED")[int(actual[index].argmax())]:
                            stats["mismatches"]+=1
                            raise RuntimeError("FPGA raw decision differs from returned logits")
                        predictions[side]={"roi":rois[side],"pixels":pixels[index],"state":raw,
                                           "logits":[int(v) for v in actual[index]]}
                    predictions=thresholds.apply(predictions)
                    for index,side in enumerate(EYE_CORNERS):
                        physical="CLOSED" if hardware.flags&(1<<(index+2)) else "OPEN"
                        if predictions[side]["state"]!=physical:
                            stats["mismatches"]+=1
                            raise RuntimeError("FPGA threshold decision differs from the host rule")
                    timings["cnn_ms"]=(hardware.left_cycles+hardware.right_cycles)/50000
                    stats["roundtrip_ms"].append(hardware.roundtrip_ms)
                    stats["core_cycles"].append([hardware.left_cycles,hardware.right_cycles])
                    board_valid=True
                else:
                    if board_valid: client.invalidate(frame_id)
                    board_valid=False
                predictions=update_blinks(blink_detectors,predictions,captured_ns//1_000_000)
                if len(rois)==2:
                    for index,side in enumerate(EYE_CORNERS):
                        count=hardware.left_blinks if index==0 else hardware.right_blinks
                        if (predictions[side]["blink_count"]&0xFFFFFFFF)!=count or\
                           predictions[side]["blink_event"]!=bool(hardware.blink_flags&(1<<index)):
                            stats["mismatches"]+=1
                            raise RuntimeError("FPGA blink event/count differs from the capture-timed reference")
                joint_event=bilateral.update(predictions,captured_ns//1_000_000)
                if event_log:
                    event_log.write({"both":joint_event},{"both":bilateral},captured_ns//1_000_000)
                predictions=stable_states.update(predictions,captured_ns//1_000_000)
                age_ms=(time.monotonic_ns()-captured_ns)/1e6
                if stopped.is_set() or age_ms>1000: raise RuntimeError("Capture stopped or processed result stale")
                stats["frames"]+=1
            elapsed=time.monotonic()-started
            if calibration_ui is not None:
                calibration_ui.calibration.observe(predictions,time.monotonic())
                if calibration_ui.apply_requested:
                    calibration_ui.apply_requested=False
                    proposed=calibration_ui.calibration.proposals()
                    if proposed is not None and all(value is not None for value in proposed.values()):
                        for side,value in proposed.items():thresholds.set(side,value)
                        try:
                            thresholds.save()
                            calibration_ui.calibration.status="SAVED. New thresholds will configure the FPGA on the next frame."
                        except OSError as exc:
                            calibration_ui.calibration.status="SAVE FAILED: "+str(exc)[:100]
                calibration_ui.render()
            if time.monotonic()-last_log>=2:
                print(f"FPGA pairs={stats['hardware_pairs']} verified={stats['verified_pairs']} mismatches={stats['mismatches']} "
                      +" ".join(f"{side}={p['state']}" for side,p in predictions.items()),flush=True)
                last_log=time.monotonic()
            if not args.headless:
                label=f"SPARTAN-6 INT8 | {args.port} / 115200 | verified pairs {stats['verified_pairs']}"
                panel=render_preview(frame,boxes,predictions,predictor,frame_id,timings,age_ms,
                    stats["hardware_pairs"]/max(elapsed,0.001),controls,execution_label=label,
                    bilateral_blinks=bilateral.count)
                if observation is not None:
                    observation.add(frame_id,captured_ns if predictions else 0,predictions,panel,predictor.checkpoint_sha256)
                cv2.imshow(FPGA_WINDOW,panel)
                key=cv2.waitKey(1)&0xff
                if key in (ord("q"),ord("Q"),27) or cv2.getWindowProperty(FPGA_WINDOW,cv2.WND_PROP_VISIBLE)<1:break
                if key in (ord("r"),ord("R")):thresholds.reset()
                if key in (ord("s"),ord("S")):controls.save_requested=True
                if controls.save_requested:
                    controls.save_requested=False
                    try:thresholds.save()
                    except OSError as exc:thresholds.status="SAVE FAILED: "+str(exc)[:100]
                if key in METHOD_KEYS and METHOD_KEYS[key]!=predictor.method:
                    if calibration_ui is not None:
                        calibration_ui.calibration.status="Finish this check before switching models; restart without --calibrate to switch."
                        continue
                    predictor=EyePredictor(METHOD_KEYS[key],args.seed,args.run_dir)
                    client.load_model(predictor)
                    thresholds.activate(predictor.checkpoint_sha256,predictor.identifier)
                    controls.dragging=None
                    applied=None
                    stats["models_used"].append(predictor.identifier)
        stats["elapsed_seconds"]=round(time.monotonic()-started,3)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        stats["error"]=str(exc);print("FPGA LIVE ERROR: "+str(exc),flush=True);exit_code=1
    finally:
        stats["raw_per_eye_events"]={side:detector.count for side,detector in blink_detectors.items()}
        stats["blinks"]={"both":bilateral.count}
        if event_log:
            stats["event_time_origin_capture_ms"]=event_log.origin_ms
            try:event_log.close()
            except OSError as exc:
                stats["event_log_error"]=str(exc);exit_code=1
        if observation is not None:observation.close()
        stop.set();cv2.destroyAllWindows()
        if worker:worker.join(timeout=2)
        if landmarker:landmarker.close()
        if client:
            try:client.invalidate(0)
            except Exception:pass
            client.close()
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(stats,indent=2)+"\n",encoding="utf-8")
        print("FPGA LIVE SUMMARY "+json.dumps({k:v for k,v in stats.items() if k not in ("core_cycles","roundtrip_ms")}),flush=True)
    return exit_code


# Blink Live

BASELINE_WINDOW = "Blink baseline | pretrained MediaPipe | Q / Esc to quit"
BASELINE_COLORS = {"OPEN": (90, 225, 120), "CLOSED": (70, 160, 250), "UNAVAILABLE": (160, 160, 160)}


def eye_scores(result):
    if not result.face_landmarks:
        return {}
    if not result.face_blendshapes:
        raise RuntimeError("The model did not return face blendshapes")
    values = {item.category_name: item.score for item in result.face_blendshapes[0]}
    scores = {}
    for side in EYE_CORNERS:
        suffix = side.title()
        if "eyeBlink" + suffix not in values:
            raise RuntimeError("The model is missing eyeBlink" + suffix)
        scores[side] = {"blink": values["eyeBlink" + suffix], "wide": values.get("eyeWide" + suffix)}
    return scores


def baseline_text(image, message, x, y, size=0.55, color=(225, 225, 225)):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def baseline_preview(frame, boxes, scores, timing, fps, inference_ms, age_ms, fpga_roundtrip_ms=None):
    panel = np.full((620, 1080, 3), 24, np.uint8)
    baseline_text(panel, "EYE / BLINK BASELINE", 18, 35, 0.85)
    execution = "MediaPipe: laptop | Timing and counts: FPGA (checked)" if fpga_roundtrip_ms is not None else "MediaPipe + timing: laptop CPU"
    baseline_text(panel, execution, 18, 66)
    states = {side: ("CLOSED" if timing.closed_mask & (1 << index) else "OPEN")
              if timing.valid_mask & (1 << index) else "UNAVAILABLE" for index, side in enumerate(EYE_CORNERS)}
    annotated = frame.copy()
    for side, (x0, y0, x1, y1) in boxes.items():
        state = states[side]
        cv2.rectangle(annotated, (x0, y0), (x1-1, y1-1), BASELINE_COLORS[state], 2)
        baseline_text(annotated, side + ": " + state, x0, max(18, y0-8), 0.5, BASELINE_COLORS[state])
    height, width = annotated.shape[:2]
    scale = min(640/width, 400/height)
    image = cv2.resize(annotated, (round(width*scale), round(height*scale)))
    panel[96:96+image.shape[0], 18:18+image.shape[1]] = image
    for index, side in enumerate(EYE_CORNERS):
        state = states[side]
        x, y = 690, 115 + 220*index
        baseline_text(panel, "SUBJECT-" + side.upper(), x, y, 0.7)
        baseline_text(panel, state, x, y+38, 0.85, BASELINE_COLORS[state])
        score = scores.get(side, {}).get("blink")
        value = "--" if score is None else f"{score:.3f}"
        baseline_text(panel, "Blink coefficient: " + value, x, y+70)
        cv2.rectangle(panel, (x, y+87), (x+340, y+107), (60, 60, 60), -1)
        if score is not None and np.isfinite(score) and 0 <= score <= 1:
            cv2.rectangle(panel, (x, y+87), (x+round(score*340), y+107), BASELINE_COLORS[state], -1)
        for threshold in (0.4, 0.6):
            position = x+round(threshold*340)
            cv2.line(panel, (position, y+82), (position, y+112), (225, 225, 225), 1)
        current = (timing.left_current_ms, timing.right_current_ms)[index]
        last = (timing.last_left_ms, timing.last_right_ms)[index]
        baseline_text(panel, "Closed now: " + (f"{current} ms" if state != "UNAVAILABLE" else "--"), x, y+140)
        baseline_text(panel, f"Last bilateral cycle: {last} ms", x, y+170, 0.5)
    baseline_text(panel, f"Total: {timing.count} | Short: {timing.count-timing.long_count} | Long: {timing.long_count} | Last overlap: {timing.last_overlap_ms} ms", 18, 525)
    baseline_text(panel, f"Processed {fps:.1f} FPS | Model {inference_ms:.1f} ms | After-read age {age_ms:.1f} ms", 18, 550)
    rtt = "" if fpga_roundtrip_ms is None else f" | UART {fpga_roundtrip_ms:.1f} ms"
    baseline_text(panel, "Capture-timed; left/right are YOUR sides. Q / Esc: exit." + rtt, 18, 580, 0.5)
    baseline_text(panel, "Scores are expression coefficients, not probabilities. No images are saved.", 18, 610, 0.5)
    return panel


def baseline_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--port", help="Optional FPGA timing backend; requires the binocular timing bitstream")
    parser.add_argument("--seconds", type=float, default=0, help="0: run until Q/Esc")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--report", type=Path, default=ASSETS/"runs/blink_baseline/latest.json")
    parser.add_argument("--events", type=Path, help="Optional new prediction CSV; never ground-truth labels")
    parser.add_argument("--session-id", default=uuid4().hex)
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    if not np.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    timer = BinocularTimer()
    pending, errors = Queue(maxsize=1), Queue(maxsize=1)
    stop, stopped = Event(), Event()
    worker = landmarker = client = None
    event_log = None
    started = time.monotonic()
    stats = {"schema_version": 2, "execution": "MediaPipe on laptop; capture timing/counts on FPGA" if args.port else "MediaPipe and timing on laptop CPU",
             "mediapipe_version": mp.__version__, "opencv_version": cv2.__version__,
             "frames": 0, "frames_with_face": 0, "skipped_capture_frames": 0,
             "images_saved": False, "training_performed": False, "camera": args.camera,
             "blink_scope": "overlapping bilateral closures; no frame-count confirmation; long closures retained separately",
             "port": args.port, "verified_samples": 0,
             "decision": {"open_at_or_below": 0.4, "closed_at_or_above": 0.6,
                          "minimum_eye_and_overlap_ms": 30, "long_closure_ms": 600,
                          "maximum_onset_difference_ms": 150, "maximum_observation_gap_ms": 250},
             "model_time_sum_ms": 0.0, "model_time_max_ms": 0.0,
             "after_read_sum_ms": 0.0, "after_read_max_ms": 0.0,
             "accuracy_scope": "unlabelled engineering run; no accuracy measurement"}
    exit_code = 0
    try:
        if args.events:
            event_log = BlinkEventLog(args.events, args.session_id)
        stats["session_id"] = args.session_id
        stats["model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        if args.port:
            client = FPGAClient(args.port)
            status = client.ping()
            if not status.get("binocular_timing"):
                raise RuntimeError("Program the binocular timing FPGA image before using --port")
            client.reset_blinks()
            stats["hardware"] = status
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        if not args.headless:
            cv2.namedWindow(BASELINE_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(BASELINE_WINDOW, 1080, 620)
        worker = Thread(target=capture_frames, args=(pending, errors, stop, stopped, args.camera), daemon=True)
        worker.start()
        started = last_capture = last_log = time.monotonic()
        timestamp, last_frame_id, last_capture_ns = -1, 0, -1
        while not args.seconds or time.monotonic()-started < args.seconds:
            if args.stop_file and args.stop_file.exists():
                break
            if stopped.is_set():
                raise RuntimeError(errors.get_nowait() if not errors.empty() else "Camera stopped")
            try:
                frame_id, captured_ns, frame = pending.get(timeout=0.05)
            except Empty:
                if time.monotonic()-last_capture > 5:
                    raise RuntimeError("Webcam stalled for five seconds")
                continue
            last_capture = time.monotonic()
            # Some camera drivers drain buffered frames within one clock tick.
            # Repeated capture times cannot define another temporal observation.
            if captured_ns <= last_capture_ns:
                continue
            last_capture_ns = captured_ns
            if (time.monotonic_ns()-captured_ns)/1e6 > 1000:
                raise RuntimeError("Stale camera frame")
            capture_ms = captured_ns//1_000_000
            if capture_ms <= timestamp:
                continue
            timestamp = capture_ms
            before = time.perf_counter_ns()
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp)
            inference_ms = (time.perf_counter_ns()-before)/1e6
            scores, boxes = eye_scores(result), {}
            left, right = (scores.get(side, {}).get("blink") for side in EYE_CORNERS)
            timing = timer.update(left, right, capture_ms & 0xffffffff)
            fpga_rtt = None
            if client is not None:
                hardware_timing, fpga_rtt = client.eye_timing(left, right, capture_ms & 0xffffffff, frame_id)
                if hardware_timing != timing:
                    raise RuntimeError("FPGA closure timing/count differs from capture-timed reference")
                stats["verified_samples"] += 1
                timing = hardware_timing
            for side in EYE_CORNERS:
                if result.face_landmarks:
                    _, box = extract_eye_roi(frame, result.face_landmarks[0], EYE_CORNERS[side])
                    if box is not None:
                        boxes[side] = box
            if event_log:
                event_log.write({"both": timing.event}, {"both": timer}, capture_ms)
            age_ms = (time.monotonic_ns()-captured_ns)/1e6
            if stopped.is_set() or age_ms > 1000:
                raise RuntimeError("Camera stopped or completed result is stale")
            stats["frames"] += 1
            stats["frames_with_face"] += bool(scores)
            stats["skipped_capture_frames"] += max(0, frame_id-last_frame_id-1)
            last_frame_id = frame_id
            stats["model_time_sum_ms"] += inference_ms
            stats["model_time_max_ms"] = max(stats["model_time_max_ms"], inference_ms)
            stats["after_read_sum_ms"] += age_ms
            stats["after_read_max_ms"] = max(stats["after_read_max_ms"], age_ms)
            elapsed = time.monotonic()-started
            if time.monotonic()-last_log >= 2:
                print(f"TIMING total={timing.count} long={timing.long_count} last_ms={timing.last_duration_ms} "
                      f"L/R={timing.last_left_ms}/{timing.last_right_ms} verified={stats['verified_samples']}", flush=True)
                last_log = time.monotonic()
            if not args.headless:
                cv2.imshow(BASELINE_WINDOW, baseline_preview(frame, boxes, scores, timing, stats["frames"]/elapsed, inference_ms, age_ms, fpga_rtt))
                if cv2.waitKey(1)&0xff in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(BASELINE_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        stats["error"] = str(exc)
        print("BLINK ERROR: " + str(exc), flush=True)
        exit_code = 1
    finally:
        elapsed_seconds = time.monotonic()-started
        stop.set()
        if event_log:
            stats["event_time_origin_capture_ms"] = event_log.origin_ms
            try:
                event_log.close()
            except OSError as exc:
                stats["event_log_error"] = str(exc)
                exit_code = 1
        if not args.headless:
            cv2.destroyAllWindows()
        if worker:
            worker.join(timeout=2)
        if landmarker:
            landmarker.close()
        if client:
            try:client.invalidate(0)
            except Exception as exc:stats["hardware_shutdown_error"] = str(exc)
            finally:client.close()
        stats["elapsed_seconds"] = round(elapsed_seconds, 3)
        stats["closures"] = {"total": timer.count, "short": timer.count-timer.long_count, "long": timer.long_count}
        stats["last_duration_ms"] = timer.last_duration_ms
        stats["last_left_ms"], stats["last_right_ms"] = timer.last_left_ms, timer.last_right_ms
        stats["processed_fps"] = stats["frames"]/max(stats["elapsed_seconds"], 0.001)
        for prefix in ("model_time", "after_read"):
            stats[prefix+"_mean_ms"] = stats.pop(prefix+"_sum_ms")/max(stats["frames"], 1)
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_name(args.report.name+".tmp")
            temporary.write_text(json.dumps(stats, indent=2)+"\n", encoding="utf-8")
            temporary.replace(args.report)
        except OSError as exc:
            print("REPORT ERROR: " + str(exc), flush=True)
            exit_code = 1
        print("BLINK SUMMARY " + json.dumps(stats), flush=True)
    return exit_code


# Replay Blinks

def cnn_predictions(frame, landmarks, predictor, thresholds, client, frame_id, timestamp_ms):
    """Use the existing canonical crops; both eyes are required for paired replay."""
    rois = {}
    if landmarks:
        for side, corners in EYE_CORNERS.items():
            roi, _ = extract_eye_roi(frame, landmarks[0], corners)
            if roi is not None:
                rois[side] = roi
    if len(rois) != 2:
        if client is not None:
            client.invalidate(frame_id)
        return {}, None
    if client is None:
        predictions, _ = predictor.predict_rois(rois)
        for side, prediction in predictions.items():
            prediction["state"], prediction["margin"] = decide(prediction["logits"], thresholds[side])
        return predictions, None

    codes = [METHODS[predictor.method].encode(rois[side]) for side in EYE_CORNERS]
    hardware = client.infer(predictor.method, *codes, frame_id, capture_ms=timestamp_ms)
    pixels = np.stack(codes)
    if predictor.method != "grayscale8":
        pixels = pixels * np.uint8(85)
    expected = infer(pixels[:, None], predictor.params)
    actual = np.array([hardware.left_logits, hardware.right_logits], dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise RuntimeError("FPGA logits differ from the independent integer reference")
    predictions = {}
    for index, side in enumerate(EYE_CORNERS):
        raw = ("OPEN", "CLOSED")[int(actual[index].argmax())]
        state, margin = decide(actual[index], thresholds[side])
        if raw != ("CLOSED" if hardware.flags & (1 << index) else "OPEN"):
            raise RuntimeError("FPGA raw decision differs from its logits")
        if state != ("CLOSED" if hardware.flags & (1 << (index + 2)) else "OPEN"):
            raise RuntimeError("FPGA threshold decision differs from the host rule")
        predictions[side] = {"state": state, "margin": margin}
    return predictions, hardware


def check_hardware_events(hardware, predictions):
    for index, side in enumerate(EYE_CORNERS):
        count = hardware.left_blinks if index == 0 else hardware.right_blinks
        if (predictions[side]["blink_count"] & 0xFFFFFFFF) != count or\
                predictions[side]["blink_event"] != bool(hardware.blink_flags & (1 << index)):
            raise RuntimeError("FPGA blink events/counts differ from the media-timed reference")


def process_frame(frame, frame_id, timestamp_ms, landmarker, detectors, predictor, thresholds, client):
    """Locate, classify, update events and collect timings for one decoded frame."""
    started = time.perf_counter_ns()
    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp_ms)
    hardware = None
    scores = {}
    if predictor is None:
        scores = eye_scores(result)
        events = {side: detector.update(scores.get(side, {}).get("blink"), timestamp_ms)
                  for side, detector in detectors.items()}
        predictions = {side: {"state": detector.state} for side, detector in detectors.items()
                       if detector.state != "UNAVAILABLE"}
    else:
        predictions, hardware = cnn_predictions(frame, result.face_landmarks, predictor,
                                                thresholds, client, frame_id, timestamp_ms)
        predictions = update_blinks(detectors, predictions, timestamp_ms)
        events = {side: item["blink_event"] for side, item in predictions.items()}
        if hardware is not None:
            check_hardware_events(hardware, predictions)
    row = {"frame_id": frame_id, "timestamp_ms": timestamp_ms,
           "face_present": int(bool(result.face_landmarks)),
           "processing_ms": (time.perf_counter_ns() - started) / 1e6,
           "fpga_pair_core_ms": hardware.pair_cycles / 50000 if hardware else "",
           "uart_roundtrip_ms": hardware.roundtrip_ms if hardware else ""}
    for side, detector in detectors.items():
        row.update({side + "_state": detector.state,
                    side + "_blink_score": scores.get(side, {}).get("blink", ""),
                    side + "_logit_margin": predictions.get(side, {}).get("margin", ""),
                    side + "_event": int(events.get(side, False)), side + "_count": detector.count})
    return row, events, len(predictions) == 2, hardware is not None


def replay_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Existing local CFR video; never a webcam index")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--backend", choices=("pretrained", "int8", "fpga"), default="pretrained")
    parser.add_argument("--fps", type=float, help="Explicit CFR rate; otherwise use video metadata")
    parser.add_argument("--timebase", choices=("cfr", "pts"), default="cfr",
                        help="cfr preserves the existing API; pts probes actual video presentation timestamps")
    parser.add_argument("--ffprobe", default="ffprobe", help="Executable used only for --timebase pts")
    parser.add_argument("--bundle", type=Path, default=ASSETS / "data/encoded/v1/bundle.json")
    parser.add_argument("--session-id", help="Default: video SHA256 plus CFR rate, shared across backends")
    parser.add_argument("--method", choices=tuple(METHODS), default="aeee2")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--run-dir", type=Path, help="Frozen TinyEye training run for int8 or fpga")
    parser.add_argument("--left-threshold", type=int, default=0)
    parser.add_argument("--right-threshold", type=int, default=0)
    parser.add_argument("--port", help="UART port for --backend fpga")
    args = parser.parse_args(argv)
    if args.backend != "pretrained" and args.run_dir is None:
        parser.error("--run-dir is required for TinyEye inference")
    if args.backend == "fpga" and not (args.port and args.port.strip()):
        parser.error("--port is required for FPGA inference")
    args.video = args.video.resolve()
    args.output = args.output.resolve()
    if not args.video.is_file():
        parser.error("video must be an existing local file")
    if args.output.exists():
        parser.error("--output must be a new directory")
    if args.session_id is not None and not args.session_id.strip():
        parser.error("--session-id must be nonempty")
    if args.fps is not None and (not math.isfinite(args.fps) or not 0 < args.fps <= 1000):
        parser.error("--fps must be finite and in (0, 1000]")
    if args.timebase == "pts" and args.fps is not None:
        parser.error("--fps overrides CFR only; cannot replace presentation timestamps")
    thresholds = {side: getattr(args, side + "_threshold") for side in EYE_CORNERS}
    if any(not -(1 << 31) <= value < (1 << 31) for value in thresholds.values()):
        parser.error("thresholds must fit signed INT32")
    if args.backend == "pretrained" and any(thresholds.values()):
        parser.error("logit thresholds apply only to CNN backends")

    video_hash = sha256(args.video)
    session_id = args.session_id
    args.output.mkdir(parents=True, exist_ok=False)
    detectors = {side: BlinkDetector() for side in EYE_CORNERS}
    stats = {
        "schema_version": 1, "status": "incomplete", "backend": args.backend,
        "video": str(args.video), "video_sha256": video_hash, "session_id": session_id,
        "frames": 0, "frames_with_face": 0, "frames_with_two_decisions": 0,
        "verified_hardware_pairs": 0, "images_saved": False,
        "clock": "CFR: floor(zero-based frame index * 1000 / fps); first frame is 0 ms",
        "cfr_requirement": "Caller must supply a constant-frame-rate video; VFR is unsupported",
        "opencv_version": cv2.__version__, "mediapipe_version": mp.__version__,
        "decision": {"open_at_or_below": 0.4, "closed_at_or_above": 0.6,
                     "blink_duration_ms": [30, 600], "maximum_observation_gap_ms": 250},
        "accuracy_scope": "Predictions only; independent annotations are required for accuracy",
        "processing_time_scope": "After decode through inference/event verification; excludes CSV I/O",
        "processing_sum_ms": 0.0, "processing_max_ms": 0.0,
    }
    capture = landmarker = client = event_log = frame_log = None
    exit_code = 0
    started = time.perf_counter()
    try:
        source_files = ["app.py", "eye_state.py", "capture.py"]
        predictor = None
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError("Cannot open the input video")
        fps = args.fps if args.fps is not None else capture.get(cv2.CAP_PROP_FPS)
        if args.timebase == "cfr" and (not math.isfinite(fps) or not 0 < fps <= 1000):
            raise ValueError("Invalid CFR metadata; supply the known rate with --fps")
        media_times = None
        if args.timebase == "pts":
            from ocular_metrics import pts_milliseconds
            media_times = pts_milliseconds(args.video, args.ffprobe)
            stats.update(clock="ffprobe presentation timestamp (best_effort/pts/pkt_pts), relative integer ms",
                         cfr_requirement="Not required in PTS mode", pts_frames=len(media_times))
            if not math.isfinite(fps) or not 0 < fps <= 1000:
                fps = None
        session_id = args.session_id or (f"{video_hash}@pts" if media_times is not None else f"{video_hash}@{fps:.17g}fps")
        stats.update(session_id=session_id, fps=fps,
                     fps_source="explicit" if args.fps is not None else "video metadata")
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        stats["metadata_frame_count"] = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else None
        stats["landmarker_sha256"] = sha256(MODEL_PATH)
        if args.backend != "pretrained":
            predictor = EyePredictor(args.method, args.seed, args.run_dir, args.bundle)
            stats.update(method=predictor.method, seed=predictor.seed,
                         checkpoint_sha256=predictor.checkpoint_sha256,
                         run_dir=str(args.run_dir.resolve()), thresholds=thresholds,
                         crop_policy="Paired: invalidate both eyes if either canonical crop is missing")
            source_files += ["integer_model.py",
                             "encoders/__init__.py", "encoders/common.py", f"encoders/{args.method}.py"]
        if args.backend == "fpga":
            source_files.append("fpga_host.py")
            client = FPGAClient(args.port)
            status = client.ping()
            if not status["timed_blinks"]:
                raise RuntimeError("FPGA requires the new capture-timed blink firmware")
            stats["hardware"] = client.load_model(predictor)
            stats["hardware_provenance_scope"] = "UART signature/fingerprint; no bitstream attestation"
            client.reset_blinks()
            client.configure(thresholds["left"], thresholds["right"])
        stats["source_sha256"] = {name: sha256(CODE_ROOT / name) for name in source_files}
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=args.backend == "pretrained")
        landmarker = vision.FaceLandmarker.create_from_options(options)
        event_log = BlinkEventLog(args.output / "events.csv", session_id)
        frame_log = (args.output / "frames.csv").open("x", newline="", encoding="utf-8")
        fields = ["session_id", "frame_id", "timestamp_ms", "face_present", "processing_ms",
                  "fpga_pair_core_ms", "uart_roundtrip_ms", "encoder", "model_sha256", "left_valid", "right_valid"]
        for side in EYE_CORNERS:
            fields += [side + suffix for suffix in ("_state", "_blink_score", "_logit_margin", "_event", "_count")]
        writer = csv.DictWriter(frame_log, fieldnames=fields)
        writer.writeheader()
        while True:
            ok, frame = capture.read()
            if not ok:
                break  # OpenCV cannot reliably distinguish EOF from a decoder failure.
            if frame is None or frame.size == 0:
                raise RuntimeError("Decoder returned an empty frame")
            frame_id = stats["frames"]
            if media_times is not None and frame_id >= len(media_times):
                raise RuntimeError("More decoded frames than probed presentation timestamps")
            timestamp_ms = media_times[frame_id] if media_times is not None else int(frame_id * 1000 / fps)
            if frame_id and timestamp_ms <= stats["last_timestamp_ms"]:
                raise ValueError("Repeated/backward media timestamp")
            if timestamp_ms > 0xFFFFFFFF:
                raise ValueError("Replay exceeds the shared uint32 millisecond clock")
            row, events, paired, verified = process_frame(
                frame, frame_id, timestamp_ms, landmarker, detectors, predictor, thresholds, client)
            row["session_id"] = session_id
            row.update(encoder=predictor.method if predictor else "mediapipe_blendshapes",
                       model_sha256=stats.get("checkpoint_sha256", stats["landmarker_sha256"]),
                       left_valid=int(row["left_state"] != "UNAVAILABLE"),
                       right_valid=int(row["right_state"] != "UNAVAILABLE"))
            event_log.write(events, detectors, timestamp_ms)
            writer.writerow(row)
            stats["frames"] += 1
            stats["frames_with_face"] += row["face_present"]
            stats["frames_with_two_decisions"] += int(paired)
            stats["verified_hardware_pairs"] += int(verified)
            stats["processing_sum_ms"] += row["processing_ms"]
            stats["processing_max_ms"] = max(stats["processing_max_ms"], row["processing_ms"])
            stats["last_timestamp_ms"] = timestamp_ms
        if not stats["frames"]:
            raise RuntimeError("No video frames were decoded")
        if media_times is not None and len(media_times) != stats["frames"]:
            raise RuntimeError("Decoded frame count differs from presentation timestamps")
        expected_frames = stats["metadata_frame_count"]
        if expected_frames is not None and expected_frames != stats["frames"]:
            raise RuntimeError("Decoded frame count differs from metadata; replay may be truncated")
        stats["status"] = "complete"
        stats["decode_scope"] = "Stopped at decoder EOF; metadata count checked when available, not a file-integrity proof"
    except KeyboardInterrupt:
        stats["error"] = "Interrupted; output is partial"
        exit_code = 130
    except Exception as exc:
        stats["error"] = str(exc)
        exit_code = 1
    finally:
        for resource in (frame_log, event_log, landmarker, client):
            if resource is not None:
                try:
                    if resource is client:
                        client.invalidate(0)
                except Exception as exc:
                    stats.setdefault("cleanup_errors", []).append(str(exc))
                    stats["status"] = "incomplete"
                    exit_code = 1
                try:
                    resource.close()
                except Exception as exc:
                    stats.setdefault("cleanup_errors", []).append(str(exc))
                    stats["status"] = "incomplete"
                    exit_code = 1
        if capture is not None:
            capture.release()
        if stats["status"] == "complete":
            from ocular_metrics import export_metrics
            export_metrics(args.output / "frames.csv", args.output / "ocular_metrics.json")
        stats["elapsed_seconds"] = time.perf_counter() - started
        stats["blinks"] = {side: detector.count for side, detector in detectors.items()}
        stats["processing_mean_ms"] = stats["processing_sum_ms"] / stats["frames"] if stats["frames"] else None
        with (args.output / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"status": stats["status"], "frames": stats["frames"],
                          "output": str(args.output), "error": stats.get("error")}))
    return exit_code


# Predict Eye

def predict(image, method, model_dir, precision="int8", bundle_path=Path(ASSETS / "data/encoded/v1/bundle.json")):
    record = json.loads((model_dir / "fit.json").read_text(encoding="utf-8"))
    if record["method"] != method:
        raise ValueError("Model method does not match requested encoder")
    config = json.loads((model_dir.parents[1] / "config.json").read_text(encoding="utf-8"))
    if sha256(bundle_path) != config["bundle_sha256"]:
        raise ValueError("Bundle metadata differs from trained model")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle["methods"][method] != METHODS[method].SPEC:
        raise ValueError("Encoder specification differs from trained model")
    for filename in (f"encoders/{method}.py", "encoders/common.py", "encoders/__init__.py"):
        if sha256(Path(__file__).parent / filename) != bundle["source_code_sha256"][filename]:
            raise ValueError(f"Encoder source differs from trained model: {filename}")
    roi = read_roi(image)
    roi.flags.writeable = False
    codes = METHODS[method].encode(roi)
    pixels = codes if method == "grayscale8" else codes * 85
    pixels = pixels.reshape(1, 1, 16, 32)
    if precision == "int8":
        path = model_dir / "int8_parameters.npz"
        if sha256(path) != record["int8_sha256"]:
            raise ValueError("INT8 checkpoint hash differs from selected fit")
        with np.load(path, allow_pickle=False) as archive:
            params = {key: archive[key] for key in archive.files}
        logits = infer(pixels, params)[0]
    else:
        import torch
        from research_model import TinyEye
        path = model_dir / "float32.pt"
        if sha256(path) != record["float32_sha256"]:
            raise ValueError("FP32 checkpoint hash differs from selected fit")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["encoder_version"] != METHODS[method].SPEC["version"]:
            raise ValueError("Encoder version differs from trained model")
        model = TinyEye()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(pixels).float() / 255)[0].numpy()
    return {"method": method, "encoder_version": METHODS[method].SPEC["version"],
            "seed": record["seed"], "precision": precision,
            "prediction": ("OPEN", "CLOSED")[int(logits.argmax())],
            "logits_OPEN_CLOSED": logits.tolist(),
            "logits_format": "int32 at the final layer common scale" if precision == "int8" else "float32",
            "note": "Model prediction, not a verified ground-truth annotation or calibrated confidence."}


def predict_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="One method/seed directory")
    parser.add_argument("--precision", choices=("int8", "float32"), default="int8")
    parser.add_argument("--bundle", type=Path, default=Path(ASSETS / "data/encoded/v1/bundle.json"))
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.image, args.method, args.model_dir, args.precision, args.bundle), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Prediction failed: {exc}\n")


def hardware_cli(action):
    """Build the canonical FPGA project; program only a matching successful artifact."""
    import subprocess
    from rtl_evidence import build_canonical, build_variant, program_variant

    parser = argparse.ArgumentParser(description="Build/program the current RTL without restoring an archived model")
    if action == "build":
        parser.add_argument("--pe-count", type=int, choices=(1, 2), default=2)
        parser.add_argument("--network", choices=("pretrained", "tiny"), default="pretrained")
        parser.add_argument("--baud", type=int, help="Firmware UART rate; default 1000000 for pretrained, 115200 for tiny")
        parser.add_argument("--output", type=Path, help="New directory for an explicit research variant; defaults build FPGA/eye_top.xise")
    else:
        parser.add_argument("--build-dir", type=Path, help="Defaults to the canonical FPGA/ build")
    args = parser.parse_args()
    hardware = ROOT / "FPGA"
    try:
        if action == "build":
            if args.output is None:
                if args.pe_count != 2 or args.network != "pretrained" or args.baud not in (None, 1000000):
                    raise ValueError("Noncanonical configurations require --output for a research build")
                result = build_canonical()
            else:
                result = build_variant(args.output, args.pe_count, network=args.network, baud=args.baud)
        else:
            result = program_variant(args.build_dir or hardware)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{action.upper()} ERROR: {exc}\n")


def main():
    """Start the pinned pretrained classifier; archived research remains explicit."""
    commands = ("demo", "live", "compare", "experiment", "dataset", "session-report", "test-all", "model", "fpga", "host", "baseline", "replay", "predict", "capture", "collect", "build", "program",
                "prepare", "import", "encode", "train", "personal", "ablation", "events", "evidence", "rtl", "pretrained")
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", nargs="?", choices=commands, default="live")
        parser.print_help()
        print("Use COMMAND --help for that command's options.")
        return 0
    command = sys.argv.pop(1) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "live"
    if command == "demo":
        from demo import main as demo_main
        return demo_main()
    if command == "experiment":
        from experiment import experiment_cli
        return experiment_cli()
    if command == "dataset":
        from public_data import dataset_cli
        return dataset_cli()
    if command == "pretrained":
        from pretrained_eye import main as pretrained_main
        return pretrained_main()
    if command == "session-report":
        from experiment import report_cli
        return report_cli()
    if command == "test-all":
        from experiment import test_cli
        return test_cli()
    if command in ("live", "compare", "model", "host", "fpga"):
        from live_pipeline import live_cli, model_cli
        if command == "model":
            return model_cli()
        return live_cli(forced_backend={"host": "cpu", "fpga": "fpga", "compare": "compare"}.get(command))
    if command == "evidence":
        from evidence import main as evidence_main
        return evidence_main()
    if command == "rtl":
        from rtl_evidence import main as rtl_main
        return rtl_main()
    if command in ("build", "program"):
        return hardware_cli(command)
    live = {"baseline": baseline_cli,
            "replay": replay_cli, "predict": predict_cli}
    if command in live:
        return live[command]()
    if command in ("capture", "collect"):
        from capture import capture_cli, collect_cli
        return {"capture": capture_cli, "collect": collect_cli}[command]()
    if command in ("prepare", "import", "encode"):
        from data_tools import prepare_cli, import_cli, encode_cli
        return {"prepare": prepare_cli, "import": import_cli, "encode": encode_cli}[command]()
    if command in ("train", "personal", "ablation"):
        from research import train_cli, personal_cli, ablation_cli
        return {"train": train_cli, "personal": personal_cli, "ablation": ablation_cli}[command]()
    if command == "events":
        from eye_state import evaluate_events_cli
        return evaluate_events_cli()
    raise SystemExit("Unknown command: " + command + ". Use --help.")


if __name__ == "__main__":
    raise SystemExit(main())
