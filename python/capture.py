"""Camera ownership, unchanged canonical ROI extraction and manual collection."""

import argparse
import math
import sys
import time
import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from mediapipe.tasks.python import vision
from dataset import DatasetError, DatasetSession, validate_identifier
from collections import Counter
from datetime import datetime
from uuid import uuid4

CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent
ASSETS = ROOT / "assets"

def text(image, message, x, y, size=0.55, color=(220, 225, 235)):
    cv2.putText(image, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

# Main

MODEL_PATH = ASSETS / "models" / "face_landmarker.task"
EYE_CORNERS = {"left": (263, 362), "right": (33, 133)}  # Subject's sides.
ROI_SIZE = (32, 16)  # OpenCV width, height.
CROP_WIDTH_MULTIPLIER = 1.6


def capture_frames(pending: Queue, errors: Queue, stop: Event, stopped: Event, camera_index=0):
    """Own the selected integer camera index; publish the latest frame or a fatal error."""
    camera = None
    try:
        camera = cv2.VideoCapture()
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        if not camera.open(camera_index, backend):
            raise RuntimeError(f"Cannot open webcam (camera {camera_index}).")
        frame_id = 0
        while not stop.is_set():
            ok, frame = camera.read()
            capture_timestamp_ns = time.monotonic_ns()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError("Webcam read failed; capture has stopped.")
            if stop.is_set():
                break
            frame_id += 1
            packet = (frame_id, capture_timestamp_ns, frame)
            try:
                pending.put_nowait(packet)
            except Full:
                try:
                    pending.get_nowait()
                except Empty:
                    pass  # Main may have consumed it since put_nowait failed.
                # This is the only producer, so the queue cannot refill here.
                pending.put_nowait(packet)
    except Exception as exc:
        errors.put_nowait(f"Capture failure: {exc}")
    finally:
        # Notify main even if the native camera release subsequently stalls.
        stopped.set()
        if camera is not None:
            camera.release()


def extract_eye_roi(frame: np.ndarray, landmarks, corners: tuple[int, int]):
    """Return one canonical ROI and its exclusive-end pixel bounds, or None."""
    image_height, image_width = frame.shape[:2]
    if len(landmarks) <= max(corners):
        return None, None
    a, b = (landmarks[index] for index in corners)
    # Measure in original-image pixels, before any rounding or resizing.
    ax, ay = a.x * image_width, a.y * image_height
    bx, by = b.x * image_width, b.y * image_height
    if not all(math.isfinite(value) for value in (ax, ay, bx, by)):
        return None, None
    crop_width = CROP_WIDTH_MULTIPLIER * math.hypot(bx - ax, by - ay)
    crop_height = crop_width / 2.0
    if crop_height < 1.0:
        return None, None
    center_x, center_y = (ax + bx) / 2.0, (ay + by) / 2.0
    left, right = center_x - crop_width / 2.0, center_x + crop_width / 2.0
    top, bottom = center_y - crop_height / 2.0, center_y + crop_height / 2.0
    # Strict policy: reject ANY out-of-image extent; never shift or pad it.
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        return None, None
    # Quantize height to the nearest pixel and width to twice that height, so
    # the actual pixel crop also stays exactly 2:1. Round its origin to the
    # nearest pixel (ties upward). The ideal geometry above remains 1.6 * distance;
    # rasterization changes width by at most 1 px and center by at most 0.5 px.
    pixel_height = math.floor(crop_height + 0.5)
    pixel_width = 2 * pixel_height
    x0 = math.floor(center_x - pixel_width / 2.0 + 0.5)
    y0 = math.floor(center_y - pixel_height / 2.0 + 0.5)
    x1, y1 = x0 + pixel_width, y0 + pixel_height
    if not (0 <= x0 < x1 <= image_width and 0 <= y0 < y1 <= image_height):
        return None, None
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, ROI_SIZE, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(roi, dtype=np.uint8), (x0, y0, x1, y1)


def show_preview(frame, record, boxes, status: str):
    """Render display copies; never annotate or enlarge the research arrays."""
    display = frame.copy() if frame is not None else np.zeros((360, 640, 3), np.uint8)
    for side, (x0, y0, x1, y1) in boxes.items():
        cv2.rectangle(display, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
        cv2.putText(display, side, (x0, max(15, y0 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    for side in EYE_CORNERS:
        panel = np.zeros((160, 256), np.uint8)
        roi = record[f"{side}_eye_roi"] if record is not None else None
        if roi is not None:
            panel[32:] = cv2.resize(roi, (256, 128), interpolation=cv2.INTER_NEAREST)
        label = f"Subject-{side}" + (": unavailable" if roi is None else "")
        cv2.putText(panel, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.imshow(f"Subject-{side} eye", panel)
    cv2.putText(display, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 1)
    if record is not None:
        # Measured AFTER processing, from the original capture timestamp.
        # This is not sensor exposure latency or camera-driver buffering time.
        age_ms = (time.monotonic_ns() - record["capture_timestamp_ns"]) / 1_000_000
        label = f"Frame {record['frame_id']} | Age since read: {age_ms:.1f} ms"
        cv2.putText(display, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1)
    cv2.imshow("Phase 1 camera | Q / ESC to quit", display)


def capture_cli() -> int:
    """Own synchronous inference and windows; coordinate capture shutdown."""
    parser = argparse.ArgumentParser(description="Canonical eye preview and optional manual dataset collection.")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--subject-id")
    parser.add_argument("--session-id")
    parser.add_argument("--condition", help="Optional session condition, at most 80 printable characters.")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.collect:
        if not args.subject_id or not args.session_id:
            parser.error("--collect requires --subject-id and --session-id.")
        if sys.platform != "win32":
            parser.error("Live collection requires Windows key-release detection; offline preparation is portable.")
        try:
            validate_identifier(args.subject_id)
            validate_identifier(args.session_id)
        except DatasetError as exc:
            parser.error(str(exc))
    elif any(value is not None for value in (args.subject_id, args.session_id, args.condition, args.data_dir)):
        parser.error("Dataset options require --collect.")
    pending, errors = Queue(maxsize=1), Queue(maxsize=1)
    stop, stopped = Event(), Event()
    worker = None
    landmarker = None
    collection = None
    exit_code = 0
    try:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Missing FaceLandmarker model: {MODEL_PATH}")
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        if args.collect:
            import ctypes

            key_state = ctypes.WinDLL("user32").GetAsyncKeyState
            key_state.argtypes = [ctypes.c_int]
            key_state.restype = ctypes.c_short
            collection = DatasetSession(args.data_dir or MODEL_PATH.parent.parent / "data",
                                        args.subject_id, args.session_id, args.condition or "")
            label_keys = {
                ord("1"): ("OPEN", "OPEN"), ord("2"): ("CLOSED", "CLOSED"),
                ord("3"): ("OPEN", "CLOSED"), ord("4"): ("CLOSED", "OPEN"),
            }
            held_keys = set()
            feedback = "Hold the intended eye pose, then press 1 / 2 / 3 / 4."
        # A native webcam call cannot be forcibly cancelled by Python. A bounded
        # join plus daemon fallback prevents such a call holding the process open.
        worker = Thread(target=capture_frames, args=(pending, errors, stop, stopped),
                        name="webcam-capture", daemon=True)
        worker.start()
        last_timestamp_ms = -1
        last_capture_ns = time.monotonic_ns()
        while True:
            if stopped.is_set():
                try:
                    message = errors.get_nowait()
                except Empty:
                    message = "Capture worker stopped."
                raise RuntimeError(message)
            # Start each iteration unavailable; never reuse a previous result.
            frame, record, boxes = None, None, {}
            status = "Waiting for a new camera frame"
            try:
                frame_id, capture_timestamp_ns, frame = pending.get(timeout=0.05)
            except Empty:
                if time.monotonic_ns() - last_capture_ns > 5_000_000_000:
                    raise RuntimeError("No camera frames for 5 seconds; capture stalled.")
            else:
                last_capture_ns = capture_timestamp_ns
                if time.monotonic_ns() - capture_timestamp_ns > 5_000_000_000:
                    raise RuntimeError("Pending camera frame is over 5 seconds old; capture stalled.")
                record = {
                    "frame_id": frame_id,
                    "capture_timestamp_ns": capture_timestamp_ns,
                    "left_eye_roi": None,
                    "right_eye_roi": None,
                }
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(capture_timestamp_ns // 1_000_000,
                                   last_timestamp_ms + 1)
                result = landmarker.detect_for_video(image, timestamp_ms)
                last_timestamp_ms = timestamp_ms  # API ordering only.
                status = "No face detected"
                if result.face_landmarks:
                    for side, corners in EYE_CORNERS.items():
                        roi, box = extract_eye_roi(frame, result.face_landmarks[0], corners)
                        record[f"{side}_eye_roi"] = roi
                        if box is not None:
                            boxes[side] = box
                    status = "Eye crops available" if len(boxes) == 2 else "Eye ROI unavailable"
            if stopped.is_set():
                continue  # Discard a result completed after capture failed.
            show_preview(frame, record, boxes, status)
            if collection is not None:
                panel = np.zeros((140, 820, 3), np.uint8)
                lines = [f"{args.subject_id} / {args.session_id} | Saved pairs: {len(collection.saved_frames)}",
                         "Subject L/R: 1 OPEN/OPEN | 2 CLOSED/CLOSED | 3 OPEN/CLOSED | 4 CLOSED/OPEN",
                         "One capture per press. Release the key and pause briefly before pressing again.",
                         feedback]
                for index, text in enumerate(lines):
                    cv2.putText(panel, text, (8, 24 + 30 * index), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 255), 1)
                cv2.imshow("Manual collection", panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if collection is not None:
                down_keys = {
                    code for code in label_keys
                    if key_state(code) & 0x8000 or key_state(0x60 + code - ord("0")) & 0x8000
                }  # Top-row digits and NumLock-on numeric keypad keys.
                # Re-arm only after release AND the OpenCV key queue is idle.
                # Idle gaps between OS repeat events do not re-arm a held key;
                # queued repeat events after release are drained without saving.
                if key == 255:
                    held_keys.intersection_update(down_keys)
                if key in label_keys and key not in held_keys:
                    held_keys.add(key)
                    try:
                        if stopped.is_set():
                            raise DatasetError("Capture stopped; sample not saved.")
                        sample_id = collection.save_pair(record, *label_keys[key])
                        feedback = f"SAVED {sample_id} | L/R: {' / '.join(label_keys[key])}"
                    except (DatasetError, OSError, cv2.error) as exc:
                        feedback = f"REJECTED: {exc}"
                    print(feedback)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Phase 1 stopped: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        stop.set()
        try:
            # Remove the previews immediately when capture stops or we exit.
            cv2.destroyAllWindows()
        finally:
            try:
                if worker is not None and worker.ident is not None:
                    worker.join(timeout=2.0)
                    if worker.is_alive():
                        print("Camera driver did not return within 2 seconds; exiting. "
                              "Worker-owned camera release could not be confirmed.", file=sys.stderr)
                        exit_code = 1
            finally:
                try:
                    if landmarker is not None:
                        landmarker.close()
                finally:
                    if collection is not None:
                        collection.close()
    return exit_code


# Collect Personal

COLLECTION_WINDOW = "AEEE guided collection | CLICK buttons | 1 OPEN / 2 CLOSED / 3 ACCEPT"
PHASES = (("train", 32), ("val", 16), ("test", 16))
BUTTONS = ((24, "1 OPEN", ord("1")), (240, "2 CLOSED", ord("2")),
           (456, "3 ACCEPT", ord("3")), (672, "4 REJECT", ord("4")),
           (888, "5 NEXT", ord("5")))


def button_at(x, y):
    if 685 <= y <= 725:
        for left, label, key in BUTTONS:
            if left <= x <= left + 196:
                return key
    return None


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def beep():
    def sound():
        try:
            import winsound
            winsound.Beep(880, 250)
        except (ImportError, RuntimeError):
            pass
    Thread(target=sound, daemon=True).start()


def collect_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--subject-id", default="P001")
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    root = args.data_dir.resolve()
    # A new directory is mandatory: do not append to or relabel an old holdout.
    root.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": 1, "purpose": "single-user personal adaptation",
             "created_local": datetime.now().isoformat(), "subject_id": args.subject_id,
             "roi": "unchanged main.extract_eye_roi: uint8 16x32",
             "split_policy": "separate acquisition sessions, both eyes/pair kept together",
             "sample_interval_seconds": 0.5, "settle_seconds": 1.5,
             "labels": "operator-specified and reviewed; never model-generated",
             "generalization": "same person and nearby acquisition blocks; not unseen-subject evidence",
             "complete": False, "accepted": [], "rejected_bursts": [], "sessions": {}}
    write_json(root / "collection.json", state)
    pending, errors = Queue(maxsize=1), Queue(maxsize=1)
    stop, stopped = Event(), Event()
    worker, landmarker, session = None, None, None
    phase_index = 0
    burst = None
    review = None
    phase_done_at = None
    next_save = 0
    last_timestamp_ms = -1
    message = "Face the camera. O = both eyes OPEN. C = close BOTH eyes after pressing C."
    held = set()
    clicks = []
    try:
        import ctypes
        key_state = ctypes.WinDLL("user32").GetAsyncKeyState
        key_state.argtypes = [ctypes.c_int]
        key_state.restype = ctypes.c_short
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False)
        landmarker = vision.FaceLandmarker.create_from_options(options)
        worker = Thread(target=capture_frames, args=(pending, errors, stop, stopped), daemon=True)
        worker.start()
        cv2.namedWindow(COLLECTION_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(COLLECTION_WINDOW, 1120, 760)
        def clicked(event, x, y, flags, parameter):
            if event == cv2.EVENT_LBUTTONUP:
                action = button_at(x, y)
                if action is not None:
                    clicks.append(action)
        cv2.setMouseCallback(COLLECTION_WINDOW, clicked)
        print("COLLECTION READY: click OPEN or CLOSED; after the beep click ACCEPT or REJECT.", flush=True)
        last_capture = time.monotonic()
        while True:
            if args.stop_file and args.stop_file.exists():
                break
            if stopped.is_set():
                raise RuntimeError(errors.get_nowait() if not errors.empty() else "Camera stopped")
            now = time.monotonic()
            phase, target = PHASES[phase_index]
            if session is None and phase_done_at is None:
                session_id = f"{phase}_{uuid4().hex[:8]}"
                state["sessions"][phase] = session_id
                session = DatasetSession(root, args.subject_id, session_id, f"personal_{phase}_operator_labels")
                write_json(root / "collection.json", state)
            counts = Counter(row["label"] for row in state["accepted"] if row["split"] == phase)
            record, frame, boxes = None, None, {}
            try:
                frame_id, capture_ns, frame = pending.get(timeout=0.05)
            except Empty:
                if now - last_capture > 5:
                    raise RuntimeError("No camera frames for five seconds")
            else:
                last_capture = now
                if (time.monotonic_ns() - capture_ns) / 1e6 > 1000:
                    raise RuntimeError("Camera frame is stale")
                timestamp_ms = max(capture_ns // 1_000_000, last_timestamp_ms + 1)
                found = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), timestamp_ms)
                last_timestamp_ms = timestamp_ms
                record = {"frame_id": frame_id, "capture_timestamp_ns": capture_ns,
                          "left_eye_roi": None, "right_eye_roi": None}
                if found.face_landmarks:
                    for side, corners in EYE_CORNERS.items():
                        roi, box = extract_eye_roi(frame, found.face_landmarks[0], corners)
                        record[f"{side}_eye_roi"] = roi
                        if box is not None:
                            boxes[side] = box
            if burst is not None:
                if now >= next_save and record is not None and len(boxes) == 2 and not stopped.is_set():
                    try:
                        sample_id = session.save_pair(record, burst["label"], burst["label"])
                        burst["samples"].append({"sample_id": sample_id,
                            "rois": [record["left_eye_roi"].copy(), record["right_eye_roi"].copy()]})
                        print(f"SAVED {phase} {burst['label']} pair={len(burst['samples'])}/{burst['needed']} (awaiting review)", flush=True)
                        next_save = now + 0.5
                        message = f"Hold BOTH eyes {burst['label']}. Captured {len(burst['samples'])}/8. Wait for beep."
                    except DatasetError as exc:
                        message = str(exc)
                if len(burst["samples"]) >= burst["needed"] or now - burst["started"] > 15:
                    review, burst = burst, None
                    beep()
                    message = "Open your eyes and REVIEW the crops below. Y = correct labels; N = reject this burst."
            if burst is None and review is None and all(counts[label] >= target for label in ("OPEN", "CLOSED")):
                if phase_done_at is None:
                    session.close()
                    session = None
                    phase_done_at = now
                    beep()
                if phase_index == len(PHASES) - 1:
                    state["complete"] = True
                    write_json(root / "collection.json", state)
                    print("COLLECTION COMPLETE " + str(root), flush=True)
                    break
                message = "Session complete. Change posture/distance slightly. Wait 10 sec, then SPACE for next session."
            panel = np.full((760, 1120, 3), (24, 20, 18), np.uint8)
            text(panel, "AEEE / GUIDED PERSONAL DATA", 24, 38, 0.85)
            text(panel, f"Session {phase_index + 1}/3: {phase.upper()} | accepted pairs OPEN {counts['OPEN']}/{target}, CLOSED {counts['CLOSED']}/{target}",
                 24, 75, 0.62, (120, 210, 250))
            text(panel, "CLICK OPEN / CLOSED below | 1.5 sec to pose, then hold until beep", 24, 110, 0.56)
            if frame is not None:
                annotated = frame.copy()
                for side, (x0, y0, x1, y1) in boxes.items():
                    cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (100, 240, 140), 2)
                height, width = annotated.shape[:2]
                scale = min(640 / width, 300 / height)
                image = cv2.resize(annotated, (round(width * scale), round(height * scale)))
                ph, pw = image.shape[:2]
                panel[130:130 + ph, 24:24 + pw] = image
            text(panel, "Human labels, not model predictions", 690, 175, 0.52)
            text(panel, "Small eye crops are saved locally.", 690, 210, 0.5)
            text(panel, "Full camera frames are not saved.", 690, 244, 0.5)
            text(panel, "No training during collection.", 690, 278, 0.5)
            text(panel, "Q / ESC: quit (partial data retained)", 690, 312, 0.47)
            text(panel, f"Eye crops ready: {len(boxes)}/2", 690, 345, 0.55,
                 (100, 230, 130) if len(boxes) == 2 else (80, 150, 250))
            if burst:
                remaining_settle = max(0, next_save - now) if not burst["samples"] else 0
                text(panel, f"{burst['label']}  {len(burst['samples'])}/{burst['needed']}", 690, 370,
                     0.95, (100, 200, 250))
                if remaining_settle:
                    text(panel, "Get into the labelled pose now", 690, 405, 0.48)
            if phase_done_at is not None and phase_index < len(PHASES) - 1:
                text(panel, f"Next session available in {max(0, 10 - int(now - phase_done_at))} sec", 690, 400, 0.5)
            # Split the feedback text across two readable lines.
            words, line, lines = message.split(), "", []
            for word in words:
                if len(line) + len(word) > 100:
                    lines.append(line)
                    line = ""
                line += word + " "
            lines.append(line)
            for index, line in enumerate(lines[:2]):
                text(panel, line, 24, 465 + 28 * index, 0.54, (120, 220, 250))
            if review:
                text(panel, f"Both eyes {review['label']}? Check crops, then CLICK ACCEPT or REJECT.", 24, 535, 0.59)
                for index, saved in enumerate(review["samples"]):
                    x = 24 + index * 134
                    for row, roi in enumerate(saved["rois"]):
                        tile = cv2.resize(roi, (128, 64), interpolation=cv2.INTER_NEAREST)
                        panel[550 + row * 72:614 + row * 72, x:x + 128] = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            else:
                text(panel, "Keep your face inside the picture and look toward the webcam.", 24, 550, 0.56)
                text(panel, "For CLOSED: click CLOSED, close both eyes, hold until beep, then review.", 24, 585, 0.55)
                text(panel, "Click again after accepting each burst. Each session needs both labels.", 24, 620, 0.55)
            for left, label, action in BUTTONS:
                cv2.rectangle(panel, (left, 685), (left + 196, 725), (80, 95, 110), -1)
                text(panel, label, left + 15, 712, 0.67)
            text(panel, "This is personal adaptation. Existing public/test/holdout data remain unchanged.", 24, 733, 0.5)
            cv2.imshow(COLLECTION_WINDOW, panel)
            key = cv2.waitKey(1) & 0xFF
            clicked_key = bool(clicks)
            if clicked_key:
                key = clicks.pop(0)
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(COLLECTION_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 255:
                held = {code for code in held if key_state(ord(chr(code).upper())) & 0x8000}
            if key in held and not clicked_key:
                continue
            if key != 255 and not clicked_key:
                held.add(key)
            if review and key in (ord("3"), ord("4"), ord("y"), ord("Y"), ord("n"), ord("N")):
                accepted = key in (ord("3"), ord("y"), ord("Y"))
                if accepted:
                    for saved in review["samples"]:
                        state["accepted"].append({"sample_id": saved["sample_id"], "split": phase,
                            "label": review["label"], "burst_id": review["id"], "reviewed_by_operator": True})
                else:
                    state["rejected_bursts"].append({"burst_id": review["id"],
                        "sample_ids": [saved["sample_id"] for saved in review["samples"]]})
                write_json(root / "collection.json", state)
                print(f"REVIEW {phase} {review['label']} accepted={accepted} pairs={len(review['samples'])}", flush=True)
                review = None
                message = "Next burst: O for both OPEN, C for both CLOSED."
            elif phase_done_at is not None and key in (ord(" "), ord("5")) and now - phase_done_at >= 10:
                phase_index += 1
                phase_done_at = None
                message = "New acquisition session. O for OPEN, C for CLOSED."
            elif burst is None and review is None and phase_done_at is None and key in (ord("1"), ord("2"), ord("o"), ord("O"), ord("c"), ord("C")):
                label = "OPEN" if key in (ord("1"), ord("o"), ord("O")) else "CLOSED"
                if counts[label] >= target:
                    message = f"Enough {label} pairs for this session. Capture the other label."
                else:
                    burst = {"id": uuid4().hex, "label": label, "started": now,
                             "samples": [], "needed": min(8, target - counts[label])}
                    next_save = now + 1.5
                    print(f"BURST START {phase} {label} requested_pairs={burst['needed']}", flush=True)
                    message = f"Get ready: BOTH eyes {label}. Hold until beep, then review with Y/N."
    except Exception as exc:
        state["error"] = str(exc)
        print(f"COLLECTION ERROR {exc}", flush=True)
    finally:
        stop.set()
        cv2.destroyAllWindows()
        if worker:
            worker.join(timeout=2)
        if landmarker:
            landmarker.close()
        if session:
            session.close()
        write_json(root / "collection.json", state)
        print(f"COLLECTION complete={state['complete']} reviewed_pairs={len(state['accepted'])}", flush=True)
    return 0 if state["complete"] else 1
