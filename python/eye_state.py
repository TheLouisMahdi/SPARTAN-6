"""Eye-state thresholds, blink events, diagnostics and independent event evaluation."""

import math
import csv
import json
import time
import cv2
import numpy as np
import argparse
import hashlib
from pathlib import Path
from capture import ROOT, CODE_ROOT, ASSETS

# Blink

class BlinkDetector:
    def __init__(self, open_threshold=0.4, close_threshold=0.6,
                 minimum_ms=30, maximum_ms=600, maximum_gap_ms=250):
        if not 0 <= open_threshold < close_threshold <= 1:
            raise ValueError("Require 0 <= open threshold < close threshold <= 1")
        if not all(math.isfinite(v) for v in (minimum_ms, maximum_ms, maximum_gap_ms)):
            raise ValueError("Timing limits must be finite")
        if not 0 <= minimum_ms <= maximum_ms or maximum_gap_ms <= 0:
            raise ValueError("Invalid blink duration or observation gap")
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.minimum_ms = minimum_ms
        self.maximum_ms = maximum_ms
        self.maximum_gap_ms = maximum_gap_ms
        self.state = "UNAVAILABLE"
        self.count = 0
        self.last_blink_ms = None
        self._last_ms = None
        self._closed_at = None

    def invalidate(self):
        """Discard temporal continuity without erasing already completed events."""
        self.state, self._closed_at = "UNAVAILABLE", None

    def update(self, score, timestamp_ms):
        """Return True only for an observed OPEN -> CLOSED -> OPEN cycle."""
        if not math.isfinite(timestamp_ms):
            raise ValueError("Timestamp must be finite")
        if self._last_ms is not None:
            if timestamp_ms <= self._last_ms:
                raise ValueError("Observations must have increasing timestamps")
            if timestamp_ms - self._last_ms > self.maximum_gap_ms:
                self.invalidate()
        self._last_ms = timestamp_ms
        if score is None or not math.isfinite(score) or not 0 <= score <= 1:
            self.invalidate()
            return False
        if score >= self.close_threshold and self.state != "CLOSED":
            self._closed_at = timestamp_ms if self.state == "OPEN" else None
            self.state = "CLOSED"
        elif score <= self.open_threshold and self.state != "OPEN":
            duration = None if self._closed_at is None else timestamp_ms - self._closed_at
            self.state, self._closed_at = "OPEN", None
            if duration is not None and self.minimum_ms <= duration <= self.maximum_ms:
                self.count += 1
                self.last_blink_ms = duration
                return True
        return False


def update_blinks(detectors, predictions, timestamp_ms):
    """Adapt explicit CNN states, not logits/probabilities, to the same event rule."""
    updated = {}
    for side, detector in detectors.items():
        prediction = predictions.get(side)
        state = prediction["state"] if prediction else "UNAVAILABLE"
        score = {"OPEN": 0.0, "CLOSED": 1.0}.get(state)
        event = detector.update(score, timestamp_ms)
        if prediction is not None:
            updated[side] = dict(prediction, blink_count=detector.count, blink_event=event,
                                 last_blink_ms=detector.last_blink_ms)
    return updated


class StableEyeStates:
    """Confirm live per-eye labels after consecutive observations; preserve raw labels separately."""
    def __init__(self, frames=2, maximum_gap_ms=250):
        if frames < 1 or maximum_gap_ms <= 0:
            raise ValueError("Require positive frame count and observation gap")
        self.frames, self.maximum_gap_ms = frames, maximum_gap_ms
        self._last_ms = None
        self.invalidate()

    def invalidate(self):
        self.states = {side: "UNAVAILABLE" for side in ("left", "right")}
        self._pending = dict(self.states)
        self._seen = {side: 0 for side in self.states}

    def update(self, predictions, timestamp_ms):
        """Prediction dictionaries + increasing capture ms -> confirmed copies; no logit changes."""
        if not math.isfinite(timestamp_ms) or (self._last_ms is not None and timestamp_ms <= self._last_ms):
            raise ValueError("Capture timestamps must be finite and increasing")
        if self._last_ms is not None and timestamp_ms - self._last_ms > self.maximum_gap_ms:
            self.invalidate()
        self._last_ms = timestamp_ms
        result = {}
        for side in self.states:
            prediction = predictions.get(side)
            state = prediction.get("state", "UNAVAILABLE") if prediction else "UNAVAILABLE"
            if state not in ("OPEN", "CLOSED"):
                self.states[side] = self._pending[side] = "UNAVAILABLE"
                self._seen[side] = 0
            else:
                self._seen[side] = min(self.frames, self._seen[side] + 1) if state == self._pending[side] else 1
                self._pending[side] = state
                if self._seen[side] >= self.frames:
                    self.states[side] = state
            if prediction is not None:
                result[side] = dict(prediction, threshold_state=state, state=self.states[side])
        return result


class BilateralBlinkDetector:
    """One live blink count: confirmed BOTH OPEN -> BOTH CLOSED -> BOTH OPEN.

    Each joint state needs two consecutive observations by default. A wink,
    mixed pair, missing eye or capture gap cancels an incomplete cycle.
    Duration is between confirmed transitions, not a calibrated eyelid measurement.
    """
    def __init__(self, frames=2):
        if frames < 1:
            raise ValueError("Require a positive confirmation frame count")
        self.frames = frames
        self.detector = BlinkDetector()
        self._last_ms = None
        self.invalidate()

    @property
    def count(self):
        return self.detector.count

    @property
    def last_blink_ms(self):
        return self.detector.last_blink_ms

    def invalidate(self):
        self._pending, self._confirmed, self._seen = None, None, 0
        self.detector.invalidate()

    def update(self, predictions, timestamp_ms):
        """Raw threshold labels for both eyes + capture ms -> one completed binocular event."""
        if not math.isfinite(timestamp_ms) or (self._last_ms is not None and timestamp_ms <= self._last_ms):
            raise ValueError("Capture timestamps must be finite and increasing")
        if self._last_ms is not None and timestamp_ms - self._last_ms > self.detector.maximum_gap_ms:
            self.invalidate()
        self._last_ms = timestamp_ms
        states = [predictions.get(side, {}).get("state") for side in ("left", "right")]
        state = states[0] if states[0] == states[1] and states[0] in ("OPEN", "CLOSED") else None
        if state is None:
            self.invalidate()
            return self.detector.update(None, timestamp_ms)
        self._seen = min(self.frames, self._seen + 1) if state == self._pending else 1
        self._pending = state
        if self._seen >= self.frames:
            self._confirmed = state
        return self.detector.update({"OPEN": 0.0, "CLOSED": 1.0}.get(self._confirmed), timestamp_ms)


class BlinkEventLog:
    """Optional prediction CSV; never a source of ground-truth labels."""
    def __init__(self, path, session_id):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle)
        self.writer.writerow(("session_id", "side", "start_ms", "end_ms"))
        self.session_id = session_id
        self.origin_ms = None

    def write(self, events, detectors, timestamp_ms):
        if self.origin_ms is None:
            self.origin_ms = timestamp_ms
        for side, event in events.items():
            if event:
                end = timestamp_ms-self.origin_ms
                self.writer.writerow((self.session_id, side, end-detectors[side].last_blink_ms, end))
        self.handle.flush()

    def close(self):
        self.handle.close()


# Eye Thresholds

RULE = "CLOSED iff int(logit_CLOSED) - int(logit_OPEN) > threshold; ties OPEN"


def threshold_value(value):
    if type(value) is not int or abs(value) > 4294967295:
        raise ValueError("Threshold must be an integer within the possible INT32 logit-difference range")
    return value


def decide(logits, threshold=0):
    margin = int(logits[1]) - int(logits[0])
    return ("CLOSED" if margin > threshold_value(threshold) else "OPEN"), margin


class ManualThresholds:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.profiles = {}
        self.active = None
        self.status = "Drag sliders to adjust. SAVE keeps settings for this model."
        if self.path is not None and self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("schema_version") != 1 or saved.get("decision_rule") != RULE:
                raise ValueError("Unsupported manual threshold settings format")
            for checkpoint, profile in saved["profiles"].items():
                for side in ("left", "right"):
                    threshold_value(profile[side])
                self.profiles[checkpoint] = dict(profile)

    def activate(self, checkpoint, identifier):
        self.active = checkpoint
        if checkpoint not in self.profiles:
            self.profiles[checkpoint] = {"identifier": identifier, "left": 0, "right": 0}
        self.status = "Per-eye settings loaded for this exact INT8 checkpoint."

    def value(self, side):
        return self.profiles[self.active][side]

    def set(self, side, value):
        if side not in ("left", "right"):
            raise ValueError("Eye side must be left or right")
        self.profiles[self.active][side] = threshold_value(value)
        self.status = "Adjusted for live preview. Click SAVE to keep these settings."

    def reset(self):
        for side in ("left", "right"):
            self.set(side, 0)
        self.status = "Both eyes reset to the original model decision (threshold 0)."

    def snapshot(self):
        return {"checkpoint_sha256": self.active,
                "left": self.value("left"), "right": self.value("right")}

    def apply(self, predictions):
        adjusted = {}
        for side, original in predictions.items():
            decision, margin = decide(original["logits"], self.value(side))
            adjusted[side] = dict(original, model_state=original.get("model_state", original["state"]),
                                  state=decision, margin=margin, threshold=self.value(side))
        return adjusted

    def save(self):
        if self.path is None:
            raise ValueError("No settings destination configured")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "decision_rule": RULE,
                   "scope": "manual live adjustments; frozen validation/test scores do not include these settings",
                   "profiles": self.profiles}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)
        self.status = "SAVED. These per-eye settings will load on the next launch."


# Score Calibration

CASES = ("NORMAL OPEN", "WIDE OPEN", "CLOSED")
SIDES = ("left", "right")


def separating_threshold(open_scores, closed_scores):
    """CLOSED iff margin > T: require a gap for every observed sample."""
    if not open_scores or not closed_scores:
        return None
    highest_open, lowest_closed = max(open_scores), min(closed_scores)
    lower, upper = max(highest_open, -2147483648), min(lowest_closed - 1, 2147483647)
    if lower > upper:
        return None
    return max(lower, min(upper, (highest_open + lowest_closed) // 2))


class ScoreCalibration:
    def __init__(self, checkpoint, path, target=30):
        self.checkpoint = checkpoint
        self.path = Path(path)
        self.target = target
        self.samples = {case: [] for case in CASES}
        self.active = None
        self.ready_at = self.expires_at = 0
        self.status = "Capture NORMAL OPEN, then WIDE OPEN, then CLOSED. Hold each pose."

    def start(self, case, now):
        self.active = case
        self.samples[case] = []
        self.ready_at, self.expires_at = now + 2, now + 15
        self.status = "Hold " + case + ": sampling starts after 2 seconds."

    def observe(self, predictions, now):
        if self.active is None:
            return
        if now > self.expires_at:
            self.status = "Capture timed out. Face the camera and repeat this case."
            self.active = None
            return
        if now < self.ready_at or any(side not in predictions for side in SIDES):
            return
        self.samples[self.active].append({side: int(predictions[side]["margin"]) for side in SIDES})
        if len(self.samples[self.active]) >= self.target:
            self.status = self.active + " captured."
            self.active = None
            self.save_report()

    def proposals(self):
        if any(len(self.samples[case]) < self.target for case in CASES):
            return None
        result = {}
        for side in SIDES:
            opened = [row[side] for case in CASES[:2] for row in self.samples[case]]
            closed = [row[side] for row in self.samples["CLOSED"]]
            result[side] = separating_threshold(opened, closed)
        return result

    def save_report(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"checkpoint_sha256": self.checkpoint,
                   "scope": "operator-labelled live threshold development; not held-out accuracy",
                   "images_saved": False, "target_pairs_per_case": self.target,
                   "margins": self.samples, "proposed_thresholds": self.proposals()}
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class CalibrationUI:
    WINDOW = "AEEE score check | normal / wide / closed"

    def __init__(self, calibration):
        import cv2
        self.cv2 = cv2
        self.calibration = calibration
        self.apply_requested = False
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1000, 310)
        cv2.setMouseCallback(self.WINDOW, self.mouse)

    def mouse(self, event, x, y, flags, parameter):
        import time
        if event != self.cv2.EVENT_LBUTTONDOWN or not 70 <= y <= 120:
            return
        for index, case in enumerate(CASES):
            if 15 + index * 245 <= x <= 245 + index * 245:
                self.calibration.start(case, time.monotonic())
                return
        if 750 <= x <= 985:
            proposed = self.calibration.proposals()
            if proposed is not None and all(value is not None for value in proposed.values()):
                self.apply_requested = True

    def render(self):
        import numpy as np
        panel = np.full((310, 1000, 3), (22, 25, 28), np.uint8)
        def line(message, x, y, scale=0.6, color=(215, 225, 230)):
            self.cv2.putText(panel, message, (x, y), self.cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
        line("Use the same light and head position. Click a pose; hold it for 30 valid pairs.", 15, 30)
        for index, label in enumerate((*CASES, "APPLY + SAVE")):
            x = 15 + index * 245
            self.cv2.rectangle(panel, (x, 70), (x + 230, 120), (65, 80, 95), -1)
            line(label, x + 10, 101)
        check = self.calibration
        line(check.status, 15, 155, 0.53)
        counts = " | ".join(f"{case}: {len(check.samples[case])}/{check.target}" for case in CASES)
        line(counts, 15, 190, 0.55)
        proposed = check.proposals()
        if proposed is not None:
            for index, side in enumerate(SIDES):
                message = f"{side.upper()}: " + (f"candidate T={proposed[side]:+d}" if proposed[side] is not None
                                                else "scores overlap; threshold alone cannot fit these samples")
                line(message, 15, 225 + 25 * index, 0.52)
        line("Only scores are recorded. APPLY is enabled only if both eyes have a separating threshold.", 15, 290, 0.49)
        self.cv2.imshow(self.WINDOW, panel)


# Live Observation

class LiveObservation:
    def __init__(self, path, maximum_pairs=120):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.maximum_pairs = maximum_pairs
        self.count = 0
        self.tiles = []
        self.sheet = 0
        self.started = None
        self.last_preview = 0
        self.finished = False
        self.records = (self.path / "observations.jsonl").open("w", encoding="utf-8")

    def add(self, frame_id, captured_ns, predictions, panel, checkpoint):
        if self.finished or not all(side in predictions for side in ("left", "right")):
            return
        now = time.monotonic()
        if self.started is None:
            self.started = now
        tile = np.full((140, 340, 3), 24, np.uint8)
        record = {"frame_id": frame_id, "capture_monotonic_ns": captured_ns,
                  "checkpoint_sha256": checkpoint, "elapsed_seconds": now-self.started,
                  "scope": "unlabelled visual inspection; not training or accuracy evaluation"}
        cv2.putText(tile, f"frame {frame_id} / {now-self.started:.2f}s", (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        for index, side in enumerate(("left", "right")):
            item = predictions[side]
            x = 5 + 170*index
            roi = cv2.resize(item["roi"], (160, 80), interpolation=cv2.INTER_NEAREST)
            tile[24:104, x:x+160] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            color = (90, 220, 100) if item["state"] == "OPEN" else (40, 160, 250)
            cv2.putText(tile, f"{side}: {item['state']}", (x, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
            cv2.putText(tile, f"m{item['margin']:+d} T{item['threshold']:+d}", (x, 134),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1)
            record[side] = {key: item[key] for key in ("state", "model_state", "logits", "margin", "threshold")}
        self.records.write(json.dumps(record)+"\n")
        self.records.flush()
        self.tiles.append(tile)
        self.count += 1
        if len(self.tiles) == 24:
            self.save_sheet()
        if now-self.last_preview >= 1:
            cv2.imwrite(str(self.path/"latest_preview.jpg"), panel)
            self.last_preview = now
        if self.count >= self.maximum_pairs or now-self.started >= 20:
            self.close()

    def save_sheet(self):
        if not self.tiles:
            return
        canvas = np.full((840, 1360, 3), 24, np.uint8)
        for index, tile in enumerate(self.tiles):
            row, column = divmod(index, 4)
            canvas[row*140:(row+1)*140, column*340:(column+1)*340] = tile
        cv2.imwrite(str(self.path/f"eyes_{self.sheet:02d}.png"), canvas)
        self.sheet += 1
        self.tiles = []

    def close(self):
        if self.finished:
            return
        self.save_sheet()
        self.records.close()
        self.finished = True
        print(f"VISUAL CHECK: {self.count} paired observations saved locally to {self.path}", flush=True)


# Evaluate Blinks

def read_events(path):
    groups = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"session_id", "side", "start_ms", "end_ms"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Event CSV requires session_id,side,start_ms,end_ms")
        for row in reader:
            start, end = float(row["start_ms"]), float(row["end_ms"])
            if not row["session_id"].strip() or row["side"] not in ("left", "right", "both"):
                raise ValueError("Invalid session or eye side")
            if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
                raise ValueError("Event times must be finite and satisfy 0 <= start_ms < end_ms")
            groups.setdefault((row["session_id"], row["side"]), []).append((start, end))
    for events in groups.values():
        events.sort()
        if any(current[0] < previous[1] for previous, current in zip(events, events[1:])):
            raise ValueError("Overlapping or duplicate events within the same session/eye")
    return groups


def match_events(truth, predictions, tolerance_ms):
    """Monotone earliest reopening-time matching, with one use per event."""
    i = j = 0
    matched = []
    while i < len(truth) and j < len(predictions):
        difference = predictions[j][1]-truth[i][1]
        if difference < -tolerance_ms:
            j += 1
        elif difference > tolerance_ms:
            i += 1
        else:
            matched.append((truth[i], predictions[j]))
            i += 1
            j += 1
    return matched


def event_metrics(truth_count, prediction_count, matched):
    tp = len(matched)
    fp, fn = prediction_count-tp, truth_count-tp
    delays = [prediction[1]-truth[1] for truth, prediction in matched]
    duration_errors = [abs((prediction[1]-prediction[0])-(truth[1]-truth[0])) for truth, prediction in matched]
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": tp/prediction_count if prediction_count else None,
            "recall": tp/truth_count if truth_count else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "mean_reopening_delay_ms": sum(delays)/tp if tp else None,
            "mean_absolute_reopening_error_ms": sum(map(abs, delays))/tp if tp else None,
            "mean_absolute_duration_error_ms": sum(duration_errors)/tp if tp else None}


def evaluate_events_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label-source", required=True, help="Independent human/reference annotation provenance")
    parser.add_argument("--tolerance-ms", type=float, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if not math.isfinite(args.tolerance_ms) or args.tolerance_ms < 0:
            raise ValueError("Tolerance must be finite and nonnegative")
        if args.truth.resolve() == args.predictions.resolve():
            raise ValueError("Ground truth and predictions must be separate independent files")
        if not args.label_source.strip():
            raise ValueError("Independent annotation provenance is required")
        truth, predictions = read_events(args.truth), read_events(args.predictions)
        sessions, matched = [], []
        for session, side in sorted(truth.keys() | predictions.keys()):
            actual, predicted = truth.get((session, side), []), predictions.get((session, side), [])
            pairs = match_events(actual, predicted, args.tolerance_ms)
            matched.extend(pairs)
            sessions.append({"session_id": session, "side": side,
                             **event_metrics(len(actual), len(predicted), pairs)})
        result = {"label_source": args.label_source,
                  "truth_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
                  "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                  "policy": "One-to-one monotone matching by reopening time; not frame-level accuracy",
                  "tolerance_ms": args.tolerance_ms,
                  "overall": event_metrics(sum(map(len, truth.values())), sum(map(len, predictions.values())), matched),
                  "sessions": sessions}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2)+"\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Event evaluation failed: {exc}\n")
