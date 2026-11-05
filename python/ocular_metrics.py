"""Descriptive, validity-aware ocular metrics. Not a fatigue diagnosis.

Extends the research package's standard-library fatigue_metrics reference.
Intervals use the previous state only when both endpoints are available and
the gap is <= max_gap_ms. The final sample has no invented duration.
"""
import csv
import json
import math
from pathlib import Path
import statistics


def observations(path):
    """Read replay frames CSV (timestamp_ms) or reference CSV (t_ms); never infer labels."""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        item = {"t_ms": float(row.get("t_ms", row.get("timestamp_ms", row.get("capture_ms", ""))))}
        for side in ("left", "right"):
            item[side + "_state"] = (row[side + "_state"] if row.get("validity", "valid") == "valid" else "UNAVAILABLE")
            value = row.get(side + "_open_fraction", "")
            item[side + "_open_fraction"] = None if value == "" else float(value)
        result.append(item)
    validate(result)
    return result


def validate(samples):
    """Check finite nonnegative strictly increasing ms, explicit states and optional [0,1] aperture."""
    previous = -1.0
    for row in samples:
        timestamp = row["t_ms"]
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
            raise ValueError("Video timestamps must be finite, nonnegative and strictly increasing")
        previous = timestamp
        for side in ("left", "right"):
            if row[side + "_state"] not in ("OPEN", "CLOSED", "UNAVAILABLE"):
                raise ValueError("Expected OPEN, CLOSED or UNAVAILABLE")
            aperture = row.get(side + "_open_fraction")
            if aperture is not None and (not math.isfinite(aperture) or not 0 <= aperture <= 1):
                raise ValueError("Calibrated aperture must be finite and in [0,1]")


def percentile(values, fraction):
    """Linear interpolation on sorted finite numeric values; None for no observations."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def summarize(samples, *, start_ms=None, end_ms=None, max_gap_ms=250,
              min_blink_ms=30, max_blink_ms=600, long_closure_ms=500,
              aperture_provenance=None, minimum_rate_duration_ms=30000):
    """Observation dictionaries -> per-eye time/event summary over [start_ms,end_ms).

    Aperture contributes to PERCLOS only with calibration_method, source_sha256
    and subject_id provenance. Missing coverage produces null, never CLOSED.
    Events must start and finish within the window; no bilateral double count.
    Rates require >=30s valid time by default. Thresholds are experimental.
    """
    validate(samples)
    if not 0 < min_blink_ms <= max_blink_ms or max_gap_ms <= 0 or long_closure_ms <= 0:
        raise ValueError("Invalid ocular duration thresholds")
    if not samples:
        return {"status": "insufficient_data", "eyes": {}, "duration_ms": 0}
    begin = samples[0]["t_ms"] if start_ms is None else start_ms
    end = samples[-1]["t_ms"] if end_ms is None else end_ms
    if not math.isfinite(begin) or not math.isfinite(end) or end < begin:
        raise ValueError("Invalid window bounds")
    duration = end - begin
    calibrated = bool(aperture_provenance and all(aperture_provenance.get(k) for k in
                       ("calibration_method", "source_sha256", "subject_id")))
    result = {"status": "computed_descriptive_only" if duration else "insufficient_data",
              "window_start_ms": begin, "window_end_ms": end, "duration_ms": duration,
              "aperture_provenance": aperture_provenance if calibrated else None,
              "definitions": {"closed_fraction": "time weighted binary CLOSED; not PERCLOS",
                  "perclos80": "calibrated aperture <=0.20; short-blink inclusive; calibrated valid time denominator",
                  "blink": "completed OPEN->CLOSED->OPEN, both transitions inside this window",
                  "long_closure": f"completed observed closure >= {long_closure_ms}ms",
                  "bilateral": "per-eye only; left/right counts must not be summed as human blinks",
                  "rate": f"per minute of valid time, null below {minimum_rate_duration_ms}ms observed",
                  "intervals": "previous-state hold with both endpoints valid; omit gaps; final point has no duration"},
              "eyes": {}}
    for side in ("left", "right"):
        observed = closed = aperture_ms = perclos_ms = 0.0
        closures, events, event_segments, gaps, segment = [], [], [], 0, 0
        observed_closures = []
        armed = None
        for a, b in zip(samples, samples[1:]):
            ta, tb = a["t_ms"], b["t_ms"]
            state, next_state = a[side + "_state"], b[side + "_state"]
            dt = min(tb, end) - max(ta, begin)
            if dt <= 0:
                continue
            if tb - ta > max_gap_ms or "UNAVAILABLE" in (state, next_state):
                gaps += 1
                segment += 1
                armed = None
                continue
            observed += dt
            closed += dt if state == "CLOSED" else 0
            if state == "CLOSED":
                lo, hi = max(ta, begin), min(tb, end)
                if observed_closures and observed_closures[-1]["end_ms"] == lo:
                    observed_closures[-1]["end_ms"] = hi
                    observed_closures[-1]["duration_ms"] = hi-observed_closures[-1]["start_ms"]
                else:
                    observed_closures.append(dict(start_ms=lo, end_ms=hi, duration_ms=hi-lo))
            aperture = a.get(side + "_open_fraction")
            if calibrated and aperture is not None and b.get(side + "_open_fraction") is not None:
                aperture_ms += dt
                perclos_ms += dt if aperture <= 0.2 else 0
            if state == "OPEN" and next_state == "CLOSED" and begin <= tb < end:
                armed = tb
            elif state == "CLOSED" and next_state == "OPEN":
                if armed is not None and tb <= end:
                    closure = {"start_ms": armed, "end_ms": tb, "duration_ms": tb - armed}
                    closures.append(closure)
                    if min_blink_ms <= closure["duration_ms"] <= max_blink_ms:
                        events.append(closure)
                        event_segments.append(segment)
                armed = None
        durations = [event["duration_ms"] for event in events]
        intervals = [b["start_ms"] - a["start_ms"] for i, (a, b) in enumerate(zip(events, events[1:]))
                     if event_segments[i] == event_segments[i + 1]]
        coverage = observed / duration if duration else None
        result["eyes"][side] = {
            "observed_ms": observed, "coverage_fraction": coverage,
            "closed_fraction": closed / observed if observed else None,
            "perclos80_fraction": perclos_ms / aperture_ms if aperture_ms else None,
            "perclos80_calibrated_ms": aperture_ms, "gap_or_unavailable_intervals": gaps,
            "blink_count": len(events) if observed else None,
            "blink_rate_per_min": len(events) * 60000 / observed if observed >= minimum_rate_duration_ms else None,
            "mean_blink_ms": statistics.mean(durations) if durations else None,
            "median_blink_ms": statistics.median(durations) if durations else None,
            "p90_blink_ms": percentile(durations, 0.9),
            "mean_interblink_ms": statistics.mean(intervals) if intervals else None,
            "long_closure_count": sum(c["duration_ms"] >= long_closure_ms for c in closures) if observed else None,
            "events": events,
            "observed_closures": observed_closures,
            "longest_observed_closure_ms": max((c["duration_ms"] for c in observed_closures), default=0) if observed else None,
            "status": "UNAVAILABLE" if not observed else "low_coverage" if coverage < 0.8 else
                      "insufficient_duration" if observed < minimum_rate_duration_ms else "descriptive_only"}
    return result


def indicator_policy():
    return json.loads((Path(__file__).resolve().parents[1]/"research_ext/protocol.json").read_text())["indicator"]


def ocular_indicator(coverage, observed_ms, closed_fraction, longest_ms, *, full_window=True, policy=None):
    policy = policy or indicator_policy()
    if (not full_window or coverage is None or observed_ms is None or closed_fraction is None or longest_ms is None
            or not all(math.isfinite(v) for v in (coverage, observed_ms, closed_fraction, longest_ms))
            or coverage < policy["minimum_coverage"] or observed_ms < policy["minimum_observed_ms"]):
        return "UNKNOWN"
    if closed_fraction >= policy["high_closed_fraction"] or longest_ms >= policy["high_closure_ms"]:
        return "HIGH"
    if closed_fraction >= policy["moderate_closed_fraction"] or longest_ms >= policy["moderate_closure_ms"]:
        return "MODERATE"
    return "LOW"


def temporal_report(samples, policy=None):
    """Existing interval metrics plus binocular timer events; no learned fatigue model."""
    from blink_timing import BinocularTimer
    policy = policy or indicator_policy()
    validate(samples)
    joint = []
    for row in samples:
        states = [row[s+"_state"] for s in ("left", "right")]
        state = "UNAVAILABLE" if "UNAVAILABLE" in states else "CLOSED" if states == ["CLOSED", "CLOSED"] else "OPEN"
        joint.append(dict(t_ms=row["t_ms"], left_state=state, right_state=state))

    def window(begin=None, end=None):
        result = summarize(samples, start_ms=begin, end_ms=end)
        joint_result = summarize(joint, start_ms=begin, end_ms=end)
        eye = joint_result.get("eyes", {}).get("left", {})
        timer, events = BinocularTimer(), []
        for row in samples:
            if (begin is not None and row["t_ms"] < begin) or (end is not None and row["t_ms"] > end):
                continue
            timing = timer.update(*[{"OPEN":0., "CLOSED":1.}.get(row[s+"_state"]) for s in ("left", "right")], int(row["t_ms"]) & 0xffffffff)
            if timing.event:
                events.append(dict(end_ms=row["t_ms"], duration_ms=timing.last_duration_ms))
        observed = eye.get("observed_ms", 0)
        durations = [e["duration_ms"] for e in events]
        result["bilateral"] = dict(observed_ms=observed, coverage=eye.get("coverage_fraction"),
            binary_closed_fraction=eye.get("closed_fraction"), longest_observed_closure_ms=eye.get("longest_observed_closure_ms"),
            closure_intervals=eye.get("observed_closures", []), blink_count=timer.count if observed else None,
            blink_rate_per_min=timer.count*60000/observed if observed >= 30000 else None,
            blink_events=events, long_closure_count=timer.long_count if observed else None,
            mean_blink_ms=statistics.mean(durations) if durations else None, max_blink_ms=max(durations, default=None))
        result["indicator"] = ocular_indicator(eye.get("coverage_fraction"), observed, eye.get("closed_fraction"),
            eye.get("longest_observed_closure_ms"), full_window=result["duration_ms"] == policy["window_ms"], policy=policy)
        return result

    result = dict(title="Experimental Ocular Fatigue Indicator", thresholds=policy, summary=window(), windows_seconds={})
    for seconds in (30, 60):
        windows = []
        if len(samples) > 1:
            begin, last = samples[0]["t_ms"], samples[-1]["t_ms"]
            while begin < last:
                end = min(begin+seconds*1000, last)
                item = window(begin, end)
                item["partial_window"] = end-begin < seconds*1000
                windows.append(item)
                begin += seconds*1000
        result["windows_seconds"][str(seconds)] = windows
    result["missing_intervals"] = [dict(start_ms=a["t_ms"], end_ms=b["t_ms"])
        for a,b in zip(joint, joint[1:]) if b["t_ms"]-a["t_ms"] > 250 or "UNAVAILABLE" in (a["left_state"], b["left_state"])]
    return result


def window_summaries(samples, window_ms=30000, **options):
    """Nonoverlapping media-time windows; the last partial window remains explicitly partial."""
    validate(samples)
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if len(samples) < 2:
        return []
    start, last = samples[0]["t_ms"], samples[-1]["t_ms"]
    result = []
    while start < last:
        end = min(last, start + window_ms)
        result.append(summarize(samples, start_ms=start, end_ms=end, **options))
        result[-1]["partial_window"] = end - start < window_ms
        start += window_ms
    return result


def export_metrics(frames_path, output, **options):
    """Existing replay CSV -> new metadata-only JSON plus 30s/60s window CSV; no face retention."""
    rows = observations(frames_path)
    windows = {str(seconds): window_summaries(rows, seconds * 1000, **options) for seconds in (30, 60)}
    result = {"summary": summarize(rows, **options), "windows_seconds": windows,
              "scope": "predictions, not independent eye/blink/drowsiness ground truth"}
    output = Path(output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    fields = ("window_seconds", "start_ms", "end_ms", "eye", "observed_ms", "coverage_fraction",
              "closed_fraction", "perclos80_fraction", "blink_count", "blink_rate_per_min", "median_blink_ms",
              "long_closure_count", "status")
    with output.with_suffix(".csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seconds, records in windows.items():
            for record in records:
                for side, eye in record["eyes"].items():
                    writer.writerow(dict(window_seconds=seconds, start_ms=record["window_start_ms"],
                        end_ms=record["window_end_ms"], eye=side, **{k: eye[k] for k in fields[4:]}))
    return result


def pts_milliseconds(video, ffprobe="ffprobe"):
    """Decode presentation timestamps using ffprobe; return integer relative ms or fail explicitly.

    No wall-clock substitute. Unsupported ffprobe, absent/nonmonotonic PTS and
    sub-ms duplicate quantization are errors, since MediaPipe VIDEO needs unique ms.
    """
    import subprocess
    command = [str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "frame=best_effort_timestamp_time,pts_time,pkt_pts_time", "-of", "json", str(video)]
    process = subprocess.run(command, capture_output=True, text=True, check=True)
    frames = json.loads(process.stdout).get("frames", [])
    times = []
    for frame in frames:
        value = next((frame[key] for key in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time")
                      if frame.get(key) not in (None, "N/A")), None)
        if value is None:
            raise ValueError("Decoder exposes no presentation timestamp; use a current ffprobe or an explicitly verified CFR source")
        times.append(float(value) * 1000)
    if not times or not all(math.isfinite(t) for t in times):
        raise ValueError("No valid video presentation timestamps")
    result = [round(t - times[0]) for t in times]
    if any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError("Repeated/backward PTS or timestamps indistinguishable at millisecond resolution")
    return result
