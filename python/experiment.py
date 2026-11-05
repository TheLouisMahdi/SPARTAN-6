"""Shared CNN comparison and reproducible live/offline experiment sessions."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import html
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import get_ident, Lock
import time
import uuid

import numpy as np

from blink_timing import BinocularTimer, MAXIMUM_GAP_MS
from encoders import METHODS
from fpga_host import METHOD_IDS
from integer_model import infer
from live_pipeline import PairResult, ROOT, SIDES
from ocular_metrics import summarize, window_summaries, percentile

ENCODERS = ("aeee2", "grayscale2")
CLOCK_HZ = 50_000_000
COMMON = ("frame_id", "capture_ms", "encoder", "model_sha256", "validity", "error_reason",
          "left_state", "right_state", "left_open_logit", "left_closed_logit",
          "right_open_logit", "right_closed_logit", "inference_ms", "total_processing_ms",
          "pair_processing_ms", "input_left_sha256", "input_right_sha256", "cpu_fp32_ms",
          "cpu_fp32_left_open_logit", "cpu_fp32_left_closed_logit", "cpu_fp32_right_open_logit", "cpu_fp32_right_closed_logit")
FPGA = ("left_cycles", "right_cycles", "pair_cycles", "uart_roundtrip_ms", "uart_upload_ms")
COMPARISON = ("frame_id", "capture_ms", "cpu_valid", "fpga_valid", "left_logits_equal",
              "right_logits_equal", "max_abs_logit_diff", "left_prediction_equal",
              "right_prediction_equal", "cpu_inference_ms", "fpga_pair_cycles",
              "fpga_kernel_ms", "uart_roundtrip_ms", "encoder", "model_sha256", "total_processing_ms")


def eyes_from_logits(logits):
    eyes = {}
    for index, side in enumerate(SIDES):
        values = None if logits is None else [int(v) for v in logits[index]]
        margin = None if values is None else values[1] - values[0]
        state = "UNAVAILABLE" if margin in (None, 0) else "CLOSED" if margin > 0 else "OPEN"
        eyes[side] = dict(state=state, logits=values, margin=margin)
    return eyes


class ComparisonPipeline:
    """One caller owns UART; a persistent worker overlaps the CPU integer reference."""
    def __init__(self, model, mode, client=None):
        if not model.ready or model.method not in (*ENCODERS, "omz_bgr32"):
            raise ValueError("MODEL_NOT_READY: select verified pretrained weights or a trained TinyEye checkpoint")
        if mode not in ("cpu", "fpga", "compare") or (mode != "cpu" and client is None):
            raise ValueError("Invalid backend/client configuration")
        self.model, self.mode, self.client = model, mode, client
        self.pool = ThreadPoolExecutor(max_workers=1) if mode != "fpga" else None
        self.owner = get_ident()
        self.previous_id = self.previous_ns = -1

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def _cpu(self, codes):
        started = time.perf_counter_ns()
        if self.model.method == "omz_bgr32":
            logits = np.stack([self.model.predict(roi, fixed=True) for roi in codes])
            result = dict(logits=logits, inference_ms=(time.perf_counter_ns()-started)/1e6)
            if self.model.precision == "float":
                started = time.perf_counter_ns()
                result["fp32_logits"] = np.stack([self.model.predict(roi) for roi in codes])
                result["cpu_fp32_ms"] = (time.perf_counter_ns()-started)/1e6
            return result
        pixels = np.stack(codes)[:, None] * np.uint8(85)
        logits = infer(pixels, self.model.params)
        return dict(logits=logits, inference_ms=(time.perf_counter_ns() - started) / 1e6)

    def predict(self, frame_id, capture_ns, rois):
        if get_ident() != self.owner:
            raise RuntimeError("UART coordinator must have one owner")
        if not self.previous_id < frame_id <= 0xffffffff or capture_ns <= self.previous_ns:
            raise ValueError("Duplicate, out-of-order or wrapped frame")
        self.previous_id, self.previous_ns = frame_id, capture_ns
        started = time.perf_counter_ns()
        names = ("cpu", "fpga") if self.mode == "compare" else (self.mode,)
        backends = {name: dict(logits=None, error="missing_eye") for name in names}
        cycles = None
        hashes = {}
        if all(rois.get(side) is not None for side in SIDES):
            pretrained = self.model.method == "omz_bgr32"
            if pretrained:
                from pretrained_eye import input_pixels
                codes = [np.ascontiguousarray(input_pixels(rois[side]).transpose(1, 2, 0)) for side in SIDES]
            else:
                codes = [METHODS[self.model.method].encode(rois[side]) for side in SIDES]
            for code in codes:
                if not pretrained and (code.dtype != np.uint8 or code.shape != (16, 32) or np.any(code > 3)):
                    raise ValueError("Encoder violated the two-bit input contract")
                code.flags.writeable = False
            hashes = {"input_"+side+"_sha256": hashlib.sha256((code.transpose(2,0,1) if pretrained else code).tobytes()).hexdigest() for side, code in zip(SIDES, codes)}
            job = self.pool.submit(self._cpu, codes) if self.pool else None
            try:
                if self.client:
                    hardware = (self.client.infer_pretrained(*codes, frame_id) if pretrained else
                                self.client.infer(self.model.method, *codes, frame_id))
                    if hardware.frame_id != frame_id or hardware.method_id != (4 if pretrained else METHOD_IDS[self.model.method]):
                        raise RuntimeError("Mismatched FPGA frame/encoder")
                    cycles = (hardware.left_cycles, hardware.right_cycles, hardware.pair_cycles)
                    backends["fpga"] = dict(logits=np.array([hardware.left_logits, hardware.right_logits], dtype=np.int64),
                        inference_ms=hardware.pair_cycles * 1000 / CLOCK_HZ,
                        left_cycles=cycles[0], right_cycles=cycles[1], pair_cycles=cycles[2],
                        uart_roundtrip_ms=hardware.roundtrip_ms, uart_upload_ms=hardware.upload_ms)
            except (TimeoutError, RuntimeError, OSError) as exc:
                backends["fpga"] = dict(logits=None, error=str(exc))
            finally:
                if job:
                    try:
                        backends["cpu"] = job.result()
                    except Exception as exc:
                        backends["cpu"] = dict(logits=None, error=str(exc))
        selected = backends["fpga" if self.mode == "compare" else self.mode]
        for data in backends.values():
            data.update(hashes)
        return PairResult(frame_id, capture_ns, eyes_from_logits(selected["logits"]),
                          (time.perf_counter_ns() - started) / 1e6, cycles, backends)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class Session:
    def __init__(self, output, model, mode, camera, port, baud, source=None):
        stamp = datetime.now(timezone.utc)
        self.path = Path(output) if output else ROOT / "assets/runs/sessions" / (stamp.strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True, exist_ok=False)
        # Align the relative clock with the existing integer-ms timer ticks.
        self.origin = time.monotonic_ns() // 1_000_000 * 1_000_000
        self.started = time.perf_counter_ns()
        self.previous_id = 0
        self.captures, self.capture_lock = deque(), Lock()
        self.captured_count = 0
        self.streams, self.writers = {}, {}
        names = ("cpu", "fpga", "comparison") if mode == "compare" else (mode,)
        self.meta = dict(schema_version=1, session_id=self.path.name, created_utc=stamp.isoformat(),
            encoder=model.method, encoder_spec=METHODS[model.method].SPEC if model.method in METHODS else
                dict(version="omz-bgr32-v1", input=[3,32,32], preprocessing="input_pixels; BGR INTER_LINEAR resize; (pixel-127)/255"),
            architecture=model.record["architecture"],
            checkpoint_sha256=model.checkpoint_sha256, mode=mode, camera=camera,
            cpu_workers=0 if mode == "fpga" else 1,
            uart_port=port if mode != "cpu" else None, baud=baud if mode != "cpu" else None,
            fpga_clock_hz=CLOCK_HZ if mode != "cpu" else None, fpga_configuration=None, pe_count=None,
            source_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
            source_dirty=bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout),
            runtime={"python": sys.version, **{p: version(p) for p in ("numpy", "opencv-contrib-python", "mediapipe", "pyserial")}},
            status="running", files=[name + ".csv" for name in names], dropped_frames=0,
            input_bytes_per_eye=3072 if model.method == "omz_bgr32" else 128, fatigue_level="UNKNOWN",
            timing_scope={"cpu_inference_ms": "One encoded eye pair through the integer reference",
                "fpga_inference_ms": "Pair computation cycles at the project clock constraint",
                "uart_roundtrip_ms": "Inference request/response including transfer, firmware computation and host wait",
                "uart_upload_ms": "Sum of both eyes' upload request/ACK durations; includes host wait, not isolated wire time",
                "total_processing_ms": "Capture through temporal acceptance, excluding CSV/display",
                "throughput_fps": "Processed frames / actual session elapsed seconds"},
            fatigue_reason="Live and window prediction time scales have not been validated")
        if source:
            self.meta.update(source_mode="offline", **source)
            self.meta.update(model_name=model.record["architecture"], model_sha256=model.checkpoint_sha256,
                fpga_model_fingerprint=None, preprocessing_contract=self.meta["encoder_spec"],
                cpu_precision=("FP32 and fixed-point" if getattr(model, "precision", "fixed") == "float" else "fixed-point") if mode != "fpga" else None,
                execution_mode=mode, start_time_utc=self.meta["created_utc"], git_commit=self.meta["source_commit"],
                git_dirty=self.meta["source_dirty"], runtime_versions=self.meta["runtime"], experiment_status="INCOMPLETE")
        if model.method == "omz_bgr32":
            self.meta["model_output_contract"] = model.record["output_contract"]
        self.meta["software_source_sha256" if source else "source_sha256"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "python").rglob("*.py"))}
        write_json(self.path / "session.json", self.meta)
        for name in names:
            stream = (self.path / (name + ".csv")).open("x", newline="", encoding="utf-8")
            self.streams[name] = stream
            writer = csv.DictWriter(stream, COMPARISON if name == "comparison" else COMMON + (FPGA if name == "fpga" else ()))
            writer.writeheader()
            self.writers[name] = writer

    def capture(self, frame_id, captured_ns):
        with self.capture_lock:
            self.captures.append((frame_id, captured_ns))
            self.captured_count = frame_id

    def _drain(self, before=None):
        with self.capture_lock:
            captures = []
            while self.captures and (before is None or self.captures[0][0] <= before):
                captures.append(self.captures.popleft())
        for frame_id, captured_ns in captures:
            if before is not None and frame_id == before:
                continue
            self.meta["dropped_frames"] += 1
            common = dict(frame_id=frame_id, capture_ms=(captured_ns-self.origin)/1e6)
            for name, writer in self.writers.items():
                if name == "comparison":
                    writer.writerow(dict(common, cpu_valid=0, fpga_valid=0,
                        encoder=self.meta["encoder"], model_sha256=self.meta["checkpoint_sha256"]))
                else:
                    writer.writerow(dict(common, encoder=self.meta["encoder"], model_sha256=self.meta["checkpoint_sha256"],
                        validity="dropped", error_reason="capture_buffer_replaced_or_shutdown",
                        left_state="UNAVAILABLE", right_state="UNAVAILABLE"))

    def hardware(self, status):
        pretrained = self.meta["encoder"] == "omz_bgr32"
        self.meta["fpga_model_fingerprint"] = status["fingerprint"]
        self.meta["fpga_configuration"] = dict(network="pretrained" if pretrained else "tiny", pe_count=None, handshake=status,
            verification="UART model fingerprint; no bitstream/PE attestation" if pretrained else "UART fingerprint and full model byte readback; no bitstream attestation")
        # The existing protocol does not expose PE count or a clock measurement.
        self.meta["fpga_clock_source"] = "50 MHz project constraint; not measured via UART"
        write_json(self.path / "session.json", self.meta)

    def record(self, result, scores, now_ns, error=None, total_processing_ms=None):
        if result.frame_id <= self.previous_id:
            raise ValueError("Duplicate/out-of-order session frame")
        self._drain(result.frame_id)
        self.previous_id = result.frame_id
        offline = self.meta.get("source_mode") == "offline"
        capture_ms = result.capture_ns/1e6 if offline else (result.capture_ns - self.origin) / 1e6
        duration = (total_processing_ms if total_processing_ms is not None else result.processing_ms) if offline else (now_ns-result.capture_ns)/1e6
        fresh = offline or 0 <= now_ns - result.capture_ns <= MAXIMUM_GAP_MS * 1_000_000
        valid = {}
        for name, data in result.backends.items():
            logits = data["logits"]
            eyes = eyes_from_logits(logits)
            classification = offline and self.meta.get("experiment_type") in ("cew", "golden")
            accepted = fresh and not error and logits is not None and (classification or all(e["state"] != "UNAVAILABLE" for e in eyes.values()))
            valid[name] = fresh and not error and logits is not None
            validity = "stale" if not fresh else "valid" if accepted else "invalid"
            row = dict(frame_id=result.frame_id, capture_ms=capture_ms, encoder=self.meta["encoder"],
                model_sha256=self.meta["checkpoint_sha256"], validity=validity,
                error_reason=error or data.get("error", "expired_at_acceptance" if not fresh else "" if accepted else "ambiguous_logits"),
                inference_ms=data.get("inference_ms"), total_processing_ms=duration, pair_processing_ms=result.processing_ms,
                cpu_fp32_ms=data.get("cpu_fp32_ms"), **{k: data.get(k) for k in ("input_left_sha256", "input_right_sha256")})
            for side in SIDES:
                row[side + "_state"] = (("OPEN", "CLOSED")[int(np.argmax(logits[SIDES.index(side)]))] if classification else eyes[side]["state"]) if accepted else "UNAVAILABLE"
                for i, label in enumerate(("open", "closed")):
                    row[side + "_" + label + "_logit"] = None if logits is None else int(logits[SIDES.index(side), i])
                    floating = data.get("fp32_logits")
                    row["cpu_fp32_"+side+"_"+label+"_logit"] = None if floating is None else float(floating[SIDES.index(side), i])
            if name == "fpga":
                row.update({key: data.get(key) for key in FPGA})
            self.writers[name].writerow(row)
            self.streams[name].flush()
        if "comparison" in self.writers:
            cpu, fpga = result.backends["cpu"], result.backends["fpga"]
            row = dict(frame_id=result.frame_id, capture_ms=capture_ms, cpu_valid=int(valid["cpu"]),
                encoder=self.meta["encoder"], model_sha256=self.meta["checkpoint_sha256"],
                total_processing_ms=duration,
                fpga_valid=int(valid["fpga"]), cpu_inference_ms=cpu.get("inference_ms"),
                fpga_pair_cycles=fpga.get("pair_cycles"), fpga_kernel_ms=fpga.get("inference_ms"),
                uart_roundtrip_ms=fpga.get("uart_roundtrip_ms"))
            if all(valid.values()):
                a, b = cpu["logits"].astype(np.int64), fpga["logits"].astype(np.int64)
                row["max_abs_logit_diff"] = int(np.abs(a - b).max())
                for i, side in enumerate(SIDES):
                    row[side + "_logits_equal"] = int(np.array_equal(a[i], b[i]))
                    row[side + "_prediction_equal"] = int(np.argmax(a[i]) == np.argmax(b[i]))
            self.writers["comparison"].writerow(row)
            self.streams["comparison"].flush()

    def finish(self, status, error=None):
        self._drain()
        for stream in self.streams.values():
            stream.close()
        self.meta.update(status=status, error=error, elapsed_seconds=(time.perf_counter_ns()-self.started)/1e9,
                         captured_frames=self.captured_count or self.previous_id)
        if self.meta.get("source_mode") == "offline":
            self.meta["experiment_status"] = status
        write_json(self.path / "session.json", self.meta)
        regenerate(self.path)
        print("SESSION: " + str(self.path), flush=True)


def latency(values):
    return dict(mean_ms=sum(values)/len(values) if values else None, median_ms=percentile(values, .5),
                p95_ms=percentile(values, .95), max_ms=max(values, default=None))


def regenerate(directory):
    """Only recorded inputs determine JSON and HTML; no clock or network at replay."""
    directory = Path(directory)
    meta = json.loads((directory / "session.json").read_text(encoding="utf-8"))
    if meta.get('report_kind') == 'stream_pipeline_v1':
        from pipeline_report import summarize as summarize_pipeline
        return summarize_pipeline(directory,meta,meta.get('frames_path'))
    if "files" not in meta:
        write_json(directory/"summary.json", meta)
        (directory/"report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Experiment status</title><h1>'+
            html.escape(meta["status"])+'</h1><pre>'+html.escape(meta.get("error", ""))+'</pre></html>\n', encoding="utf-8")
        return meta
    temporal = meta.get("experiment_type") not in ("cew", "golden")
    result = dict(session=meta, backends={}, fatigue_level="UNKNOWN", fatigue_reason=meta["fatigue_reason"])
    for name in ("cpu", "fpga"):
        if name + ".csv" not in meta["files"]:
            continue
        with (directory / (name + ".csv")).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        samples = [dict(t_ms=float(r["capture_ms"]), **{s+"_state": r[s+"_state"] if r["validity"] == "valid" else "UNAVAILABLE" for s in SIDES}) for r in rows]
        metrics = summarize(samples) if temporal else {"eyes": {}}
        for eye in metrics.get("eyes", {}).values():
            eye["binary_closed_fraction"] = eye.pop("closed_fraction")
            eye["max_blink_ms"] = max((e["duration_ms"] for e in eye["events"]), default=None)
        timer, events = BinocularTimer(), []
        for sample in samples if temporal else []:
            scores = [{"OPEN": 0., "CLOSED": 1.}.get(sample[s+"_state"]) for s in SIDES]
            timing = timer.update(*scores, int(sample["t_ms"]) & 0xffffffff)
            if timing.event:
                events.append(dict(end_ms=sample["t_ms"], duration_ms=timing.last_duration_ms))
        accepted = sum(r["validity"] == "valid" for r in rows)
        processed = sum(r["validity"] != "dropped" for r in rows)
        backend = dict(processed_frames=processed, accepted_frames=accepted, invalid_results=processed-accepted,
            dropped_frames=meta["dropped_frames"], captured_frames=meta["captured_frames"],
            throughput_fps=processed/meta["elapsed_seconds"] if meta.get("elapsed_seconds") else None,
            ocular=metrics, bilateral=dict(count=timer.count, long_count=timer.long_count, events=events),
            windows_seconds={str(s): window_summaries(samples, s*1000) for s in (30, 60)} if temporal else {})
        if not any(eye["observed_ms"] > 0 for eye in metrics.get("eyes", {}).values()):
            backend["bilateral"].update(count=None, long_count=None)
        backend["bilateral"].update(mean_duration_ms=sum(e["duration_ms"] for e in events)/len(events) if events else None,
            max_duration_ms=max((e["duration_ms"] for e in events), default=None),
            definition="Completed binocular cycles from BinocularTimer; long threshold 600ms")
        backend["missing_data_intervals"] = [dict(start_ms=a["t_ms"], end_ms=b["t_ms"])
            for a, b in zip(samples, samples[1:]) if b["t_ms"]-a["t_ms"] > MAXIMUM_GAP_MS or any(
                row[s+"_state"] == "UNAVAILABLE" for row in (a, b) for s in SIDES)]
        for field in ("inference_ms", "cpu_fp32_ms", "total_processing_ms", "pair_processing_ms", "uart_roundtrip_ms", "uart_upload_ms"):
            backend[field] = latency([float(r[field]) for r in rows if r.get(field) != "" and r.get(field) is not None])
        backend["error_count"] = sum(r["validity"] == "invalid" for r in rows)
        backend["timeout_count"] = sum("timeout" in r["error_reason"].lower() or "no matching fpga reply" in r["error_reason"].lower() for r in rows)
        if not temporal:
            for key in ("ocular", "bilateral", "windows_seconds", "missing_data_intervals"):
                backend.pop(key)
        elif meta.get("source_mode") == "offline":
            from ocular_metrics import temporal_report
            backend["temporal"] = temporal_report(samples, meta.get("indicator_thresholds"))
        result["backends"][name] = backend
    if "comparison.csv" in meta["files"]:
        with (directory / "comparison.csv").open(newline="", encoding="utf-8") as stream:
            pairs = list(csv.DictReader(stream))
        comparable = [r for r in pairs if r["max_abs_logit_diff"] != ""]
        result["comparison"] = dict(matched_frames=len(comparable),
            exact_output_mismatches=sum(int(r["max_abs_logit_diff"]) != 0 for r in comparable),
            max_abs_logit_diff=max((int(r["max_abs_logit_diff"]) for r in comparable), default=None),
            prediction_agreement={side: sum(r[side+"_prediction_equal"] == "1" for r in comparable)/len(comparable)
                if comparable else None for side in SIDES})
    result["limitations"] = ["Predictions are not ground truth; no accuracy or system speedup established.",
        "Binary closure fractions are not calibrated PERCLOS80.",
        "Concurrent comparison is not independent backend benchmarking.",
        "Missing/stale observations break temporal continuity; final samples have no invented duration.",
        "Fatigue classification remains UNKNOWN without validation at the intended time scale."]
    if meta.get("source_mode") == "offline":
        result["limitations"] = ["Exact numerical agreement does not establish ground-truth accuracy.",
            "FP32 scores are uncalibrated; fixed-point and FP32 logits have different scales.",
            "Concurrent timings are not independent CPU-only/FPGA-only benchmarks.",
            "Binary closure is not PERCLOS80; the ocular indicator is an unvalidated engineering heuristic.",
            "UTA-RLDD labels describe predominant whole-video self-report, never eye-frame truth."]
        with (directory/"samples.csv").open(newline="", encoding="utf-8") as stream:
            sample_rows = list(csv.DictReader(stream))
        result["samples"] = dict(registered_source_samples=meta.get("registered_samples"), observations=len(sample_rows),
            decoded_source_samples=meta.get("decoded_source_samples"),
            excluded=[dict(sample_id=r["sample_id"], reason=r["error_reason"]) for r in sample_rows if r["error_reason"]])
        count = sum(any(r.get(k+"_prediction") in ("OPEN", "CLOSED") for k in ("cpu", "fpga")) for r in sample_rows)
        result["samples"].update(classified_observations=count,
            throughput_samples_per_second=count/meta["elapsed_seconds"] if meta.get("elapsed_seconds") else None)
        if meta["experiment_type"] == "cew":
            result["accuracy"] = cew_accuracy(sample_rows)
        elif meta["experiment_type"] == "video":
            ocular = {name: backend["temporal"] for name,backend in result["backends"].items()}
            write_json(directory/"ocular.json", ocular)
            result["processed_video_count"] = int(meta["status"] == "PASS")
            result["processed_video_frames_per_second"] = meta["captured_frames"]/meta["elapsed_seconds"] if meta.get("elapsed_seconds") else None
        if not temporal:
            result.pop("fatigue_level", None)
            result.pop("fatigue_reason", None)
    write_json(directory / "summary.json", result)
    title = meta["architecture"] + " session " + meta["session_id"]
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>' + html.escape(title) + '</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #aaa}</style>',
        '<h1>' + html.escape(title) + '</h1><p>Status: '+html.escape(meta["status"])+'</p>']
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(v) if v is not None else 'Unavailable')+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    body.append(table(("Dataset", "Model", "Mode", "Model SHA256"), [(meta.get("source_dataset", "live"), meta["architecture"], meta["mode"], meta["checkpoint_sha256"])]))
    body.append('<h2>Measured pair latency (ms)</h2>')
    timing_rows = []
    for name, backend in result["backends"].items():
        for field in ("inference_ms", "cpu_fp32_ms", "uart_upload_ms", "uart_roundtrip_ms", "pair_processing_ms", "total_processing_ms"):
            stats = backend[field]
            if stats["mean_ms"] is not None:
                timing_rows.append((name, field, *(stats[k] for k in ("mean_ms", "median_ms", "p95_ms", "max_ms"))))
    body.append(table(("Backend", "Stage", "Mean", "Median", "P95", "Maximum"), timing_rows))
    if "samples" in result:
        sample_stats = result["samples"]
        body.append(table(("Registered sources", "Decoded sources", "Classified observations", "Excluded observations", "Samples/second"),
            [(sample_stats["registered_source_samples"], sample_stats["decoded_source_samples"], sample_stats["classified_observations"], len(sample_stats["excluded"]), sample_stats["throughput_samples_per_second"])]))
    if "comparison" in result:
        pair_stats = result["comparison"]
        body.append('<h2>Exact fixed-point comparison</h2>'+table(("Matched transactions", "Mismatches", "Maximum integer difference"),
            [(pair_stats["matched_frames"], pair_stats["exact_output_mismatches"], pair_stats["max_abs_logit_diff"])]))
    if "accuracy" in result:
        body.append('<h2>CEW accuracy (independent labels)</h2>')
        for name, metric in result["accuracy"]["overall"].items():
            body.append('<h3>'+html.escape(name)+'</h3><p>Accuracy: '+html.escape(str(metric["accuracy"]))+
                '; Macro-F1: '+html.escape(str(metric["macro_f1"]))+'</p><p>Ground-truth rows; prediction columns: OPEN, CLOSED</p><table>')
            for row in metric["confusion_matrix"]:
                body.append('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>')
            body.append('</table>')
    for name, backend in result["backends"].items() if temporal else []:
        body.append('<h2>' + name.upper() + ' ocular trend (30 seconds)</h2><table><tr><th>Start ms</th><th>End ms</th><th>Eye</th><th>Coverage</th><th>Binary closed fraction</th></tr>')
        for window in backend["windows_seconds"]["30"]:
            for side, eye in window["eyes"].items():
                body.append('<tr>' + ''.join('<td>'+html.escape(str(v))+'</td>' for v in (window["window_start_ms"], window["window_end_ms"], side, eye["coverage_fraction"], eye["closed_fraction"])) + '</tr>')
        body.append('</table>')
        if "temporal" in backend:
            body.append('<h2>Experimental Ocular Fatigue Indicator</h2><p>Proposed engineering heuristic; independent of UTA-RLDD labels.</p><table><tr><th>Start ms</th><th>End ms</th><th>Indicator</th><th>Bilateral coverage</th><th>Closed fraction</th></tr>')
            for window in backend["temporal"]["windows_seconds"]["60"]:
                body.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (window["window_start_ms"], window["window_end_ms"], window["indicator"], window["bilateral"]["coverage"], window["bilateral"]["binary_closed_fraction"]))+'</tr>')
            body.append('</table>')
    body.append('<h2>Limitations</h2><ul>'+''.join('<li>'+html.escape(v)+'</li>' for v in result["limitations"])+'</ul>')
    body.append('<details><summary>Reproducibility metadata and recorded measurements</summary><pre>' + html.escape(json.dumps(result, indent=2, sort_keys=True)) + '</pre></details></html>\n')
    (directory / "report.html").write_text('\n'.join(body), encoding="utf-8")
    return result


def report_cli(argv=None):
    parser = argparse.ArgumentParser(description="Regenerate offline session JSON and HTML")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    regenerate(args.directory)
    return 0


SAMPLE_FIELDS = ("sample_id", "source_sample_id", "source_path", "source_sha256", "source_type", "subject_id",
    "label_provenance", "label_scope", "label", "source_color_mode", "frame_id", "side", "media_timestamp_ms",
    "input_sha256", "cpu_prediction", "cpu_fp32_prediction", "fpga_prediction", "cpu_valid", "fpga_valid", "error_reason")


def cew_accuracy(rows):
    from research import metrics
    def calculate(selected, key):
        pairs = [(r["label"], r.get(key)) for r in selected if r.get(key) in ("OPEN", "CLOSED")]
        if not pairs:
            return dict(count=0, accuracy=None, macro_f1=None, precision=None, recall=None, confusion_matrix=[[0,0],[0,0]])
        actual, predicted = zip(*[(int(a == "CLOSED"), int(b == "CLOSED")) for a,b in pairs])
        result = metrics(actual, np.asarray(predicted, dtype=np.int64))
        result.update(precision=float(np.mean([v["precision"] for v in result["per_class"].values()])),
                      recall=float(np.mean([v["recall"] for v in result["per_class"].values()])))
        return result
    names = {"cpu_fixed":"cpu_prediction", "cpu_fp32":"cpu_fp32_prediction", "fpga":"fpga_prediction"}
    available = {k:v for k,v in names.items() if any(r.get(v) for r in rows)}
    matched = [r for r in rows if r.get("cpu_prediction") and r.get("cpu_fp32_prediction")]
    return dict(overall={name:calculate(rows,key) for name,key in available.items()},
        by_source_type={source:{name:calculate([r for r in rows if r["source_type"] == source],key) for name,key in available.items()}
                        for source in sorted({r["source_type"] for r in rows})},
        fp32_fixed_decision_agreement=dict(count=len(matched), fraction=sum(r["cpu_prediction"] == r["cpu_fp32_prediction"] for r in matched)/len(matched) if matched else None),
        denominator="Successfully classified eye observations; exclusions reported separately. Two eyes are not two subjects.",
        source_labels="Face-level labels apply to extracted eyes; patch labels apply to one eye file")


def create_localizer(video=False):
    from capture import mp, vision, MODEL_PATH
    options = vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH),
        delegate=mp.tasks.BaseOptions.Delegate.CPU), running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)


def locate_eyes(image, localizer, timestamp_ms=None):
    import cv2
    from capture import mp, color_eye_rois
    value = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = localizer.detect(value) if timestamp_ms is None else localizer.detect_for_video(value, timestamp_ms)
    return color_eye_rois(image, result.face_landmarks[0]) if result.face_landmarks else {}


def read_color_image(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError("Unreadable or non-uint8 image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), "grayscale_replicated"
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Unsupported source color format")
    return np.ascontiguousarray(image), "original_bgr"


def cew_observations(rows, meta, localizer):
    from data_tools import sha256
    for source in rows:
        base = dict(source_sample_id=source["sample_id"], **{k:source[k] for k in ("source_path", "source_sha256", "source_type", "subject_id", "label", "label_provenance")})
        sides = ("patch",) if source["source_type"] == "eye_patch" else SIDES
        rois, error, color = {}, "", ""
        try:
            if sha256(Path(source["source_path"])) != source["source_sha256"]:
                raise ValueError("Source SHA256 changed since registration")
            image, color = read_color_image(source["source_path"])
            meta["decoded_source_samples"] += 1
            rois = {"patch":image} if sides == ("patch",) else locate_eyes(image, localizer)
        except (OSError, ValueError) as exc:
            error = str(exc)
        for side in sides:
            yield dict(base, sample_id=source["sample_id"]+"_"+side, label_scope="eye" if side == "patch" else "face",
                source_color_mode=color, image=rois.get(side), error_reason=error or ("" if side in rois else "missing_eye"))


def sample_row(observation, result=None, index=None, temporal=False):
    row = {key:observation.get(key, "") for key in SAMPLE_FIELDS}
    if result is not None:
        row.update(frame_id=result.frame_id, side=SIDES[index], media_timestamp_ms=result.capture_ns/1e6 if temporal else "")
        errors = [row["error_reason"]] if row["error_reason"] else []
        for name,data in result.backends.items():
            logits = data["logits"]
            valid = logits is not None
            row[name+"_valid"] = int(valid)
            if valid:
                state = eyes_from_logits(logits)[SIDES[index]]["state"] if temporal else ("OPEN", "CLOSED")[int(np.argmax(logits[index]))]
                row[name+"_prediction"] = state if state != "UNAVAILABLE" else ""
            if name == "cpu" and data.get("fp32_logits") is not None:
                row["cpu_fp32_prediction"] = ("OPEN", "CLOSED")[int(np.argmax(data["fp32_logits"][index]))]
            row["input_sha256"] = data.get("input_"+SIDES[index]+"_sha256", "")
            if data.get("error"):
                errors.append(name+": "+data["error"])
        row["error_reason"] = "; ".join(errors)
    return row


def offline_source(args):
    from data_tools import sha256
    from public_data import read_cew_manifest, read_registry, registered_videos, CEW_URL
    from ocular_metrics import pts_milliseconds, indicator_policy
    if args.workload == "cew":
        rows = read_cew_manifest(args.manifest)
        provenance = json.loads((args.manifest.parent/"provenance.json").read_text(encoding="utf-8-sig"))
        if (provenance.get("approved_local_use") is not True or provenance.get("source_url") != CEW_URL
                or provenance.get("manifest_sha256") != sha256(args.manifest)):
            raise ValueError("CEW manifest/rights provenance mismatch; register the authorized source first")
        return rows, dict(experiment_type="cew", source_dataset="CEW", source_sha256=sha256(args.manifest),
            source_license_or_terms_reference=CEW_URL, dataset_provenance=provenance, registered_samples=len(rows),
            decoded_source_samples=0, pair_semantics="independent_test_samples", padding_policy="Odd final eye repeats only as an unscored UART padding input",
            replay_policy="NOT_APPLICABLE", indicator_thresholds=None)
    if args.workload == "golden":
        return [], dict(experiment_type="golden", source_dataset="deterministic engineering vectors", source_sha256=None,
            source_license_or_terms_reference="Locally specified input arrays; not dataset accuracy evidence",
            registered_samples=4, decoded_source_samples=4, pair_semantics="independent_engineering_vectors", replay_policy="NOT_APPLICABLE", indicator_thresholds=None)
    registry = read_registry(args.registry)
    videos = [v for entry in registry["datasets"] if entry["dataset"] == "UTA-RLDD" for v in registered_videos(entry)]
    if len({v["video_id"] for v in videos}) != len(videos) or len({v["sha256"] for v in videos}) != len(videos):
        raise ValueError("Duplicate registered video IDs/content")
    matches = [v for v in videos if v["video_id"] == args.video_id]
    if len(matches) != 1:
        raise ValueError("video-id must match one registered UTA-RLDD clip")
    video = matches[0]
    try:
        probe_version = subprocess.run([str(args.ffprobe), "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except FileNotFoundError as exc:
        raise ValueError("NOT_RUN: ffprobe is required for original video presentation timestamps") from exc
    timestamps = pts_milliseconds(video["path"], args.ffprobe)
    return (video,timestamps), dict(experiment_type="video", source_dataset="UTA-RLDD", source_sha256=video["sha256"],
        source_license_or_terms_reference=video["source_url"], dataset_provenance=video, registry_sha256=sha256(args.registry),
        registered_samples=len(timestamps), decoded_source_samples=0, pair_semantics="same_frame_bilateral_observation",
        replay_policy="OFFLINE_UNTHROTTLED", timebase="Original presentation timestamps, relative integer milliseconds; never nominal FPS",
        ffprobe=str(args.ffprobe), ffprobe_version=probe_version, indicator_thresholds=indicator_policy(), video_label_scope="predominant whole-video self-report only",
        clip_self_report_level={"0":"ALERT", "5":"LOW_VIGILANCE", "10":"DROWSY"}[video["source_label"]])


def run_offline(args):
    import cv2
    from pretrained_eye import PretrainedEye, URL, CHECKSUM, LICENSE_URL
    source, metadata = offline_source(args)
    try:
        model = PretrainedEye(precision="fixed" if getattr(args, "cpu_precision", "both") == "fixed" else "float")
    except (OSError, ValueError) as exc:
        raise ValueError("MODEL_NOT_READY: "+str(exc)) from exc
    session = Session(args.output, model, args.mode, None, args.port, 1_000_000,
        source=dict(metadata, official_model_source=URL, official_model_sha384=CHECKSUM, model_license=LICENSE_URL))
    session.meta["files"].append("samples.csv")
    client = pipeline = localizer = capture = samples_stream = None
    status, error, code = "INCOMPLETE", None, 1
    try:
        samples_stream = (session.path/"samples.csv").open("x", newline="", encoding="utf-8")
        writer = csv.DictWriter(samples_stream, SAMPLE_FIELDS)
        writer.writeheader()
        if args.mode != "cpu":
            try:
                from fpga_host import FPGAClient
                client = FPGAClient(args.port, baud=1_000_000)
                verified = client.load_model(model)
            except (OSError, TimeoutError) as exc:
                status = "NO_FPGA"
                raise RuntimeError("NO_FPGA: "+str(exc)) from exc
            session.hardware(verified)
        pipeline = ComparisonPipeline(model, args.mode, client)

        def process(frame_id, timestamp_ms, observations, rois, started):
            result = pipeline.predict(frame_id, int(timestamp_ms*1_000_000), rois)
            session.record(result, None, time.monotonic_ns(), total_processing_ms=(time.perf_counter_ns()-started)/1e6)
            for index, observation in enumerate(observations):
                writer.writerow(sample_row(observation, result, index, args.workload == "video"))
            samples_stream.flush()
            if client is not None:
                counters = (getattr(client.reader, "bad_packets", None), getattr(client, "unmatched_responses", None))
                if any(isinstance(value, int) and value > 0 for value in counters):
                    raise RuntimeError("UART integrity gate failed: malformed/CRC packet or unmatched response")
            if any(data.get("error") != "missing_eye" and data.get("error") for data in result.backends.values()):
                raise RuntimeError("Inference failure: "+"; ".join(data["error"] for data in result.backends.values() if data.get("error")))
            if result.backends.get("cpu", {}).get("logits") is not None and result.backends.get("fpga", {}).get("logits") is not None:
                if not np.array_equal(result.backends["cpu"]["logits"], result.backends["fpga"]["logits"]):
                    raise RuntimeError("Exact CPU/FPGA integer mismatch; diagnostic logits retained")

        if args.workload == "video":
            video,timestamps = source
            localizer = create_localizer(video=True)
            capture = cv2.VideoCapture(video["path"])
            if not capture.isOpened():
                raise ValueError("Cannot open registered video")
            for index, timestamp in enumerate(timestamps):
                started = time.perf_counter_ns()
                ok, frame = capture.read()
                if not ok or frame is None or not frame.size:
                    raise ValueError("Video decode interrupted before final registered PTS")
                session.meta["decoded_source_samples"] += 1
                rois = locate_eyes(frame, localizer, timestamp)
                records = [dict(sample_id=f"{video['video_id']}:{index}:{side}", source_sample_id=video["video_id"], source_path=video["path"],
                    source_sha256=video["sha256"], source_type="video_frame", subject_id=video["subject_id"], label_scope="no_eye_state_ground_truth",
                    source_color_mode="video_decoded_bgr", error_reason="" if side in rois else "missing_eye") for side in SIDES]
                process(index+1, timestamp, records, rois, started)
            ok,_ = capture.read()
            if ok:
                raise ValueError("Decoded frame count exceeds registered PTS count")
        else:
            if args.workload == "cew":
                if any(row["source_type"] != "eye_patch" for row in source):
                    localizer = create_localizer()
                observations = cew_observations(source, session.meta, localizer)
            else:
                images = [np.full((32,32,3),v,np.uint8) for v in (0,127,255)]
                images.append(np.arange(3072,dtype=np.uint8).reshape(32,32,3))
                observations = [dict(sample_id=f"golden_{i}", source_sample_id=f"golden_{i}", source_type="engineering_vector",
                    source_sha256=hashlib.sha256(image.tobytes()).hexdigest(), source_color_mode="specified_bgr", image=image)
                    for i,image in enumerate(images)]
            pending, frame_id = [], 0
            started = time.perf_counter_ns()
            for observation in observations:
                if observation.get("image") is None:
                    writer.writerow(sample_row(observation))
                    samples_stream.flush()
                    continue
                pending.append(observation)
                if len(pending) < 2:
                    continue
                frame_id += 1
                process(frame_id, frame_id, pending, {s:o["image"] for s,o in zip(SIDES,pending)}, started)
                pending = []
                started = time.perf_counter_ns()
            if pending:
                frame_id += 1
                process(frame_id, frame_id, pending, dict(left=pending[0]["image"],right=pending[0]["image"]), started)
            if session.previous_id == 0:
                raise ValueError("No valid eye inputs were available for inference")
        status, code = "PASS", 0
    except KeyboardInterrupt:
        error, code = "Interrupted experiment", 130
    except Exception as exc:
        error = str(exc)
        if session.previous_id == 0 and status != "NO_FPGA" and "pretrained CNN bitstream" in error:
            status = "FAIL"
    finally:
        if client is not None:
            bad_packets = getattr(getattr(client, "reader", None), "bad_packets", None)
            session.meta["uart_bad_packets"] = bad_packets if isinstance(bad_packets, int) else None
            for name in ("request_count", "timeout_count", "unmatched_responses"):
                value = getattr(client, name, None)
                session.meta["uart_"+name] = value if isinstance(value, int) else None
        for resource in (samples_stream, pipeline, localizer, capture, client):
            if resource is not None:
                try:
                    resource.release() if resource is capture else resource.close()
                except Exception as exc:
                    error, status, code = str(exc), "INCOMPLETE", 1
        try:
            session.finish(status, error)
        except Exception as exc:
            session.meta.update(status="INCOMPLETE", experiment_status="INCOMPLETE", error="Output failure: "+str(exc))
            write_json(session.path/"session.json", session.meta)
            code = 1
    print(json.dumps(dict(status=session.meta["status"], output=str(session.path), error=session.meta.get("error"))))
    return code


def experiment_cli(argv=None):
    parser = argparse.ArgumentParser(description="Offline evaluation with the verified pretrained CNN; never opens a webcam or programs hardware")
    sub = parser.add_subparsers(dest="workload", required=True)
    comparison=sub.add_parser('pipeline-compare')
    comparison.add_argument('--runs',nargs='+',type=Path,required=True)
    comparison.add_argument('--output',type=Path,required=True)
    for name in ("cew", "video", "golden"):
        command = sub.add_parser(name)
        command.add_argument("--mode", choices=("cpu", "fpga", "compare"), default="compare" if name == "golden" else "cpu")
        command.add_argument("--port", default="COM26")
        command.add_argument("--cpu-precision", choices=("both", "fixed"), default="both",
                             help="both: measure FP32 and fixed references; fixed: one reference pass for CPU-only throughput")
        command.add_argument("--output", type=Path)
        if name == "cew":
            command.add_argument("--manifest", type=Path, default=ROOT/"assets/datasets/cew/manifest.csv")
        elif name == "video":
            command.add_argument("--registry", type=Path, required=True)
            command.add_argument("--video-id", required=True)
            command.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    if args.workload == 'pipeline-compare':
        from pipeline_report import compare
        result=compare(args.runs,args.output)
        print(json.dumps(result,indent=2))
        return 0 if result['status']=='PASS' else 1
    args.output = args.output or ROOT/"assets/runs/experiments"/(args.workload+"_"+args.mode+"_"+uuid.uuid4().hex[:12])
    if args.output.exists():
        parser.error("--output must be a new directory")
    if args.mode != "cpu" and not args.port.strip():
        parser.error("--port must be nonempty")
    try:
        return run_offline(args)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        reason = str(exc)
        status = "MODEL_NOT_READY" if "MODEL_NOT_READY" in reason else "NOT_RUN" if "NOT_RUN:" in reason else "DATA_NOT_AVAILABLE" if isinstance(exc, FileNotFoundError) or "DATA_NOT_AVAILABLE" in reason else "FAIL"
        diagnostic = dict(schema_version=1, status=status, experiment_status=status, error=reason, experiment_type=args.workload)
        try:
            if isinstance(exc, FileExistsError):
                raise OSError("Existing output was preserved")
            if not args.output.exists():
                args.output.mkdir(parents=True)
                write_json(args.output/"session.json", diagnostic)
                regenerate(args.output)
            elif (args.output/"session.json").is_file():
                try:
                    diagnostic = json.loads((args.output/"session.json").read_text(encoding="utf-8"))
                except ValueError:
                    pass
                diagnostic.update(status="INCOMPLETE", experiment_status="INCOMPLETE", error=reason)
                write_json(args.output/"session.json", diagnostic)
        except OSError as output_error:
            diagnostic["output_error"] = str(output_error)
        print(json.dumps(dict(status=diagnostic["status"], error=reason, output=str(args.output), output_error=diagnostic.get("output_error"))))
        return 1


def test_cli(argv=None):
    parser = argparse.ArgumentParser(description="Sequential offline regressions; never programs hardware")
    parser.add_argument("repeat", type=int, nargs="?", default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 20:
        parser.error("repeat must be in 1..20")
    output = ROOT / "assets/runs" / ("regression_" + uuid.uuid4().hex[:12])
    output.mkdir(parents=True)
    commands = [["-m", "unittest", "discover", "-s", "research_ext", "-p", "test_*.py"],
                ["python/app.py", "pretrained", "verify", "--offline"],
                ["python/app.py", "rtl", "simulate"], ["python/app.py", "rtl", "system"],
                ["python/app.py", "pretrained", "verify-uart"],
                ["research_ext/verify_timing.py", "simulate"]]
    for run in range(args.repeat):
        for index, command in enumerate(commands):
            command = list(command)
            if index:
                command += ["--output", str(output / f"run_{run+1}_{index}")]
            with (output / f"run_{run+1}_{index}.log").open("w") as log:
                code = subprocess.run([sys.executable, "-B", *command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, EYE_RUN_TRAINING_FIXTURES="0")).returncode
            print(f"run={run+1} stage={index} exit={code} output={output}", flush=True)
            if code:
                return code
    print("NOT_RUN_NO_BOARD: physical board tests not requested")
    return 0
