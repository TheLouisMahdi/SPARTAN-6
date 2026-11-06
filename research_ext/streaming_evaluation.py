"""Offline feasibility evidence and bounded legacy-UART measurements. No camera.

Experimental codecs are deliberately not sent to the legacy FPGA firmware.
"""
import argparse
from collections import deque
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import queue
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
from stream_codec import Decoder, encode, pack_runs, RAW, ZERO, XOR_ZERO, HYBRID, OVERHEAD


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def stats(values):
    return {"mean": statistics.mean(values), "p95": float(np.percentile(values, 95)),
            "max": max(values)} if values else None


def sequences(count=64):
    rng = np.random.default_rng(1701)
    texture = rng.integers(0, 256, (3, 32, 32), np.uint8)
    other = rng.integers(0, 256, texture.shape, np.uint8)
    return {
        "synthetic_identical": [texture.tobytes()] * count,
        "synthetic_gradual_all_pixels": [((texture.astype(np.uint16)+i) % 256).astype(np.uint8).tobytes() for i in range(count)],
        "synthetic_sparse_changes": [np.where(np.indices(texture.shape)[2] == i % 32, other, texture).astype(np.uint8).tobytes() for i in range(count)],
        "synthetic_translation": [np.roll(texture, i, axis=2).tobytes() for i in range(count)],
        "synthetic_alternating_states": [(texture if i % 2 else other).tobytes() for i in range(count)],
        "synthetic_random": [rng.integers(0, 256, texture.shape, np.uint8).tobytes() for _ in range(count)],
        "synthetic_uniform": [bytes([i % 256])*3072 for i in range(count)],
        "synthetic_max_difference": [bytes([255*(i % 2)])*3072 for i in range(count)],
    }


def registered_eyes(limit=None):
    import cv2
    from pretrained_eye import input_pixels
    rows = list(csv.DictReader((ROOT / "assets/datasets/cew/manifest.csv").open(newline="", encoding="utf-8-sig")))
    selected = rows if limit is None else sorted(rows, key=lambda r: r["source_sha256"])[:limit]
    for row in selected:
        path = Path(row["source_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError("registered CEW hash mismatch")
        image = cv2.imread(str(path))
        yield row, input_pixels(image).tobytes(), image


def compression(output):
    # A synthetic frame is repeated for left/right; the temporal axis is frame i.
    sources = {name: [eye for data in frames for eye in (data, data)]
               for name, frames in sequences().items()}
    sources["CEW_STATIC_INDEPENDENT_NOT_TEMPORAL"] = [data for _, data, _ in registered_eyes()]
    results = []
    for name, eyes in sources.items():
        for mode in (RAW, ZERO, XOR_ZERO, HYBRID):
            decoder = Decoder()
            refs = [None, None]
            sizes, costs, operations, expansion = [], [], [], []
            fallback = 0
            for i, data in enumerate(eyes):
                eye, frame = i % 2, i // 2
                # Independent CEW photographs always start with a keyframe.
                key = name.startswith("CEW")
                started = time.perf_counter_ns()
                record, candidate = encode(data, eye, frame, frame*20_000_000,
                                           frame//4, mode, refs[eye], key=key and mode == XOR_ZERO)
                costs.append((time.perf_counter_ns()-started)/1e6)
                decoded = decoder.decode(record)
                if decoded["data"] != data:
                    raise AssertionError("lossless codec mismatch")
                refs[eye] = (frame, data)
                sizes.append(len(record))
                operations.append(decoded["decode_operations"])
                expansion.append(candidate / 3072)
                fallback += int(decoded["codec"] == RAW and mode != RAW)
            # Proposed stop/wait record encapsulation, NOT measured wire traffic.
            pair_wire = 2*statistics.mean([s + 25*math.ceil(s/2048) + 41 for s in sizes])
            results.append({"source": name, "mode": mode, "eyes": len(eyes),
                            "raw_bytes": len(eyes)*3072, "record_bytes": sum(sizes),
                            "record_ratio": sum(sizes)/(len(eyes)*3072),
                            "encoding_ms_per_eye": stats(costs), "decode_operations": stats(operations),
                            "worst_candidate_expansion": max(expansion), "raw_fallback_eyes": fallback,
                            "independent_keyframe_bytes": 3072+OVERHEAD,
                            "estimated_pair_wire_bytes": pair_wire,
                            "wire_only_upper_pairs_s": 100000/pair_wire,
                            "wire_50Hz_possible": pair_wire <= 2000,
                            "integer_byte_mismatches": 0,
                            "timing_provenance": "HOST_MEASUREMENT",
                            "wire_provenance": "MODELLED_ESTIMATE_NOT_IMPLEMENTED_PROTOCOL"})
    adversarial = bytes([1, 0])*1536
    encoded, candidate = encode(adversarial, 0, 0, 0, 0, ZERO)
    if candidate != 4608 or Decoder().decode(encoded)["data"] != adversarial:
        raise AssertionError("adversarial RAW fallback failed")
    write(output / "compression.json", {"record_overhead": OVERHEAD,
          "expansion_scope": "per-source maxima are observed, not global bounds",
          "adversarial_zero_RLE_candidate_bytes": candidate,
          "adversarial_zero_RLE_expansion": candidate/3072,
          "adversarial_RAW_fallback_record_bytes": len(encoded),
          "hybrid_all_literal_candidate_bytes": len(pack_runs(bytes(range(256))*12, repeat=True)),
          "all_codecs_provable_record_byte_bound_after_RAW_fallback": 3072+OVERHEAD,
          "synthetic_eyes": "same generated tensor on both eyes; identities/references remain separate",
          "CEW_scope": "4846 independent static photographs; never temporal evidence",
          "results": results})
    return results


def model_stream(period_ms, batch, service_ms=133.38, count=200, capacity=9, max_wait_ms=20):
    """Finite credit model; encoder/wire/decoder/kernel/result serialized, no overlap."""
    outstanding = deque()
    end = 0.
    trace = []
    maximum = 0
    for i in range(count):
        arrival = i*period_ms
        while outstanding and outstanding[0] <= arrival:
            outstanding.popleft()
        if len(outstanding) == capacity:
            trace.append({"frame_id": i, "arrival_ms": arrival, "status": "REJECTED_FULL"})
            continue
        group_first = (i//batch)*batch*period_ms
        release = max(arrival, min(group_first+(batch-1)*period_ms, group_first+max_wait_ms))
        start = max(release, end)
        end = start+service_ms
        outstanding.append(end)
        maximum = max(maximum, len(outstanding))
        trace.append({"frame_id": i, "arrival_ms": arrival, "status": "VALID",
                      "batch_id": i//batch, "batch_wait_ms": release-arrival,
                      "backpressure_ms": start-release, "completed_ms": end,
                      "end_to_end_ms": end-arrival})
    valid = [r for r in trace if r["status"] == "VALID"]
    return {"provenance": "MODELLED_ESTIMATE", "period_ms": period_ms, "batch": batch,
            "max_batch_wait_ms": max_wait_ms, "capacity_including_active": capacity,
            "arrived": count, "completed": len(valid), "rejected": count-len(valid),
            "max_occupancy": maximum, "successful_pairs_s": len(valid)*1000/end,
            "e2e_ms": stats([r["end_to_end_ms"] for r in valid]),
            "unbounded_batch_fill_ms": (batch-1)*period_ms,
            "trace": trace}


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in tracked if p and (ROOT/p).is_file()}
    write(output / "source_before.json", hashes)
    (output / "status_before.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    write(output / "baseline.json", {"local_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
          "remote_main": subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT).decode().strip(),
          "kernel_eye_cycles": 367739, "kernel_pair_ms": 14.70956, "legacy_wire_pair_bytes": 6388,
          "previous_physical_eyes": 4846, "previous_integer_mismatches": 0,
          "slices": 1318, "LUTs": 4204, "registers": 2745, "BRAM18": 29, "DSP": 10,
          "period_ns": 19.431, "slack_ns": .569,
          "bitstream_sha256": hashlib.sha256((ROOT/"fpga/eye_top.bit").read_bytes()).hexdigest(),
          "baseline_evidence": "assets/runs/physical_validation_20261105_ffc1e3c4/summary.json",
          "resource_evidence": "fpga/eye_top_map.mrp; fpga/eye_top.twr",
          "camera_access": False})
    compression(output)
    write(output / "model.json", {"phases_ms": {"encoding": 0, "wire_minimum": 63.88,
          "decoder": 0, "kernel": 14.70956, "USB_host_and_serial_excess": 133.38-63.88-14.70956},
          "phase_note": "legacy calibrated service estimate; no proposed decoder measurement",
          "runs": [model_stream(period, batch) for period in (10, 16.67, 20, 33.33, 40, 100) for batch in (1, 2, 4, 8)]})


def physical(output, sustained_seconds=60):
    """Only configuration A: unchanged legacy RAW; logical batches roll immediately.

    The source owns its arrival journal; one consumer owns the UART. Rejected
    identities are reconstructed from deterministic source indices and emitted
    before the next accepted identity, so overload needs no unbounded status RAM.
    """
    from fpga_host import FPGAClient, encode_packet
    from pretrained_eye import PretrainedEye
    from serial.tools import list_ports
    ports = [p for p in list_ports.comports() if p.vid == 0x10c4 and p.pid == 0xea60
             and p.serial_number == "04D5A2633F36EF11B7ABFF13A04C7D1C"]
    if len(ports) != 1:
        raise RuntimeError("expected the registered board CP210x")
    model = PretrainedEye(precision="fixed")
    # Select four registered examples in each of all four source directories.
    selected = {}
    for row, data, image in registered_eyes():
        group = Path(row["source_path"]).parent.name
        if len(selected.setdefault(group, [])) < 4:
            selected[group].append((row["sample_id"], image))
    fixtures = [item for group in sorted(selected) for item in selected[group]]
    rng = np.random.default_rng(417)
    fixtures += [(f"synthetic_uniform_{v}", np.full((32, 32, 3), v, np.uint8)) for v in (0, 127, 255)]
    fixtures += [("synthetic_random_417", rng.integers(0, 256, (32, 32, 3), np.uint8))]
    expected = [tuple(int(x) for x in model.predict(image, fixed=True)) for _, image in fixtures]
    client = FPGAClient(ports[0].device, 1000000)

    class Meter:
        def __init__(self, serial):
            self.serial = serial
            self.tx = self.rx = 0
        def __getattr__(self, name):
            return getattr(self.serial, name)
        def write(self, data):
            count = self.serial.write(data)
            self.tx += count
            return count
        def read(self, size):
            data = self.serial.read(size)
            self.rx += len(data)
            return data

    meter = Meter(client.serial)
    client.serial = meter
    runs = []
    try:
        identity = client.load_model(model)
        corrupt = bytearray(encode_packet(1, 0xdeadbeef))
        corrupt[-1] ^= 1
        meter.write(corrupt)
        client.ping()  # Bad CRC must not produce a stale/misattributed response.
        try:
            client.request(0x20, b"\xff\xff\x01")
        except RuntimeError as exc:
            if "05" not in str(exc):
                raise
        else:
            raise AssertionError("invalid upload offset accepted")
        client.ping()

        def trial(name, period_ns, count, batch):
            pending = queue.Queue(maxsize=8)
            stop = threading.Event()
            done = threading.Event()
            source_errors = []
            state = {"arrived": 0, "accepted": 0, "rejected": 0,
                     "max_queue": 0, "backpressure_ns": 0}
            origin = time.perf_counter_ns()+50_000_000
            base = 0xfffffff0
            folder = output / name
            folder.mkdir(exist_ok=False)

            def source():
                try:
                    with (folder / "arrivals.jsonl").open("w") as journal:
                        for i in range(count):
                            if stop.is_set():
                                break
                            scheduled = origin + i*(period_ns or 0)
                            if period_ns:
                                if stop.wait(max(0, (scheduled-time.perf_counter_ns())/1e9)):
                                    break
                            produced = time.perf_counter_ns()
                            state["arrived"] += 1
                            if period_ns:
                                try:
                                    pending.put_nowait(i)
                                    status = "ACCEPTED"
                                except queue.Full:
                                    status = "REJECTED_FULL"
                            else:
                                before = time.perf_counter_ns()
                                while not stop.is_set():
                                    try:
                                        pending.put(i, timeout=.1)
                                        break
                                    except queue.Full:
                                        continue
                                if stop.is_set():
                                    break
                                state["backpressure_ns"] += time.perf_counter_ns()-before
                                status = "ACCEPTED"
                            state["accepted" if status == "ACCEPTED" else "rejected"] += 1
                            state["max_queue"] = max(state["max_queue"], pending.qsize())
                            journal.write(json.dumps({"index": i, "frame_id": (base+i)&0xffffffff,
                                "capture_ns": scheduled, "producer_observed_ns": produced,
                                "status": status})+"\n")
                except Exception as exc:
                    source_errors.append(repr(exc))
                finally:
                    done.set()

            producer = threading.Thread(target=source, name="offline-independent-source")
            producer.start()
            next_result = completed = mismatches = 0
            rtts, uploads, kernels, e2es, finishes = [], [], [], [], []
            tx0, rx0 = meter.tx, meter.rx
            started = time.perf_counter_ns()
            try:
                with (folder / "ordered_results.jsonl").open("w") as sink:
                    def rejection(i):
                        sink.write(json.dumps({"index": i, "frame_id": (base+i)&0xffffffff,
                            "capture_ns": origin+i*(period_ns or 0), "batch_id": i//batch,
                            "status": "REJECTED_FULL", "emitted_ns": time.perf_counter_ns()})+"\n")
                    while not done.is_set() or not pending.empty():
                        try:
                            i = pending.get(timeout=.05)
                        except queue.Empty:
                            continue
                        for missing in range(next_result, i):
                            rejection(missing)
                        a, b = (2*i)%len(fixtures), (2*i+1)%len(fixtures)
                        result = client.infer_pretrained(fixtures[a][1], fixtures[b][1], (base+i)&0xffffffff)
                        finished = time.perf_counter_ns()
                        mismatch = int(tuple(result.left_logits) != expected[a])+int(tuple(result.right_logits) != expected[b])
                        mismatches += mismatch
                        completed += 1
                        rtts.append(result.roundtrip_ms)
                        uploads.append(result.upload_ms)
                        kernels.append(result.pair_cycles/50000)
                        if period_ns:
                            e2es.append((finished-origin-i*period_ns)/1e6)
                        finishes.append(finished)
                        sink.write(json.dumps({"index": i, "frame_id": result.frame_id,
                            "capture_ns": origin+i*(period_ns or 0), "batch_id": i//batch,
                            "source_ids": [fixtures[a][0], fixtures[b][0]], "status": "VALID",
                            "left_logits": result.left_logits, "right_logits": result.right_logits,
                            "left_cycles": result.left_cycles, "right_cycles": result.right_cycles,
                            "emitted_ns": finished, "integer_mismatches": mismatch})+"\n")
                        next_result = i+1
                    for missing in range(next_result, state["arrived"]):
                        rejection(missing)
            finally:
                stop.set()
                producer.join(2)
            if producer.is_alive() or source_errors:
                raise RuntimeError(f"source failure: {source_errors}")
            elapsed = (time.perf_counter_ns()-started)/1e9
            terminal = [json.loads(line) for line in (folder/"ordered_results.jsonl").read_text().splitlines()]
            if [r["index"] for r in terminal] != list(range(state["arrived"])):
                raise AssertionError("missing, duplicate or unordered terminal identity")
            if completed != state["accepted"] or mismatches:
                raise AssertionError("physical correctness failure")
            summary = dict(state, name=name, provenance="PHYSICAL_MEASUREMENT", configuration="A_LEGACY_RAW",
                source="OFFLINE_SYNTHETIC_SCHEDULE_CEW_STATIC_AND_SYNTHETIC_TENSORS",
                timestamp_note="capture_ns is a synthetic schedule, not measured capture",
                batch_id_scope="host bookkeeping only; rolling, no batch-fill wait; legacy wire has no frame metadata",
                period_ns=period_ns, batch_size=batch, completed=completed, dropped=0,
                integer_mismatches=mismatches, terminal_records=len(terminal), duration_s=elapsed,
                tx_bytes=meter.tx-tx0, rx_bytes=meter.rx-rx0,
                wire_bytes_per_completed_pair=(meter.tx-tx0+meter.rx-rx0)/completed,
                successful_pairs_s=completed/elapsed,
                peak_intercompletion_pairs_s=1e9/min(np.diff(finishes)) if len(finishes)>1 else None,
                rtt_ms=stats(rtts), upload_ms=stats(uploads), kernel_ms=stats(kernels),
                end_to_end_ms=stats(e2es), encoding_ms=0, decoder_cycles=None,
                backpressure_ms=state["backpressure_ns"]/1e6,
                queue_capacity=8, active_uart_credit=1,
                crc_errors=client.reader.bad_packets, timeouts=client.timeout_count,
                sequence_errors=client.unmatched_responses)
            write(folder / "summary.json", summary)
            runs.append(summary)
            write(output / "streaming.json", {"identity": identity, "runs": runs})
            print(name, completed, state["rejected"], round(summary["successful_pairs_s"], 3), flush=True)

        for repeat in (1, 2):
            for batch in (1, 2, 4, 8):
                trial(f"unpaced_b{batch}_r{repeat}", None, 32, batch)
            for rate in (10, 25, 30, 50, 60, 100):
                for batch in (1, 2, 4, 8):
                    trial(f"paced_{rate}Hz_b{batch}_r{repeat}", round(1e9/rate), 32, batch)
        trial("sustained_50Hz", 20_000_000, sustained_seconds*50, 4)
        rollback = client.ping()
        write(output/"rollback.json", {"status": "PASS", "identity": rollback,
              "physical_integer_mismatches": sum(r["integer_mismatches"] for r in runs),
              "injected_bad_request_CRC": 1, "invalid_offset_rejected": True,
              "camera_access": False})
    finally:
        client.close()


def full_cew(output):
    """Fresh physical equivalence only; not an independent accuracy evaluation."""
    from fpga_host import FPGAClient
    from pretrained_eye import PretrainedEye
    from serial.tools import list_ports
    ports = [p for p in list_ports.comports() if p.vid == 0x10c4 and p.pid == 0xea60
             and p.serial_number == "04D5A2633F36EF11B7ABFF13A04C7D1C"]
    if len(ports) != 1:
        raise RuntimeError("registered board not found")
    model = PretrainedEye(precision="fixed")
    client = FPGAClient(ports[0].device, 1000000)
    count = mismatches = 0
    folders = {}
    started = time.perf_counter_ns()
    try:
        identity = client.load_model(model)
        source = iter(registered_eyes())
        with (output/"full_cew_eyes.jsonl").open("x") as log:
            for first in source:
                second = next(source)
                result = client.infer_pretrained(first[2], second[2], count//2)
                if result.left_cycles != 367739 or result.right_cycles != 367739:
                    raise AssertionError("baseline cycle count changed")
                for item, actual in zip((first, second), (result.left_logits, result.right_logits)):
                    row, _, image = item
                    reference = tuple(int(v) for v in model.predict(image, fixed=True))
                    mismatch = tuple(actual) != reference
                    mismatches += int(mismatch)
                    group = Path(row["source_path"]).parent.name
                    folders[group] = folders.get(group, 0)+1
                    log.write(json.dumps({"sample_id": row["sample_id"], "sha256": row["source_sha256"],
                        "fixed_logits": reference, "hardware_logits": actual, "mismatch": mismatch})+"\n")
                    count += 1
                if count % 200 == 0:
                    log.flush()
                    print("CEW", count, "mismatches", mismatches, flush=True)
                if mismatches:
                    raise AssertionError("physical CEW integer mismatch")
        write(output/"full_cew.json", {"status": "PASS", "provenance": "PHYSICAL_MEASUREMENT",
            "scope": "reused registered static CEW numerical equivalence, not new accuracy",
            "eyes": count, "integer_mismatches": mismatches, "source_folders": folders,
            "identity": identity, "duration_s": (time.perf_counter_ns()-started)/1e9,
            "crc_errors": client.reader.bad_packets, "timeouts": client.timeout_count,
            "sequence_errors": client.unmatched_responses, "camera_access": False})
    finally:
        client.close()


def report(output):
    baseline = json.loads((output/"baseline.json").read_text())
    measured = json.loads((output/"streaming.json").read_text())
    codec = json.loads((output/"compression.json").read_text())
    full = json.loads((output/"full_cew.json").read_text())
    regression_log = (output/"regression.log").read_text()
    if any(f"stage={stage} exit=0" not in regression_log for stage in range(6)):
        raise AssertionError("original test_all did not complete all six stages")
    regression_path = Path(re.search(r"stage=0 exit=0 output=(.+)", regression_log).group(1).strip())
    unit_log = (regression_path/"run_1_0.log").read_text(errors="replace")
    unit_count = int(re.search(r"Ran (\d+) tests", unit_log).group(1))
    skipped = int(re.search(r"OK \(skipped=(\d+)\)", unit_log).group(1))
    verification = {"test_all": "PASS", "Python_tests": unit_count,
        "Python_passed": unit_count-skipped, "Python_opt_in_training_skipped": skipped,
        "original_RTL_stages_passed": 5, "regression_path": str(regression_path),
        "memory_probe_simulation": "PASS" if "MEMORY_GATE_SIM_PASS" in (output/"memory_sim.log").read_text() else "FAIL"}
    original = json.loads((output/"source_before.json").read_text())
    protected = {p: value for p, value in original.items()
                 if p.startswith("fpga/rtl/") or p.startswith("fpga/generated/")
                 or p in ("fpga/board_pins.ucf", "fpga/eye_top.xise", "test_all.bat")}
    changed = [p for p, value in protected.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != value]
    if changed or hashlib.sha256((ROOT/"fpga/eye_top.bit").read_bytes()).hexdigest() != baseline["bitstream_sha256"]:
        raise AssertionError(f"baseline changed: {changed}")
    if full["eyes"] != 4846 or full["integer_mismatches"]:
        raise AssertionError("incomplete physical equivalence")
    gate = (output/"memory_gate.syr").read_text(errors="replace")
    if "RAMB16BWER                  : 4" not in gate or "RAMB8BWER                   : 1" not in gate:
        raise AssertionError("memory synthesis gate not verified")
    runs = measured["runs"]
    sustained = next(r for r in runs if r["name"] == "sustained_50Hz")
    unpaced = [r for r in runs if r["period_ns"] is None]
    service = statistics.mean(r["rtt_ms"]["mean"] for r in unpaced)
    write(output/"model.json", {"provenance": "MODELLED_ESTIMATE", "calibration_rtt_ms": service,
        "phase_ms": {"source_period": "per run", "encoding": 0, "legacy_wire_minimum": 63.88,
            "codec_decode": None, "kernel": 14.70956, "result_wire_included_in_wire": True,
            "USB_host_serial_excess_estimate": service-63.88-14.70956},
        "notes": "RAW only; serial phases, no overlap. Capacity is 8 queued + 1 active; 20 ms maximum batch wait.",
        "runs": [model_stream(period, batch, service_ms=service) for period in (10, 16.67, 20, 33.33, 40, 100) for batch in (1, 2, 4, 8)]})
    extensions = {".py": "Python", ".v": "Verilog", ".vh": "Verilog", ".sv": "SystemVerilog",
                  ".tcl": "Tcl", ".bat": "Batchfile", ".cmd": "Batchfile", ".ps1": "PowerShell"}
    counts = {}
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    for p in set(paths):
        file = ROOT/p
        language = extensions.get(file.suffix.lower())
        if p and file.is_file() and language:
            counts[language] = counts.get(language, 0)+file.stat().st_size
    remote_languages = json.loads((output/"github_languages.json").read_text())
    language = {"GitHub_API_bytes": remote_languages,
        "GitHub_python_percent": remote_languages["Python"]*100/sum(remote_languages.values()),
        "GitHub_scope": "published baseline; no new push; GitHub languages endpoint read only",
        "local_source_bytes_estimate": counts,
        "local_python_percent_estimate": counts["Python"]*100/sum(counts.values()),
        "local_scope": "extension-based bytes, includes tests and headers; NOT GitHub Linguist",
        "production_Verilog_added_bytes": 0, "below_80_target_met": False}
    write(output/"language.json", language)
    summary = {"status": "BASELINE_PRESERVED_NO_GAIN", "configuration": "A_LEGACY_RAW_ONLY",
        "optimized": None, "new_protocol_implemented": False, "new_bitstream": None,
        "candidate_gate": {"status": "REJECTED_MEMORY_BEFORE_PROTOCOL", "ISE": "14.7 XST",
            "reference_bytes": 6144, "staging_bytes": 3072, "additional_full_BRAM": 4,
            "additional_half_BRAM": 1, "additional_LUTs": 21, "additional_registers": 3,
            "baseline_plus_full_blocks": 33, "device_full_blocks": 32,
            "route": "NOT_RUN_RESOURCE_GATE_FAILED", "physical_candidate": "NOT_PROGRAMMED",
            "scope": "selected overlap architecture, not proof every serialized architecture is impossible"},
        "baseline_preserved": True, "baseline_protected_source_files": len(protected),
        "bitstream_sha256": baseline["bitstream_sha256"], "model_sha256": "4daa100034482525a26c9afb9297c16580a531189e66e3d2b2ac7d32becfd593",
        "physical_trials": len(runs), "physical_pairs": sum(r["completed"] for r in runs),
        "physical_integer_mismatches": sum(r["integer_mismatches"] for r in runs)+full["integer_mismatches"],
        "fresh_full_CEW": full, "sustained_50Hz": sustained,
        "unpaced_throughput_trials": [r["successful_pairs_s"] for r in unpaced],
        "batch_comparison": [{"batch": b, "successful_pairs_s": [r["successful_pairs_s"] for r in unpaced if r["batch_size"] == b]} for b in (1, 2, 4, 8)],
        "speedup": None, "bandwidth_saved": 0, "new_50Hz_support": False,
        "language": language, "camera_access": False, "flash_programmed": False,
        "verification": verification,
        "task_source_files": ["README.md", "docs/VALIDATION_REPORT.md", "docs/STREAMING_MICROBATCH.md",
            "python/stream_codec.py", "research_ext/streaming_evaluation.py", "research_ext/test_stream_codec.py"],
        "commit": None, "commit_reason": "optimization resource gate failed; no completed optimization commit",
        "required_title_if_accepted": "Optimization: Streaming Micro-Batch NPU", "pushed": False}
    write(output/"comparison.json", summary)
    (output/"git_status_final.txt").write_bytes(subprocess.check_output(["git", "status", "--short"], cwd=ROOT))
    table = "".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        r["name"], r["arrived"], r["completed"], r["rejected"], f'{r["successful_pairs_s"]:.4f}',
        f'{r["rtt_ms"]["mean"]:.3f}', r["max_queue"]))+"</tr>" for r in runs)
    compression_table = "".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in (
        r["source"], r["mode"], round(r["estimated_pair_wire_bytes"], 2),
        r["wire_50Hz_possible"], round(r["encoding_ms_per_eye"]["mean"], 4), r["integer_byte_mismatches"]))+"</tr>" for r in codec["results"])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Streaming micro-batch feasibility</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:1400px}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:.4rem;text-align:left}}pre{{white-space:pre-wrap}}</style>
<h1>BASELINE_PRESERVED_NO_GAIN</h1><p>Offline inputs only. No camera. No new protocol or optimized bitstream accepted.</p>
<p>Original test_all: {unit_count} Python tests ({unit_count-skipped} passed, {skipped} pre-existing opt-in training skip), all five original RTL stages passed. Additional Icarus memory probe passed. No existing test was removed.</p>
<p>The selected overlapped reference/staging memory probe used 4 full + 1 half BRAM in ISE 14.7. Baseline already uses 29 full: the candidate exceeds the 32-block device. Decoder/control cost is additional. No candidate route or programming.</p>
<h2>Legacy versus optimized</h2><table><tr><th>Metric</th><th>Legacy measured</th><th>Optimized</th></tr>
<tr><td>Codec / wire bytes per pair</td><td>RAW / {sustained['wire_bytes_per_completed_pair']}</td><td>Not implemented</td></tr>
<tr><td>Upload / RTT mean ms</td><td>{sustained['upload_ms']['mean']:.3f} / {sustained['rtt_ms']['mean']:.3f}</td><td>Unavailable</td></tr>
<tr><td>Pair kernel cycles / ms</td><td>735478 / 14.70956</td><td>Unchanged baseline</td></tr>
<tr><td>50 Hz run</td><td>{sustained['completed']} / {sustained['arrived']} completed; {sustained['rejected']} explicitly rejected; queue max 8</td><td>No new 50 Hz support</td></tr>
<tr><td>Fresh full CEW</td><td>4846 eyes, zero integer mismatches</td><td>No alternative firmware</td></tr>
<tr><td>Slices / LUT / FF / DSP / BRAM18</td><td>1318 / 4204 / 2745 / 10 / 29</td><td>No accepted replacement</td></tr>
<tr><td>Baseline period / slack / Fmax</td><td>19.431 ns / +0.569 ns / 51.464155 MHz</td><td>Not routed</td></tr></table>
<h2>Physical trials</h2><p>Two repetitions of unpaced batches and independent paced schedules. Every accepted result is released immediately. Batch metadata is host-only. All {sum(r['completed'] for r in runs)} completed pairs were checked against fixed CPU logits; all terminal identities checked in order, including uint32 wrap. The sustained run lasted {sustained['duration_s']:.3f} s including drain, completed {sustained['successful_pairs_s']:.4f} pairs/s, and had p95 end-to-end latency {sustained['end_to_end_ms']['p95']:.3f} ms. No speedup is claimed.</p>
<table><tr><th>Trial</th><th>Arrived</th><th>Completed</th><th>Rejected</th><th>Pairs/s</th><th>RTT mean ms</th><th>Max queue</th></tr>{table}</table>
<h2>Codec reference measurements</h2><p>Codec 0 RAW, 1 zero RLE, 2 same-eye XOR/RLE, 3 literal/repeat hybrid. Encoding time is HOST_MEASUREMENT. Wire size/50 Hz columns are MODELLED_ESTIMATE for an unimplemented encapsulation, not hardware throughput. CEW keys are static, never temporal-video evidence. Separate JSON contains p95, decode operations, expansion and RAW fallback counts.</p>
<table><tr><th>Source</th><th>Codec</th><th>Estimated pair wire bytes</th><th>Wire-only 50 Hz possible</th><th>Encode mean ms/eye</th><th>Byte mismatches</th></tr>{compression_table}</table>
<h2>Language accounting</h2><p>GitHub baseline Python {language['GitHub_python_percent']:.3f}%. Local extension estimate {language['local_python_percent_estimate']:.3f}% (not Linguist). Production Verilog added: zero. Below-80% target not met.</p>
<h2>Evidence</h2><ul><li><a href="baseline.json">Baseline</a></li><li><a href="compression.json">Compression</a></li><li><a href="model.json">Bounded timing model with traces</a></li><li><a href="streaming.json">Physical trials</a></li><li><a href="full_cew.json">Full CEW</a></li><li><a href="comparison.json">Final comparison</a></li><li><a href="rollback.json">Legacy rollback</a></li><li><a href="sustained_50Hz/ordered_results.jsonl">Ordered terminal trace</a></li><li><a href="memory_gate.syr">ISE memory probe</a></li></ul>
<p>Protected baseline source hashes and BIT are unchanged. Existing historical 4846-eye evidence is preserved. No completed-optimization commit was created; no push or shutdown performed.</p></html>'''
    (output/"report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--full-cew", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--sustained-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.report:
        report(args.output.resolve())
    elif args.full_cew:
        full_cew(args.output.resolve())
    elif args.physical:
        physical(args.output.resolve(), args.sustained_seconds)
    else:
        prepare(args.output.resolve())
