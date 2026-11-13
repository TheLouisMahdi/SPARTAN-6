"""Pin-level UART integration simulation of the real RTL, using synthetic data."""
from dataclasses import astuple
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np

from blink_timing import BinocularTimer
from encoders import METHODS
from fpga_host import METHOD_IDS, PacketReader, encode_packet, eye_payload, model_image
from integer_model import infer
from rtl_evidence import ROOT, synthetic_parameters, write_hex, baseline_sources
from data_tools import sha256


def simulate_system(output, pe_count=2, baseline_ref=None):
    """Exercise boot rejection, model loading, both CNNs, CRC recovery and blink timing.

    Serial bit timing is accelerated; compute-cycle measurements from this bench
    are engineering fixtures, not real camera throughput or accuracy evidence.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("rtl", "generated"):
        shutil.copytree(ROOT / "FPGA" / folder, output / folder)
    shutil.copy2(ROOT / "RESEARCH/tb_system.v", output)
    result = {"status": "FAIL", "pe_count": pe_count, "scope": "offline synthetic pin-level UART integration; no board or accuracy measurement"}
    requests, expected, sizes, reply_flags, led_values = [], [], [], [], []

    def add(kind, payload=b"", *, response=None, leds=0xff, corrupt=False):
        sequence = len(sizes) + 1
        packet = bytearray(encode_packet(kind, sequence, payload))
        if corrupt:
            packet[-1] ^= 1
        requests.extend(packet)
        sizes.append(len(packet))
        reply_flags.append(not corrupt)
        led_values.append(leds)
        if not corrupt:
            expected.append((kind | 0x80, sequence, response))

    params = synthetic_parameters()
    image = model_image(params)
    fingerprint = int(hashlib.sha256(image).hexdigest()[:8], 16)
    method = "grayscale8"
    pixels = np.stack([np.arange(512, dtype=np.uint8).reshape(16, 32), np.full((16, 32), 255, np.uint8)])
    logits = infer(pixels[:, None], params)
    try:
        add(1, response={"loaded": False, "fingerprint": 0}, leds=0)
        add(0x10, eye_payload(method, *pixels), response=b"\x02", leds=0)
        add(2, bytes([METHOD_IDS[method]]) + struct.pack("<I", fingerprint) + image, response=b"\0", leds=0)
        add(1, response={"loaded": True, "fingerprint": fingerprint})
        add(5, struct.pack("<H", 0), response=b"\0" + image[:16])
        add(3, struct.pack("<ii", 0, 0), response=b"\0")
        add(0x10, eye_payload(method, *pixels), response={"logits": logits.tolist()})
        add(0x11, response=b"\0", leds=0)
        for method in ("grayscale2", "sobel2", "aeee2"):
            codes = np.stack([METHODS[method].encode(eye) for eye in pixels])
            mapped = codes * np.uint8(85)
            expected_logits = infer(mapped[:, None], params)
            add(2, bytes([METHOD_IDS[method]]) + struct.pack("<I", fingerprint) + image, response=b"\0", leds=0)
            add(0x10, eye_payload(method, *codes), response={"method_id": METHOD_IDS[method], "logits": expected_logits.tolist()})
        timed_payload = eye_payload(method, *codes)
        timed_payload = timed_payload[:1] + struct.pack("<I", 100) + timed_payload[1:]
        add(0x12, timed_payload, response={"method_id": METHOD_IDS[method], "logits": expected_logits.tolist(), "timed": True})
        add(1, corrupt=True)
        add(1, response={"loaded": True, "fingerprint": fingerprint, "method_id": METHOD_IDS[method]})
        add(0x13, response=b"\0")
        timer = BinocularTimer()
        observations = [(0, 0, 0), (33, 1, 1), (66, 0, 0), (99, 1, 0), (132, 0, 0),
                        (165, 1, 1), (365, 1, 1), (565, 1, 1), (765, 1, 1), (865, 0, 0),
                        (900, None, 1), (933, 0, 0)]
        for ms, left, right in observations:
            timing = timer.update(left, right, ms)
            valid = int(left is not None) | (int(right is not None) << 1)
            payload = struct.pack("<IBHH", ms, valid, int((left or 0)*65535), int((right or 0)*65535))
            closed, known, event, *words = astuple(timing)
            reply = struct.pack("<4B8I", 0, 1, closed, known | (int(event) << 7), *words)
            leds = ((closed & 1) | ((closed & 2) << 2)) if known == 3 else 0
            add(0x14, payload, response=reply, leds=leds)
        write_hex(output / "requests.hex", requests)
        write_hex(output / "request_sizes.hex", sizes, 4)
        write_hex(output / "reply_expected.hex", reply_flags)
        write_hex(output / "led_expected.hex", led_values)
        (output / "system.vh").write_text(f"`define REQUESTS {len(sizes)}\n`define REQUEST_BYTES {len(requests)}\n`define SYSTEM_PE_COUNT {pe_count}\n`define SYSTEM_PRETRAINED 0\n")
        comparison = baseline_sources(output, baseline_ref)
        binary = ROOT / "RESULTS/tools/iverilog/ucrt64/bin"
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
        command = [str(binary / "iverilog.exe"), "-g2001", "-s", "tb_system", "-o", "system.vvp", *comparison,
                   *(str(p.relative_to(output)) for p in sorted((output / "rtl").glob("*.v"))), "tb_system.v"]
        with (output / "compile.log").open("w") as log:
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        with (output / "simulation.log").open("w") as log:
            subprocess.run([str(binary / "vvp.exe"), "system.vvp"], cwd=output, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600 if baseline_ref else 180)
        log = (output / "simulation.log").read_text()
        if "SYSTEM_PASS" not in log or "FAIL" in log:
            raise ValueError("Pin-level integration failed; see simulation.log")
        if baseline_ref is not None:
            if "EQUIVALENCE_PASS" not in log:
                raise ValueError("Baseline comparison did not complete")
            result["baseline_ref"] = baseline_ref
            result["cycle_equivalence"] = "PASS"
        raw = bytes(int(line, 16) for line in (output / "replies.hex").read_text().splitlines())
        reader = PacketReader()
        replies = reader.feed(raw)
        if reader.bad_packets or reader.pending or len(replies) != len(expected):
            raise ValueError("Replies failed independent host CRC/framing validation")
        for (kind, sequence, payload), (want_kind, want_sequence, want) in zip(replies, expected):
            if (kind, sequence) != (want_kind, want_sequence):
                raise ValueError("Wrong reply kind or sequence")
            if isinstance(want, bytes):
                if payload != want:
                    raise ValueError(f"Reply payload mismatch for sequence {sequence}")
            elif "loaded" in want:
                if len(payload) != 12 or payload[0] != 0 or payload[1] != want.get("method_id", 0) or payload[4:8] != b"AEEE":
                    raise ValueError("Bad FPGA ping signature")
                if bool(payload[2]) != want["loaded"] or struct.unpack_from("<I", payload, 8)[0] != want["fingerprint"]:
                    raise ValueError("FPGA boot/model activation state mismatch")
            else:
                if len(payload) != (40 if want.get("timed") else 32) or payload[0] != 0 or payload[1] != want.get("method_id", 0):
                    raise ValueError("Bad CNN reply length/status")
                actual = struct.unpack_from("<4i", payload, 4)
                if tuple(np.array(want["logits"]).ravel()) != actual:
                    raise ValueError("CNN UART logits differ from the integer reference")
                decisions = int(actual[1] > actual[0]) | (int(actual[3] > actual[2]) << 1)
                if payload[2] != decisions | (decisions << 2):
                    raise ValueError("CNN raw/configured decision flags differ from logits")
                if want.get("timed") and struct.unpack_from("<2I", payload, 32) != (0, 0):
                    raise ValueError("First timestamped observation must not create a legacy blink")
                left_cycles, right_cycles, pair_cycles = struct.unpack_from("<3I", payload, 20)
                if min(left_cycles, right_cycles) <= 0 or pair_cycles < left_cycles + right_cycles:
                    raise ValueError("Invalid hardware cycle counters")
                cycles = {"left": left_cycles, "right": right_cycles, "pair": pair_cycles}
                result.setdefault("cnn_cycles", cycles)
                method_name = next(name for name, value in METHOD_IDS.items() if value == payload[1])
                result.setdefault("cnn_cycles_by_method", {})[method_name] = cycles
        result.update(status="PASS", requests=len(sizes), crc_checked_replies=len(replies),
                      bilateral_cycles=timer.count, long_closures=timer.long_count,
                      rtl_sha256={p.name: sha256(p) for p in sorted((output / "rtl").glob("*")) if p.is_file()})
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
