"""Offline ONNX, fixed-point and real RTL checks; never opens camera or board."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request

import cv2
import numpy as np

from pretrained_eye import ROOT, PretrainedEye, fixed_layers, input_pixels, network, write_hex, LAYER_OUTPUTS


def verify(output, pe_count=2, offline=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = PretrainedEye()
    rng = np.random.default_rng(417)
    rois = [np.full((32, 32, 3), v, np.uint8) for v in (0, 127, 255)]
    rois += [rng.integers(0, 256, (32, 32, 3), np.uint8)]
    names = ["black", "neutral", "white", "synthetic_seed417"]
    public = []
    for i in (() if offline else range(1, 4)):
        url = f"https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/public/open-closed-eye-0001/assets/{i}.png"
        data = urllib.request.urlopen(url, timeout=45).read()
        path = output / f"official_example_{i}.png"
        path.write_bytes(data)
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Official example is not a decodable image")
        rois.append(image)
        names.append(f"official_example_{i}")
        public.append({"file": path.name, "source": url, "sha256": hashlib.sha256(data).hexdigest()})
    inputs, expected, comparisons = [], [], []
    net = network(model.raw)
    for name, roi in zip(names, rois):
        pixels = input_pixels(roi)
        layers = fixed_layers(pixels, model.params, model.shifts)
        net.setInput(((pixels.astype(np.float32)-127)/255)[None])
        floating = net.forward(["onnx_node_output_0!"+node for node in LAYER_OUTPUTS])
        errors = [float(np.abs(fixed.astype(np.float64)/2.0**model.fractions[i+1] - fp.reshape(fixed.shape)).max())
                  for i, (fixed, fp) in enumerate(zip(layers, floating))]
        comparisons.append({"input": name, "layer_max_absolute_error": errors,
                            "fp32_logits": floating[-1].ravel().tolist(), "fixed_logits": layers[-1].tolist(),
                            "argmax_agrees": bool(np.argmax(floating[-1]) == np.argmax(layers[-1]))})
        inputs.extend(pixels.ravel())
        expected.extend(np.concatenate([layer.ravel() for layer in layers]))
    write_hex(output / "inputs.hex", inputs, 8)
    write_hex(output / "expected.hex", expected, 32)
    (output / "suite.vh").write_text(f"`define CASES {len(rois)}\n`define OMZ_PE_COUNT {pe_count}\n")
    for folder in ("rtl", "generated"):
        shutil.copytree(ROOT / "FPGA" / folder, output / folder)
    shutil.copy2(ROOT / "RESEARCH/tb_omz.v", output)
    binary = ROOT / "RESULTS/tools/iverilog/ucrt64/bin"
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"])
    commands = [[str(binary / "iverilog.exe"), "-g2001", "-s", "tb_omz", "-o", "omz.vvp", "rtl/omz_cnn.v", "tb_omz.v"],
                [str(binary / "vvp.exe"), "omz.vvp"]]
    for name, command in zip(("compile.log", "simulation.log"), commands):
        with (output / name).open("w") as log:
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    log = (output / "simulation.log").read_text()
    result = {"status": "PASS" if "OMZ_PASS" in log and "FAIL" not in log else "FAIL",
              "scope": "arithmetic equivalence; deterministic input fixtures, NOT an accuracy benchmark" if offline else "arithmetic equivalence; synthetic and official examples, NOT an accuracy benchmark",
              "pe_count": pe_count, "training_performed": False, "calibration_performed": False,
              "model_sha256": model.checkpoint_sha256, "official_examples": public,
              "comparisons": comparisons, "all_argmax_agree": all(c["argmax_agrees"] for c in comparisons),
              "cycles": [int(n) for n in re.findall(r"OMZ_CASE_PASS \d+ cycles=(\d+)", log)]}
    (output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def verify_uart(output, pe_count=2, baseline_ref=None):
    """Send actual serial bits through the entire top and decode replies on PC."""
    from dataclasses import astuple
    import struct
    from blink_timing import BinocularTimer
    from fpga_host import encode_packet, PacketReader

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("rtl", "generated"):
        shutil.copytree(ROOT / "FPGA" / folder, output / folder)
    bench = (ROOT / "RESEARCH/tb_system.v").read_text().replace("timeout<600000", "timeout<3000000")
    (output / "tb_system.v").write_text(bench)
    packets, sizes, expected, reply_flags, leds = [], [], [], [], []

    def add(kind, payload=b"", response=b"\0", led=0xff, corrupt=False):
        sequence = len(sizes)+1
        packet = bytearray(encode_packet(kind, sequence, payload))
        if corrupt:
            packet[-1] ^= 1
        packets.extend(packet)
        sizes.append(len(packet))
        reply_flags.append(not corrupt)
        leds.append(led)
        if not corrupt:
            expected.append((kind | 0x80, sequence, response))

    model = PretrainedEye(precision="fixed")
    add(1, response=struct.pack("<4B4sI", 0, 4, 1, 0xc0, b"AEEE", int(model.checkpoint_sha256[:8], 16)), led=0)
    add(0x21, response=b"\2")
    add(0x20, struct.pack("<H", 1024)+b"\0", response=b"\5")
    add(0x20, struct.pack("<H", 3000)+bytes(100), response=b"\5")
    add(1, corrupt=True)
    # Distinct color planes detect BGR/CHW swaps and chunk offset mistakes.
    for offset in (13, 179):
        roi = ((np.arange(3072, dtype=np.uint16).reshape(32, 32, 3)*17+offset) % 256).astype(np.uint8)
        data = input_pixels(roi).tobytes()
        for start in range(0, 3072, 1024):
            add(0x20, struct.pack("<H", start)+data[start:start+1024])
        # The packet carries native ROM channels; host inference reorders them.
        add(0x21, response={"logits": fixed_layers(input_pixels(roi), model.params, model.shifts)[-1].tolist()})
        add(0x21, response=b"\2")  # Cannot reuse the previous eye after inference.
    timer = BinocularTimer()
    for ms, left, right in ((0,0,0),(40,1,1),(100,0,0),(130,1,0),(190,0,0),(220,1,1),(420,1,1),(620,1,1),(820,0,0)):
        timing = timer.update(left, right, ms)
        closed, known, event, *words = astuple(timing)
        response = struct.pack("<4B8I", 0, 1, closed, known | (int(event)<<7), *words)
        add(0x14, struct.pack("<IBHH", ms, 3, left*65535, right*65535), response=response,
            led=(closed & 1) | ((closed & 2)<<2))
    for name, values, bits in (("requests",packets,8),("request_sizes",sizes,16),
                                ("reply_expected",reply_flags,8),("led_expected",leds,8)):
        write_hex(output / (name+".hex"), values, bits)
    (output / "system.vh").write_text(f"`define REQUESTS {len(sizes)}\n`define REQUEST_BYTES {len(packets)}\n`define SYSTEM_PE_COUNT {pe_count}\n`define SYSTEM_PRETRAINED 1\n")
    from rtl_evidence import baseline_sources
    comparison = baseline_sources(output, baseline_ref)
    binary = ROOT / "RESULTS/tools/iverilog/ucrt64/bin"
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"])
    commands = [[str(binary / "iverilog.exe"), "-g2001", "-s", "tb_system", "-o", "uart.vvp", *comparison,
                 *(str(p.relative_to(output)) for p in sorted((output / "rtl").glob("*.v"))), "tb_system.v"],
                [str(binary / "vvp.exe"), "uart.vvp"]]
    for name, command in zip(("compile.log", "simulation.log"), commands):
        with (output / name).open("w") as log:
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600 if baseline_ref else 300)
    log = (output / "simulation.log").read_text()
    if "SYSTEM_PASS" not in log or "FAIL" in log:
        raise ValueError("Pretrained UART simulation failed; see simulation.log")
    if baseline_ref is not None and "EQUIVALENCE_PASS" not in log:
        raise ValueError("Baseline comparison did not complete")
    reader = PacketReader()
    decoded = reader.feed(bytes(int(line,16) for line in (output / "replies.hex").read_text().splitlines()))
    if reader.bad_packets or reader.pending or len(decoded) != len(expected):
        raise ValueError("Reply count, CRC or framing failure")
    cycles = []
    for (kind, sequence, payload), (want_kind, want_sequence, want) in zip(decoded, expected):
        if (kind, sequence) != (want_kind, want_sequence):
            raise ValueError("Reply command/sequence mismatch")
        if isinstance(want, bytes):
            if payload != want:
                raise ValueError(f"Unexpected UART reply {kind:02x}: {payload.hex()}")
        else:
            if len(payload) != 17 or payload[0] != 0:
                raise ValueError("Invalid pretrained logit reply")
            opened, closed, elapsed, fingerprint = struct.unpack_from("<iiII", payload, 1)
            if [opened, closed] != want["logits"] or fingerprint != int(model.checkpoint_sha256[:8],16) or elapsed <= 0:
                raise ValueError("UART logits, cycle count or model identity mismatch")
            cycles.append(elapsed)
    result = {"status": "PASS", "scope": "pin-level UART simulation; serial bit timing accelerated; no physical board",
              "requests": len(sizes), "crc_checked_replies": len(decoded), "cnn_cycles": cycles,
              "bilateral_cycles": timer.count, "long_closures": timer.long_count,
              "led0_left_led3_right": "PASS", "six_digit_scan": "PASS", "model_sha256": model.checkpoint_sha256}
    if baseline_ref is not None:
        result.update(baseline_ref=baseline_ref, cycle_equivalence="PASS")
    (output / "result.json").write_text(json.dumps(result,indent=2)+"\n")
    return result
