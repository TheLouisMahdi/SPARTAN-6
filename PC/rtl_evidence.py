"""Isolated synthetic RTL evidence; never changes the shipped model or bitstream."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from time import time_ns

import numpy as np
import integer_model as im
from encoders import METHODS
from fpga_host import model_image
from data_tools import sha256
from capture import ROOT


def write_hex(path, values, width=2):
    """Write flat integer bytes/words for Verilog $readmemh; new files only."""
    with Path(path).open("x", encoding="ascii") as stream:
        stream.writelines(f"{int(value) & ((1 << (4 * width)) - 1):0{width}x}\n" for value in values)


def baseline_sources(output, baseline_ref):
    """Compile a preserved Git top beside current RTL for cycle-level comparison."""
    if baseline_ref is None:
        return []
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", baseline_ref + "^{commit}"], cwd=ROOT, text=True).strip()
    try:
        source = subprocess.check_output(["git", "show", revision + ":FPGA/rtl/eye_top.v"], cwd=ROOT).decode()
    except subprocess.CalledProcessError:
        # Older commits used a lowercase directory name on Windows.
        source = subprocess.check_output(["git", "show", revision + ":fpga/rtl/eye_top.v"], cwd=ROOT).decode()
    source, count = re.subn(r"\bmodule eye_top\b", "module eye_top_baseline", source)
    if count != 1:
        raise ValueError("Baseline must contain exactly one eye_top module")
    (output / "eye_top_baseline.v").write_text(source, encoding="ascii")
    shutil.copy2(ROOT / "RESEARCH/top_equivalence.vh", output)
    return ["-DCHECK_BASELINE", "eye_top_baseline.v"]


def synthetic_parameters(seed=611, mode="mixed"):
    """Return legal TinyEye INT8 parameters, synthetic and unrelated to any person."""
    rng = np.random.default_rng(seed)
    params = {}
    for name, shape in zip(im.LAYERS, ((4, 1, 3, 3), (8, 4, 3, 3), (8, 96), (2, 8))):
        params[name + "_weight"] = rng.integers(-20, 21, shape, dtype=np.int8)
        params[name + "_bias"] = rng.integers(-1000, 2000, shape[0], dtype=np.int32)
        if mode != "mixed":
            params[name + "_bias"][:] = 100000 if mode == "saturate" else -1000000
        if name != "fc2":
            params[name + "_multiplier"] = np.array(1 << (24 if mode == "saturate" else 18), np.int64)
    return params


def layer_values(pixels, params):
    """uint8 N,1,16,32 -> exact stored magnitudes at both pools and FC1, then INT32 logits."""
    x = (pixels.astype(np.int16) - 128).astype(np.int8)
    result = []
    for name in ("conv1", "conv2"):
        x = im.pool(im.requantize(im.conv(x, params[name + "_weight"], params[name + "_bias"]),
                                  params[name + "_multiplier"]))
        result.append((x.astype(np.int16) + 128).astype(np.uint8))
    x = im.requantize(im.linear(x.reshape(len(x), -1), params["fc1_weight"], params["fc1_bias"]),
                      params["fc1_multiplier"])
    result.append((x.astype(np.int16) + 128).astype(np.uint8))
    result.append(im.linear(x, params["fc2_weight"], params["fc2_bias"]))
    return result


def goldens(output, public_run=None, public_data=None, bundle=None, seed=42):
    """Create a new deterministic arithmetic suite; contains no human images or accuracy results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(611)
    yy, xx = np.indices((16, 32))
    rois = [np.zeros((16, 32), np.uint8), np.full((16, 32), 255, np.uint8),
            (xx * 255 // 31).astype(np.uint8), (yy * 255 // 15).astype(np.uint8),
            (((xx + yy) % 2) * 255).astype(np.uint8), rng.integers(0, 256, (16, 32), np.uint8)]
    names = ("zeros", "max", "horizontal", "vertical", "checker", "seed611")
    inputs, models, expected, cases = [], [], [[] for _ in range(4)], []
    for mode in ("mixed", "saturate", "relu_zero"):
        params = synthetic_parameters(mode=mode)
        for method, encoder in METHODS.items():
            for name, roi in zip(names, rois):
                codes = encoder.encode(roi)
                pixels = codes if method == "grayscale8" else codes * np.uint8(85)
                layers = layer_values(pixels[None, None], params)
                np.testing.assert_array_equal(layers[-1], im.infer(pixels[None, None], params))
                inputs.extend(pixels.ravel())
                models.extend(model_image(params))
                for values, layer in zip(expected, layers):
                    values.extend(layer.ravel())
                cases.append({"method": method, "input": name, "model": mode})
    if any(value is not None for value in (public_run, public_data, bundle)):
        if not all(value is not None for value in (public_run, public_data, bundle)):
            raise ValueError("Public vectors require --public-run, --public-data and --bundle together")
        from evidence import public_splits
        from app import EyePredictor
        from data_tools import load_partitions
        from dataset import read_roi
        public_run, public_data = Path(public_run), Path(public_data)
        config = json.loads((public_run / "config.json").read_text())
        if not config.get("public_only") or not (public_run / "selection_frozen.json").is_file():
            raise ValueError("Public golden models must be selected on frozen public TRAIN/VAL, never personal")
        rows = load_partitions(public_data, public_splits(public_data))["test"]
        for method, encoder in METHODS.items():
            predictor = EyePredictor(method, seed, public_run, bundle)
            for row in sorted(rows, key=lambda item: item["sample_id"])[:8]:
                codes = encoder.encode(read_roi(public_data / row["roi_path"]))
                pixels = codes if method == "grayscale8" else codes * np.uint8(85)
                layers = layer_values(pixels[None, None], predictor.params)
                inputs.extend(pixels.ravel())
                models.extend(model_image(predictor.params))
                for values, layer in zip(expected, layers):
                    values.extend(layer.ravel())
                cases.append(dict(method=method, input=row["sample_id"], model=predictor.checkpoint_sha256,
                                  scope="authorized public arithmetic vector, not an accuracy estimate"))
    write_hex(output / "inputs.hex", inputs)
    write_hex(output / "models.hex", models)
    for name, values in zip(("pool1", "pool2", "fc1", "logits"), expected):
        write_hex(output / (name + ".hex"), values, 8 if name == "logits" else 2)
    (output / "generated").mkdir()
    write_hex(output / "generated/model.hex", models[:1208] + [0] * (2048 - 1208))
    (output / "suite.vh").write_text(f"`define CASES {len(cases)}\n", encoding="ascii")
    info = {"scope": "arithmetic fixtures, NOT public accuracy evidence; public cases explicitly tagged", "seed": 611,
            "cases": cases, "files": {p.name: sha256(p) for p in output.glob("*.hex")},
            "source_sha256": {n: sha256(ROOT / "PC" / n) for n in
                              ("integer_model.py", "rtl_evidence.py", "fpga_host.py")}}
    (output / "goldens.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def simulate(output, ise=Path(r"C:\Xilinx\14.7\ISE_DS"), simulator="iverilog",
             public_run=None, public_data=None, bundle=None, seed=42):
    """Run the real ISE ISim compiler and self-checking P1/P2 bench in a new directory."""
    output = Path(output).resolve()
    goldens(output, public_run, public_data, bundle, seed)
    rtl = ROOT / "FPGA/rtl"
    names = ("mac_int8", "requantize_q24", "cnn_addresses", "activation_store", "model_store", "eye_cnn", "eye_cnn_parallel")
    for name in names:
        shutil.copy2(rtl / (name + ".v"), output)
    shutil.copy2(ROOT / "RESEARCH/tb_cnn.v", output)
    (output / "bench.prj").write_text("".join(f'verilog work "{n}.v"\n' for n in (*names, "tb_cnn")), encoding="ascii")
    (output / "run.tcl").write_text("run all\nquit\n", encoding="ascii")
    result = {"status": "NOT_RUN_MISSING_DEPENDENCY", "tool": simulator, "output": str(output)}
    settings = ise / "settings64.bat"
    command = None
    if simulator == "iverilog":
        local = ROOT / "RESULTS/tools/iverilog/ucrt64/bin/iverilog.exe"
        compiler = str(local) if local.is_file() else shutil.which("iverilog")
        if compiler:
            binary = Path(compiler).parent
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"])
            command = [compiler, "-g2001", "-s", "tb_cnn", "-o", "bench.vvp", *(n + ".v" for n in (*names, "tb_cnn"))]
            with (output / "compile.log").open("w") as log:
                compiled = subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            result["compile_command"] = command
            command = [str(binary / ("vvp.exe" if os.name == "nt" else "vvp")), "bench.vvp"] if compiled.returncode == 0 else None
            result["compile_returncode"] = compiled.returncode
            if compiled.returncode:
                result["status"] = "FAIL"
    elif settings.is_file():
        script = f'@echo off\r\ncall "{settings}"\r\nfuse -prj bench.prj -o bench.exe work.tb_cnn > compile.log 2>&1\r\nif errorlevel 1 exit /b 1\r\nbench.exe -tclbatch run.tcl > simulation.log 2>&1\r\nexit /b %errorlevel%\r\n'
        (output / "simulate.cmd").write_text(script, encoding="ascii")
        command = [os.environ["COMSPEC"], "/d", "/c", str(output / "simulate.cmd")]
        env = None
    if command:
        result["command"] = command
        try:
            if simulator == "iverilog":
                with (output / "simulation.log").open("w") as log:
                    process = subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            else:
                process = subprocess.run(command, cwd=output, timeout=180)
        except subprocess.TimeoutExpired:
            result.update(status="FAIL_TIMEOUT", reason="Simulator exceeded its bounded runtime; see compile/simulation logs")
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        log = (output / "simulation.log").read_text(errors="replace") if (output / "simulation.log").exists() else ""
        result.update(status="PASS" if process.returncode == 0 and "SUITE_PASS" in log and "FAIL" not in log else "FAIL",
                      returncode=process.returncode)
        import re
        result["cases"] = [{"case": int(n), "p1_cycles": int(a), "p2_cycles": int(b),
                            "concurrent_mac_cycles": int(c)} for n, a, b, c in
                           re.findall(r"CASE_PASS (\d+) P1=(\d+) P2=(\d+) CONCURRENT=(\d+)", log)]
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_variant(output, pe_count, ise=Path(r"C:\Xilinx\14.7\ISE_DS"), network="tiny", baud=None):
    """Build P1/P2 with ISE in a fresh directory, never install/copy over the active BIT."""
    if pe_count not in (1, 2):
        raise ValueError("Only proven architectures P1/P2 are selectable; P4 is not implemented")
    if network not in ("tiny", "pretrained"):
        raise ValueError("Unknown CNN architecture")
    if network == "pretrained":
        _check_pretrained_weights()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("rtl", "generated"):
        shutil.copytree(ROOT / "FPGA" / folder, output / folder)
    shutil.copy2(ROOT / "FPGA/board_pins.ucf", output)
    # Specialize only the isolated source copy, independent of the current default.
    top = output / "rtl/eye_top.v"
    text, replacements = re.subn(r"parameter integer PE_COUNT=\d+", f"parameter integer PE_COUNT={pe_count}", top.read_text())
    if replacements != 1:
        raise ValueError("Expected exactly one PE_COUNT parameter in eye_top.v")
    text = re.sub(r"parameter integer PRETRAINED=\d+", f"parameter integer PRETRAINED={int(network == 'pretrained')}", text)
    text = re.sub(r"integer BAUD=\d+", f"integer BAUD={baud or (1000000 if network == 'pretrained' else 115200)}", text)
    top.write_text(text)
    sources = " ".join(p.stem for p in sorted((output / "rtl").glob("*.v")))
    tcl = f'''project new eye_top.xise
project set family Spartan6
project set device xc6slx9
project set package ftg256
project set speed -2
project set synthesis_tool {{XST (VHDL/Verilog)}}
foreach source {{{sources}}} {{xfile add rtl/$source.v}}
xfile add board_pins.ucf
project set top eye_top
project set {{Unused IOB Pins}} Float -process {{Generate Programming File}}
project set {{FPGA Start-Up Clock}} {{JTAG Clock}} -process {{Generate Programming File}}
project set {{UserID Code (8 Digit Hexadecimal)}} 41454545 -process {{Generate Programming File}}
foreach step {{{{Synthesize - XST}} {{Implement Design}} {{Generate Programming File}}}} {{
    if {{![process run $step]}} {{project close; error "Failed $step"}}
}}
project close
'''
    (output / "build.tcl").write_text(tcl, encoding="ascii")
    return _build_project(output, pe_count, ise, network, baud)


def _check_pretrained_weights():
    from pretrained_eye import PretrainedEye, DIRECTORY
    PretrainedEye()
    record = json.loads((DIRECTORY / "deployment.json").read_text())
    for name, digest in record["rtl_files_sha256"].items():
        if sha256(ROOT / "FPGA/generated" / name) != digest:
            raise ValueError("Generated pretrained weights changed; run pretrained prepare again")


def build_canonical(ise=Path(r"C:\Xilinx\14.7\ISE_DS")):
    """Cleanly implement the official pretrained P2 project in FPGA/."""
    top = (ROOT / "FPGA/rtl/eye_top.v").read_text()
    for name, value in {"PRETRAINED": 1, "PE_COUNT": 2, "CLOCK_HZ": 50000000, "BAUD": 1000000}.items():
        match = re.search(r"parameter integer " + name + r"=(\d+)", top)
        if not match or int(match[1]) != value:
            raise ValueError(f"Canonical project requires {name}={value}; use --output for research variants")
    _check_pretrained_weights()
    return _build_project(ROOT / "FPGA", 2, ise, "pretrained", 1000000, canonical=True)


def _build_project(output, pe_count, ise, network, baud, canonical=False):
    settings = Path(ise) / "settings64.bat"
    result = {"status": "NOT_RUN_MISSING_DEPENDENCY", "pe_count": pe_count, "tool": "ISE 14.7",
              "network": network, "baud": baud or (1000000 if network == "pretrained" else 115200),
              "clock_hz": 50000000, "board_test": "NOT_RUN_NO_BOARD",
              "scope": "canonical artifacts rebuilt; physical FPGA not programmed" if canonical else "implementation only; active bitstream unchanged"}
    result["source_sha256"] = {p.relative_to(ROOT / "FPGA").as_posix(): sha256(p)
        for folder in ("rtl", "generated") for p in sorted((ROOT / "FPGA" / folder).glob("*")) if p.is_file()}
    result["source_sha256"]["board_pins.ucf"] = sha256(ROOT / "FPGA/board_pins.ucf")
    if settings.is_file():
        if not canonical:
            (output / "build.cmd").write_text(f'@echo off\r\ncall "{settings}"\r\nxtclsh build.tcl > build.log 2>&1\r\nexit /b %errorlevel%\r\n', encoding="ascii")
        command = [os.environ["COMSPEC"], "/d", "/c", str(output / "build.cmd")]
        result["command"] = command
        env = dict(os.environ, ISE_SETTINGS=str(settings))
        process = subprocess.run(command, cwd=output, env=env, timeout=1800)
        result["returncode"] = process.returncode
        report = output / "eye_top_map.mrp"
        if not report.exists():
            report = output / "eye_top.mrp"
        mapped = report.read_text(errors="replace") if report.exists() else ""
        for field, pattern in {"LUT": r"Number of Slice LUTs:\s*([\d,]+)",
                               "FF": r"Number of Slice Registers:\s*([\d,]+)",
                               "DSP48A1": r"Number of DSP48A1s:\s*([\d,]+)",
                               "BRAM18": r"Number of RAMB16BWERs:\s*([\d,]+)",
                               "BRAM9": r"Number of RAMB8BWERs:\s*([\d,]+)",
                               "slices": r"Number of occupied Slices:\s*([\d,]+)"}.items():
            match = re.search(pattern, mapped)
            result[field] = int(match[1].replace(",", "")) if match else None
        timing_path = output / "eye_top.twr"
        timing = timing_path.read_text(errors="replace") if timing_path.exists() else ""
        period = re.search(r"Minimum period is\s*([\d.]+)ns", timing)
        errors = re.search(r"(\d+) timing errors detected", timing)
        result.update(Fmax_MHz=1000 / float(period[1]) if period else None,
                      timing_met=int(errors[1]) == 0 if errors else None,
                      reports={p.name: sha256(p) for p in output.glob("*") if p.suffix in (".mrp", ".twr", ".syr")})
        log = "\n".join((output / name).read_text(errors="replace") for name in
                        ("synthesize.log", "build.log", "generate_svf.log") if (output / name).is_file())
        result["status"] = "PASS" if process.returncode == 0 and (output / "eye_top.bit").exists() and result["timing_met"] else (
            "DOES_NOT_FIT" if re.search(r"overmapped|too many|exceeds.*available", log + mapped, re.I) else "FAIL")
        if (output / "eye_top.bit").is_file():
            result["bit_sha256"] = sha256(output / "eye_top.bit")
    if canonical:
        result["source_sha256"].update({name: sha256(output / name) for name in
            ("eye_top.xise", "create_project.tcl", "build.tcl", "build.cmd", "generate_svf.cmd")})
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def board_suite(output, port):
    """Synthetic UART arithmetic test on the currently programmed core; restore its model bytes.

    No webcam, personal checkpoint, threshold change or invented board result.
    CRC/sequence validation is the existing FPGAClient contract. Trace contains
    synthetic test traffic only; the prior model is held in RAM and not logged.
    """
    from fpga_host import FPGAClient, eye_payload, METHOD_IDS
    from ocular_metrics import percentile
    import hashlib
    import struct
    import time
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "NOT_RUN_NO_BOARD", "port": port, "clock_hz": 50000000,
              "scope": "synthetic arithmetic/transport evidence, not accuracy; UART does not attest PE_COUNT"}
    client = None
    saved = None
    records, trace = [], []
    try:
        client = FPGAClient(port)
        status = client.ping()
        saved = (status, client.read_model())
    except (OSError, TimeoutError, RuntimeError) as exc:
        result["reason"] = str(exc)
    else:
        class TracePort:
            def __init__(self, serial):
                self.serial = serial
            def __getattr__(self, name):
                return getattr(self.serial, name)
            def write(self, data):
                trace.append(dict(direction="tx", monotonic_ns=time.perf_counter_ns(), hex=data.hex()))
                return self.serial.write(data)
            def read(self, size):
                data = self.serial.read(size)
                if data:
                    trace.append(dict(direction="rx", monotonic_ns=time.perf_counter_ns(), hex=data.hex()))
                return data
        original_port = client.serial
        client.serial = TracePort(original_port)
        try:
            yy, xx = np.indices((16, 32))
            rois = [np.zeros((16, 32), np.uint8), np.full((16, 32), 255, np.uint8),
                    (xx * 255 // 31).astype(np.uint8), (yy * 255 // 15).astype(np.uint8),
                    (((xx + yy) % 2) * 255).astype(np.uint8),
                    np.random.default_rng(611).integers(0, 256, (16, 32), np.uint8)]
            for mode in ("mixed", "saturate", "relu_zero"):
                params = synthetic_parameters(mode=mode)
                image = model_image(params)
                fingerprint = int(hashlib.sha256(image).hexdigest()[:8], 16)
                for method, encoder in METHODS.items():
                    client.request(2, bytes([METHOD_IDS[method]]) + struct.pack("<I", fingerprint) + image)
                    ping = client.ping()
                    if ping["method_id"] != METHOD_IDS[method] or ping["fingerprint"] != fingerprint or client.read_model() != image:
                        raise RuntimeError("Synthetic model method/fingerprint/full readback mismatch")
                    for index, roi in enumerate(rois):
                        left, right = encoder.encode(roi), encoder.encode(np.ascontiguousarray(roi[:, ::-1]))
                        batch = np.stack([left, right])[:, None]
                        if method != "grayscale8":
                            batch = batch * np.uint8(85)
                        layers = layer_values(batch, params)
                        frame_id = 10000 + len(records)
                        reply = client.infer(method, left, right, frame_id, capture_ms=1000 + len(records) * 100)
                        actual = np.array([reply.left_logits, reply.right_logits], np.int32)
                        if not np.array_equal(actual, layers[-1]):
                            raise RuntimeError(f"Board logits mismatch: {mode}/{method}/input{index}; expected {layers[-1].tolist()}, got {actual.tolist()}")
                        raw_flags = int(actual[0, 1] > actual[0, 0]) | (int(actual[1, 1] > actual[1, 0]) << 1)
                        if reply.flags & 3 != raw_flags:
                            raise RuntimeError("Board raw eye-state flags differ from logits")
                        if client.read_memory(1536, 96) != layers[1][1].tobytes() or client.read_memory(1280, 8) != layers[2][1].tobytes():
                            raise RuntimeError("Board retained pool2/FC1 activations differ")
                        request_bytes = 12 + len(eye_payload(method, left, right)) + 4
                        records.append(dict(case=len(records), model=mode, method=method, input=index,
                            checkpoint_sha256=hashlib.sha256(image).hexdigest(), exact_logits=True,
                            left_logits=reply.left_logits, right_logits=reply.right_logits,
                            left_cycles=reply.left_cycles, right_cycles=reply.right_cycles,
                            left_compute_ms=reply.left_cycles / 50000, right_compute_ms=reply.right_cycles / 50000,
                            pair_cycles=reply.pair_cycles, roundtrip_ms=reply.roundtrip_ms,
                            request_bytes=request_bytes, response_bytes=52,
                            wire_floor_ms=(request_bytes + 52) * 10 / 115200 * 1000))
            result.update(status="PASS", pairs=len(records), bad_packets=client.reader.bad_packets,
                          p50_pair_roundtrip_ms=percentile([r["roundtrip_ms"] for r in records], 0.5),
                          p95_pair_roundtrip_ms=percentile([r["roundtrip_ms"] for r in records], 0.95))
        except Exception as exc:
            result.update(status="FAIL", reason=str(exc))
        finally:
            client.serial = original_port
            try:
                previous, image = saved
                client.request(2, bytes([previous["method_id"]]) + struct.pack("<I", previous["fingerprint"]) + image)
                restored = client.ping()
                if restored["fingerprint"] != previous["fingerprint"] or restored["method_id"] != previous["method_id"] or client.read_model() != image:
                    raise RuntimeError("Prior model restoration did not verify")
                client.invalidate(0)
                result["prior_model_restored"] = True
            except Exception as exc:
                result.update(status="FAIL", restoration_error=str(exc))
            with (output / "uart_trace.jsonl").open("x") as stream:
                for item in trace:
                    stream.write(json.dumps(item) + "\n")
    finally:
        if client is not None:
            client.close()
    (output / "cases.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def program_svf(svf, output, expected_usercode=0x41454545):
    """Play an ISE-generated SVF using the retained Xilinx cable transport; never synthesize with Vivado."""
    import socket
    import time
    if not isinstance(expected_usercode, int) or not 0 <= expected_usercode <= 0xffffffff:
        raise ValueError("Expected USERCODE must be an explicit uint32")
    usercode_label = "AEEE" if expected_usercode == 0x41454545 else f"{expected_usercode:08X}"
    svf, output = Path(svf).resolve(), Path(output).resolve()
    if not svf.is_file():
        raise FileNotFoundError(svf)
    output.mkdir(parents=True, exist_ok=False)
    driver = Path(r"C:\Xilinx\Vivado\2023.2\bin")
    # This is the already installed JTAG driver/XSDB transport, not a Spartan-6
    # synthesis backend. All BIT/SVF generation is ISE 14.7.
    try:
        with socket.create_connection(("127.0.0.1", 3121), timeout=1):
            pass
    except OSError:
        with (output / "hw_server.log").open("wb") as log:
            subprocess.Popen([os.environ["COMSPEC"], "/d", "/c", str(driver / "hw_server.bat"), "-s", "tcp::3121"],
                stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
        deadline = time.monotonic() + 10
        while True:
            try:
                with socket.create_connection(("127.0.0.1", 3121), timeout=1):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Existing Xilinx JTAG transport did not start")
                time.sleep(0.2)
    player = (ROOT / "FPGA/play_ise_svf.tcl").as_posix()
    script = f'''proc main {{}} {{
connect -url tcp:127.0.0.1:3121
jtag targets -set -filter {{name == "xc6slx9"}} -timeout 5
set devices {{}}
foreach target [jtag targets -target-properties] {{
    if {{[dict exists $target is_fpga] && [dict get $target is_fpga]}} {{lappend devices $target}}
}}
if {{[llength $devices]!=1 || [dict get [lindex $devices 0] name] ne "xc6slx9" || [dict get [lindex $devices 0] idcode] ne "24001093" || [dict get [lindex $devices 0] irlen]!=6}} {{error {{Expected one verified XC6SLX9}}}}
jtag targets -set -filter {{name == "xc6slx9"}}
jtag lock 10000
source {{{player}}}
play_ise_svf {{{svf.as_posix()}}}
set sequence [jtag sequence]
$sequence state RESET
$sequence irshift -state IDLE -capture -integer 6 8
$sequence drshift -state IDLE -capture -tdi 0 32
set result [$sequence run -integer]
$sequence delete
if {{([lindex $result 0]&32)==0 || [lindex $result 1]!=0x{expected_usercode:08X}}} {{error {{DONE/USERCODE verification failed}}}}
jtag unlock
puts {{PROGRAM_PASS DONE USERCODE_{usercode_label}}}
}}
if {{[catch {{main}} message]}} {{catch {{jtag unlock}};puts stderr [lindex [split $message "\\n"] 0];exit 1}}
exit 0
'''
    (output / "program.tcl").write_text(script, encoding="ascii")
    command = [os.environ["COMSPEC"], "/d", "/c", str(driver / "xsdb.bat"), "program.tcl"]
    with (output / "program.log").open("w") as log:
        process = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    log = (output / "program.log").read_text(errors="replace")
    result = {"status": "PASS" if process.returncode == 0 and "PROGRAM_PASS" in log else "FAIL",
              "svf_sha256": sha256(svf), "expected_usercode": expected_usercode,
              "command": command, "returncode": process.returncode}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def program_variant(output):
    """Generate an SVF from a successful isolated ISE build and explicitly program it."""
    output = Path(output).resolve()
    result = json.loads((output / "result.json").read_text())
    if result["status"] != "PASS" or not result["timing_met"]:
        raise ValueError("Programming requires an observed successful ISE build and timing result")
    if result.get("bit_sha256") != sha256(output / "eye_top.bit"):
        raise ValueError("Build BIT hash is missing or changed; rebuild before programming")
    sources = result.get("source_sha256", {})
    if not sources or any(sha256(ROOT / "FPGA" / name) != value for name, value in sources.items()):
        raise ValueError("FPGA sources changed since this build; rebuild before programming")
    script = output / "svf.cmd"
    script.write_text('setMode -bscan\nsetCable -port svf -file eye_top.svf\naddDevice -p 1 -file eye_top.bit\nprogram -p 1\nquit\n')
    (output / "svf_generate.cmd").write_text('@echo off\r\ncall C:\\Xilinx\\14.7\\ISE_DS\\settings64.bat\r\nimpact -batch svf.cmd > svf.log 2>&1\r\nexit /b %errorlevel%\r\n')
    process = subprocess.run([os.environ["COMSPEC"], "/d", "/c", str(output / "svf_generate.cmd")], cwd=output, timeout=120)
    if process.returncode or not (output / "eye_top.svf").is_file():
        raise RuntimeError("ISE iMPACT SVF generation failed; see svf.log")
    return program_svf(output / "eye_top.svf", output / ("jtag_program_" + str(time_ns())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("golden", "simulate", "system", "build", "board", "program"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pe-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--simulator", choices=("iverilog", "isim"), default="iverilog")
    parser.add_argument("--baseline-ref", help="Pre-extraction Git revision for cycle-level system comparison")
    parser.add_argument("--port", help="UART port for the board test")
    parser.add_argument("--public-run", type=Path)
    parser.add_argument("--public-data", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44, 45, 46), default=42)
    args = parser.parse_args(argv)
    if args.action == "board" and not (args.port and args.port.strip()):
        parser.error("--port is required for the board test")
    if args.action == "system":
        from system_sim import simulate_system
        result = simulate_system(args.output, args.pe_count, baseline_ref=args.baseline_ref)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    result = program_variant(args.output) if args.action == "program" else board_suite(args.output, args.port) if args.action == "board" else build_variant(args.output, args.pe_count) if args.action == "build" else (
        simulate(args.output, simulator=args.simulator, public_run=args.public_run,
                 public_data=args.public_data, bundle=args.bundle, seed=args.seed) if args.action == "simulate" else
        goldens(args.output, args.public_run, args.public_data, args.bundle, args.seed))
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
