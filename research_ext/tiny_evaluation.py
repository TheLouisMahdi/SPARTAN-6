"""Isolated compact USER1 build preparation; no programming side effects."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
from research_ext.jtag_evaluation import prepare_cnn


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("source structure changed: " + old)
    return text.replace(old, new)


def prepare_tiny(output, dsp=False):
    prepare_cnn(output)
    top = output / "rtl/eye_top.v"
    text = top.read_text()
    for old, new in (
        ("parameter integer PRETRAINED=1", "parameter integer PRETRAINED=0"),
        ("wire uart_input_we,", "wire [7:0] transport_method;wire [31:0] transport_fingerprint;wire transport_model_loaded;\n    wire uart_input_we,"),
        ("jtag_cnn_port jtag_port(clk,jtag_available,", "jtag_tiny_port jtag_port(clk,jtag_available,transport_model_loaded,transport_method,transport_fingerprint,"),
        (".pretrained_upload_active(uart_upload_active));", ".pretrained_upload_active(uart_upload_active), .transport_method(transport_method), .transport_fingerprint(transport_fingerprint), .transport_model_loaded(transport_model_loaded));"),
    ):
        text = replace_once(text, old, new)
    top.write_text(text)
    controller = output / "rtl/command_controller.v"
    text = controller.read_text()
    text = replace_once(text, "output wire pretrained_upload_active);",
                        "output wire pretrained_upload_active, output wire [7:0] transport_method,output wire [31:0] transport_fingerprint,output wire transport_model_loaded);")
    text = replace_once(text, "reg model_loaded=PRETRAINED;", "reg model_loaded=PRETRAINED;\n    assign transport_method=active_method;assign transport_fingerprint=fingerprint;assign transport_model_loaded=model_loaded;")
    controller.write_text(text)
    tcl = output / "build.tcl"
    tcl.write_text(replace_once(tcl.read_text(), "4A544E31", "4A545931"))
    if dsp:
        core = output / "rtl/eye_cnn_parallel.v"
        core.write_text(core.read_text().replace("requantize_q24 ", "requantize_q24_dsp "))


def build_tiny(output, dsp=False):
    from rtl_evidence import _build_project
    import re
    prepare_tiny(output, dsp)
    result = _build_project(output, 2, Path(r"C:\Xilinx\14.7\ISE_DS"), "tiny", 1000000)
    timing = (output / "eye_top.twr").read_text() if (output / "eye_top.twr").exists() else ""
    errors = re.findall(r"(\d+) timing errors detected", timing)
    result["timing_met"] = bool(errors and all(int(x) == 0 for x in errors))
    if not result["timing_met"] and result["status"] == "PASS":
        result["status"] = "FAIL_TIMING"
    result.update(physical="NOT_MEASURED", requantization="dsp" if dsp else "serial", usercode="4A545931", scope="TinyEye P2 UART 1M / JTAG candidate",
                  source_sha256={p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for folder in ("rtl", "generated") for p in (output / folder).glob("*") if p.is_file()})
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


def simulate_tiny(output, checkpoint, dsp=False):
    """Test the real P2 core with frozen trained coefficients and packed inputs."""
    import binascii
    import os
    import struct
    import subprocess
    import numpy as np
    from fpga_host import eye_payload, model_image
    from integer_model import infer
    from live_pipeline import read_parameters
    from tiny_jtag import tiny_packet
    prepare_tiny(output, dsp)
    params = read_parameters(checkpoint)
    fingerprint = int(hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:8], 16)
    image = model_image(params)
    (output / "generated/model.hex").write_text("\n".join(f"{b:02x}" for b in image + bytes(2048-len(image)))+"\n")
    cases = [(0xfffffffe, 0), (0xffffffff, 0), (0xffffffff, 3),
             (0, 1), (0, 8), (0, 9), (0, 2), (0, 0), (1, 0)]
    vector = bytearray()
    rng = np.random.default_rng(417)
    for i, (sequence, status) in enumerate(cases):
        codes = rng.integers(0, 4, (16, 32), dtype=np.uint8)
        if i in (0, 7):
            codes[:] = 0 if i == 0 else 3
        codes[0, :4] = np.arange(4, dtype=np.uint8)
        packed = eye_payload("grayscale2", codes, codes)[1:129]
        frame, eye, capture = 77+i, i % 2, 123456789+i*20000000
        packet = bytearray(tiny_packet(sequence, frame, capture, eye, packed,
                                      fingerprint ^ (1 if status == 9 else 0)))
        if status == 2:
            packet[66] = 2
            packet[-3:-1] = struct.pack("<H", binascii.crc_hqx(packet[41:-3], 0xffff))
        if status == 1:
            packet[-2] ^= 1
        logits = infer((codes * np.uint8(85))[None, None], params)[0]
        vector.extend(struct.pack("<HBIIBQii", len(packet), status, sequence, frame, eye, capture, *map(int, logits))+packet)
    (output / "cnn_vectors.hex").write_text("\n".join(f"{b:02x}" for b in vector)+"\n")
    bench = (ROOT / "research_ext/tb_jtag_tiny.v").read_text().replace("32'h353bd6ca", f"32'h{fingerprint:08x}")
    (output / "tb_jtag_tiny.v").write_text(bench)
    binary = ROOT / "assets/tools/iverilog/ucrt64/bin"
    env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ["PATH"])
    sources = [f"rtl/{name}.v" for name in ("jtag_tiny_port", "jtag_async_fifo", "eye_cnn_parallel", "mac_int8", "requantize_q24", "cnn_addresses", "activation_store", "model_store")]
    if dsp:
        sources.append("rtl/requantize_q24_dsp.v")
    with (output / "simulation.log").open("w") as log:
        for command in ([str(binary/"iverilog.exe"), "-g2001", f"-DVECTOR_BYTES={len(vector)}", "-s", "tb_tiny_candidate", "-o", "tiny.vvp", *sources, "tb_jtag_tiny.v"],
                        [str(binary/"vvp.exe"), "tiny.vvp"]):
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    text = (output / "simulation.log").read_text()
    if "TINY_JTAG_SIM_PASS" not in text or "FAIL" in text:
        raise AssertionError(text[-1600:])
    from fpga_host import encode_packet
    upload = encode_packet(2, 1, bytes([1])+struct.pack("<I", fingerprint)+image)
    (output / "model_upload.hex").write_text("\n".join(f"{b:02x}" for b in upload)+"\n")
    with (output / "ownership.log").open("w") as log:
        for command in ([str(binary/"iverilog.exe"), "-g2001", f"-DMODEL_PACKET_BYTES={len(upload)}", "-s", "tb_tiny_ownership", "-o", "ownership.vvp",
                         *[p.relative_to(output).as_posix() for p in (output / "rtl").glob("*.v")], str(ROOT / "research_ext/tb_tiny_ownership.v")],
                        [str(binary/"vvp.exe"), "ownership.vvp"]):
            subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    ownership = (output / "ownership.log").read_text()
    if "TINY_OWNERSHIP_PASS" not in ownership or "FAIL" in ownership:
        raise AssertionError(ownership[-1600:])
    import csv
    import re
    rows = [dict(zip(("frame_id", "capture_ns", "eye", "open", "closed", "cycles"), map(int, match)))
            for match in re.findall(r"TINY_RESULT frame=(\d+) capture=(\d+) eye=(\d+) open=(-?\d+) closed=(-?\d+) cycles=(\d+)", text)]
    with (output / "ordered_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("frame_id", "capture_ns", "eye", "open", "closed", "cycles"))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="PASS", provenance="RTL_SIMULATION", requantization="dsp" if dsp else "serial", cases=len(cases), eyes=4,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  packed_symbols="all four codes checked on every input write", physical="NOT_MEASURED",
                  UART_model_ownership="PASS", ordered_results=rows)
    (output / "simulation.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "build", "simulate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dsp", action="store_true", help="Optional equivalent DSP multiplier; canonical core unchanged")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_tiny(args.output.resolve(), args.dsp)
    elif args.action == "build":
        print(json.dumps(build_tiny(args.output.resolve(), args.dsp), indent=2))
    else:
        if args.checkpoint is None:
            parser.error("simulate requires --checkpoint")
        print(json.dumps(simulate_tiny(args.output.resolve(), args.checkpoint.resolve(), args.dsp), indent=2))
