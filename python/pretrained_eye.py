"""Pinned Open Model Zoo eye CNN: original FP32 and deterministic RTL arithmetic.

No fitting, fine tuning or data-dependent calibration is performed here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from threading import local
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "assets/models/open-closed-eye-0001"
URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/open-closed-eye-0001/open_closed_eye.onnx"
CHECKSUM = "2615bce53b55620c629db21b043057600ccc53466f053c0a8277c43577c2db21e48f330cf9b15213016d17cddb8cba27"
LICENSE_URL = "https://raw.githubusercontent.com/opencv/openvino_training_extensions/develop/LICENSE"
NODES = ("8", "11", "14", "15")
SHAPES = ((10, 3, 3, 3), (20, 10, 3, 3), (50, 20, 3, 3), (2, 50, 1, 1))
LAYER_OUTPUTS = ("10", "13", "14", "16")
# The publisher README reverses the semantics used by its gaze-demo estimator.
# The pinned graph's native channels are CLOSED, OPEN; public APIs use OPEN, CLOSED.
NATIVE_CLASSES = ("CLOSED", "OPEN")
OUTPUT_CONTRACT = {"native_classes": list(NATIVE_CLASSES), "classes": ["OPEN", "CLOSED"],
                   "canonical_from_native": [1, 0],
                   "semantics_source": "https://github.com/openvinotoolkit/open_model_zoo/blob/2c9b5c563ab3a202d085a42e2bbfc48f77961f4b/demos/gaze_estimation_demo/cpp/src/eye_state_estimator.cpp"}


def canonical_logits(native):
    """Reorder native ONNX/ROM channels at the host boundary; arithmetic is unchanged."""
    values = np.asarray(native)
    if values.shape[-1:] != (2,):
        raise ValueError("Expected two native pretrained output channels")
    return values[..., [1, 0]].copy()


def read_model(directory=DIRECTORY):
    raw = (Path(directory) / "model.onnx").read_bytes()
    if len(raw) != 46164 or hashlib.sha384(raw).hexdigest() != CHECKSUM:
        raise ValueError("Pretrained ONNX checksum differs from the pinned official model")
    return raw


def network(raw):
    net = cv2.dnn.readNetFromONNX(np.frombuffer(raw, np.uint8))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def quantized_parameters(raw):
    """Power-of-two scales from conservative L1 bounds, not calibration images."""
    net = network(raw)
    result, fractions, shifts, bounds = {}, [16], [], []
    bound = 128 / 255
    for i, (node, shape) in enumerate(zip(NODES, SHAPES)):
        blobs = net.getLayer(net.getLayerId("onnx_node_output_0!" + node)).blobs
        w = blobs[0].astype(np.float64)
        b = blobs[1].reshape(-1).astype(np.float64) if len(blobs) > 1 else np.zeros(shape[0])
        if w.shape != shape or b.shape != (shape[0],):
            raise ValueError("Official model layer layout changed")
        wf = math.floor(math.log2(16383 / np.abs(w).max()))
        wi = np.floor(w * 2.0**wf + .5).astype(np.int16)
        bi = np.floor(b * 2.0**(fractions[-1] + wf) + .5).astype(np.int64)
        # Include previous rounding error in the recursively propagated range.
        input_bound = math.ceil(bound * 2.0**fractions[-1]) + 1
        acc_bound = int((input_bound * np.abs(wi.astype(np.int64)).reshape(shape[0], -1).sum(1)
                         + np.abs(bi)).max())
        if acc_bound >= 2**47:
            raise ValueError("Pretrained layer exceeds signed 48-bit MAC range")
        real_bound = acc_bound / 2.0**(fractions[-1] + wf)
        out_fraction = min(16, math.floor(math.log2((2**30 - 1) / max(real_bound, 1e-30))))
        shift = fractions[-1] + wf - out_fraction
        if not 1 <= shift <= 46:
            raise ValueError("Unsupported fixed-point shift")
        result[f"w{i}"] = wi
        result[f"b{i}"] = bi
        shifts.append(shift)
        fractions.append(out_fraction)
        bound = real_bound + 2.0**(-out_fraction)
        bounds.append({"weight_fraction": wf, "output_fraction": out_fraction,
                       "accumulator_absolute_bound": acc_bound, "activation_absolute_bound": bound})
    return result, shifts, fractions, bounds


def write_hex(path, values, bits):
    Path(path).write_text("".join(f"{int(v) & ((1 << bits)-1):0{bits//4}x}\n" for v in values), encoding="ascii")


def prepare(directory=DIRECTORY):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "model.onnx"
    try:
        read_model(directory)
    except (FileNotFoundError, ValueError):
        raw = urllib.request.urlopen(URL, timeout=45).read()
        if len(raw) != 46164 or hashlib.sha384(raw).hexdigest() != CHECKSUM:
            raise ValueError("Downloaded model failed official SHA384 verification")
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(target)
    raw = read_model(directory)
    params, shifts, fractions, bounds = quantized_parameters(raw)
    fingerprint = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    generated = ROOT / "fpga/generated"
    weights = np.concatenate([params[f"w{i}"].ravel() for i in range(4)])
    write_hex(generated / "omz_weights.hex", weights, 16)
    # Three 4096-word banks avoid XST rounding the entire ROM up to 16384 words.
    for i in range(3):
        bank = weights[i*4096:(i+1)*4096]
        write_hex(generated / f"omz_weights{i}.hex", np.pad(bank, (0, 4096-len(bank))), 16)
    bias_values = np.concatenate([params[f"b{i}"] for i in range(4)])
    write_hex(generated / "omz_biases.hex", np.pad(bias_values, (0, 128-len(bias_values))), 48)
    config = f"localparam [31:0] OMZ_FINGERPRINT=32'h{fingerprint:08x};\n"
    config += "function [5:0] omz_shift;\ninput [1:0] layer;\nbegin\ncase(layer)\n"
    config += "".join(f"2'd{i}:omz_shift=6'd{shift};\n" for i, shift in enumerate(shifts))
    config += "endcase\nend\nendfunction\n"
    (generated / "omz_config.vh").write_text(config, encoding="ascii")
    record = {"architecture": "open-closed-eye-0001", "source": URL, "sha384": CHECKSUM,
              "sha256": hashlib.sha256(raw).hexdigest(), "input": [3, 32, 32],
              "preprocessing": "BGR; (pixel-127)/255; no AEEE", "classes": ["OPEN", "CLOSED"],
              "output_contract": OUTPUT_CONTRACT,
              "weights": "signed INT16", "activations": "signed INT32", "accumulator": "signed INT48",
              "shifts": shifts, "activation_fractions": fractions, "bounds": bounds,
              "training_performed": False, "calibration_performed": False,
              "accuracy_on_this_system": "not measured", "license": LICENSE_URL}
    record["rtl_files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in generated.glob("omz_*")}
    (directory / "deployment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if not (directory / "LICENSE").exists():
        (directory / "LICENSE").write_bytes(urllib.request.urlopen(LICENSE_URL, timeout=45).read())
    return record


def input_pixels(roi):
    if not isinstance(roi, np.ndarray) or roi.dtype != np.uint8 or roi.ndim != 3 or roi.shape[2] != 3 or not roi.size:
        raise ValueError("Pretrained model requires a nonempty uint8 BGR eye crop")
    return np.ascontiguousarray(cv2.resize(roi, (32, 32), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1))


def fixed_layers(pixels, params, shifts):
    """Independent vectorized integer reference, C/H/W; final layer is global MAX."""
    x = np.floor((pixels.astype(np.float64) - 127) * (65536 / 255) + .5).astype(np.int64)
    values = []
    for i in range(4):
        w = params[f"w{i}"].astype(np.int64)
        k = w.shape[-1]
        windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))
        dot = np.einsum("cyxij,ocij->oyx", windows, w, optimize=True) + params[f"b{i}"][:, None, None]
        x = (dot + (1 << (shifts[i]-1))) >> shifts[i]
        if np.any(x < -(1 << 31)) or np.any(x >= (1 << 31)):
            raise OverflowError("Fixed-point activation overflow")
        if i < 2:
            h, width = x.shape[1]//2, x.shape[2]//2
            x = np.maximum(x[:, :2*h, :2*width].reshape(len(x), h, 2, width, 2).max((2, 4)), 0)
        if i == 3:
            x = x.max((1, 2))
        values.append(x.astype(np.int32))
    return values


class PretrainedEye:
    """The same pinned weights for independent CPU workers and a ROM-based FPGA."""
    def __init__(self, directory=DIRECTORY, precision="float"):
        self.raw = read_model(directory)
        self.params, self.shifts, self.fractions, _ = quantized_parameters(self.raw)
        self.precision = precision
        self.record = {"state": "pretrained", "architecture": "open-closed-eye-0001", "output_contract": OUTPUT_CONTRACT}
        self.method, self.ready = "omz_bgr32", True
        self.checkpoint_sha256 = hashlib.sha256(self.raw).hexdigest()
        self.workers = local()

    def predict(self, roi, fixed=False):
        pixels = input_pixels(roi)
        if fixed or self.precision == "fixed":
            return canonical_logits(fixed_layers(pixels, self.params, self.shifts)[-1])
        if not hasattr(self.workers, "net"):
            self.workers.net = network(self.raw)
        net = self.workers.net
        net.setInput(((pixels.astype(np.float32)-127) / 255)[None])
        # Pre-softmax scores retain the original decision without pretending
        # Softmax outputs are calibrated probabilities of a real eye state.
        return canonical_logits(net.forward("onnx_node_output_0!16").reshape(2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "status", "verify", "verify-uart"))
    parser.add_argument("--output", type=Path, help="New output directory for offline verification")
    parser.add_argument("--pe-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--baseline-ref", help="Pre-extraction Git revision for verify-uart cycle comparison")
    parser.add_argument("--offline", action="store_true", help="Verify local weights with deterministic input fixtures; do not download official example images")
    args = parser.parse_args(argv)
    try:
        if args.action in ("verify", "verify-uart"):
            if args.output is None:
                parser.error("verify requires --output pointing to a new directory")
            from pretrained_verify import verify, verify_uart
            record = (verify(args.output, args.pe_count, offline=args.offline) if args.action == "verify" else
                      verify_uart(args.output, args.pe_count, baseline_ref=args.baseline_ref))
        else:
            record = prepare() if args.action == "prepare" else json.loads((DIRECTORY / "deployment.json").read_text())
        read_model()
        print(json.dumps(record, indent=2))
        return 0 if record.get("status", "PASS") == "PASS" else 1
    except (OSError, ValueError, cv2.error, subprocess.SubprocessError) as exc:
        parser.exit(1, f"PRETRAINED MODEL ERROR: {exc}\n")
