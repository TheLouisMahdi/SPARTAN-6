"""Portable INT8-weight/activation reference with specified integer arithmetic.

Not FPGA RTL, not an optimized runtime. All convolutions/FC accumulators are
int32; requantization uses int64, Q24 multiplier, half-up rounding and saturation.
Input and hidden activations are affine signed int8 (zero point -128).
"""
from pathlib import Path
import json

import numpy as np

FRACTION_BITS = 24
LAYERS = ("conv1", "conv2", "fc1", "fc2")


def make_parameters(model, calibration, batch_size=512):
    """Only TRAIN calibration tensors may be supplied. No test/val fitting."""
    import torch
    maxima = np.zeros(3, np.float64)
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(calibration), batch_size):
            acts = model.activations(calibration[offset:offset + batch_size])
            maxima = np.maximum(maxima, [float(x.max()) for x in acts])
    scales = [1 / 255] + [float(value / 255) if value > 0 else 1.0 for value in maxima]
    params, metadata = {}, {
        "version": "integer-eye-v1", "input_scale": scales[0], "activation_zero_point": -128,
        "activation_storage": "int8; real=(q+128)*scale, nonnegative magnitudes 0..255",
        "weight_storage": "int8 symmetric per-layer scale, range -127..127; numpy rint ties-to-even",
        "bias_storage": "little-endian int32, scale=input_scale*weight_scale; rint ties-to-even",
        "requantization": "u=clip((max(acc,0)*M+2**23)>>24,0,255); q=u-128; int64 product",
        "pooling": "2x2 stride2 floor, signed int8 max", "flatten": "C,H,W",
        "output": "Two int32 accumulators at one common scale; argmax, ties select OPEN (0)",
        "calibration_count": len(calibration), "layers": [],
    }
    for index, name in enumerate(LAYERS):
        layer = getattr(model, name)
        weights = layer.weight.detach().cpu().numpy().astype(np.float64)
        bias = layer.bias.detach().cpu().numpy().astype(np.float64)
        weight_scale = float(np.max(np.abs(weights)) / 127) or 1.0
        qw = np.clip(np.rint(weights / weight_scale), -127, 127).astype(np.int8)
        qb64 = np.rint(bias / (scales[index] * weight_scale)).astype(np.int64)
        # Proven worst-case unsigned activation * signed weight accumulation bound.
        bound = 255 * np.abs(qw.astype(np.int64)).reshape(len(qw), -1).sum(axis=1) + np.abs(qb64)
        if np.any(bound > np.iinfo(np.int32).max):
            raise OverflowError(f"int32 accumulator would overflow: {name}")
        params[name + "_weight"] = qw
        params[name + "_bias"] = qb64.astype("<i4")
        record = {"name": name, "weight_shape": list(qw.shape), "weight_scale": weight_scale,
                  "input_scale": scales[index], "max_abs_accumulator_bound": int(bound.max())}
        if index < 3:
            multiplier = int(np.rint(scales[index] * weight_scale / scales[index + 1] * (1 << FRACTION_BITS)))
            if multiplier <= 0 or int(bound.max()) * multiplier + (1 << 23) > np.iinfo(np.int64).max:
                raise OverflowError(f"Invalid Q24 multiplier/product bound: {name}")
            params[name + "_multiplier"] = np.array(multiplier, dtype=np.int64)
            record.update(output_scale=scales[index + 1], multiplier=multiplier)
        else:
            record["output_scale"] = scales[index] * weight_scale
        metadata["layers"].append(record)
    return params, metadata


def save_parameters(directory, params, metadata):
    directory = Path(directory)
    np.savez(directory / "int8_parameters.npz", **params)
    (directory / "int8_format.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    # Explicit byte streams for a later RTL loader. The JSON describes shapes/order.
    with (directory / "int8_weights.bin").open("xb") as weights, (directory / "int32_biases.bin").open("xb") as biases:
        for name in LAYERS:
            weights.write(params[name + "_weight"].tobytes(order="C"))
            biases.write(params[name + "_bias"].astype("<i4").tobytes())


def conv(x, weight, bias):
    magnitude = x.astype(np.int32) + 128
    patches = np.lib.stride_tricks.sliding_window_view(magnitude, (3, 3), axis=(2, 3))
    n, _, h, w, _, _ = patches.shape
    matrix = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, weight.shape[1] * 9)
    result = matrix @ weight.astype(np.int32).reshape(len(weight), -1).T + bias
    return result.reshape(n, h, w, len(weight)).transpose(0, 3, 1, 2)


def linear(x, weight, bias):
    return (x.astype(np.int32) + 128) @ weight.astype(np.int32).T + bias


def requantize(acc, multiplier):
    positive = np.maximum(acc, 0).astype(np.int64)
    magnitude = (positive * int(multiplier) + (1 << 23)) >> FRACTION_BITS
    return (np.clip(magnitude, 0, 255) - 128).astype(np.int8)


def pool(x):
    h, w = x.shape[2] // 2 * 2, x.shape[3] // 2 * 2
    return np.maximum.reduce([x[:, :, dy:h:2, dx:w:2] for dy in (0, 1) for dx in (0, 1)])


def infer(pixels, params):
    """pixels: uint8 N,1,16,32; low-bit codes already mapped to 0,85,170,255."""
    if pixels.dtype != np.uint8 or pixels.ndim != 4 or pixels.shape[1:] != (1, 16, 32):
        raise ValueError("Integer input must be uint8 N,1,16,32")
    x = (pixels.astype(np.int16) - 128).astype(np.int8)
    for name in ("conv1", "conv2"):
        x = pool(requantize(conv(x, params[name + "_weight"], params[name + "_bias"]),
                            params[name + "_multiplier"]))
    x = x.reshape(len(x), -1)
    x = requantize(linear(x, params["fc1_weight"], params["fc1_bias"]), params["fc1_multiplier"])
    return linear(x, params["fc2_weight"], params["fc2_bias"])
