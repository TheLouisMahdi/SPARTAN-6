"""Uniform, fixed four-bin grayscale quantization; no fitted thresholds."""

from .common import validate_roi

SPEC = {
    "version": "grayscale2-v1", "bits_per_pixel": 2, "bytes_per_eye": 128,
    "formula": "C(y,x) = floor(I(y,x)/64)",
    "symbols": {"0": "0..63", "1": "64..127", "2": "128..191", "3": "192..255"},
}


def encode(roi):
    validate_roi(roi)
    return roi // 64
