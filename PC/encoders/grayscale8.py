"""Lossless baseline: all original canonical grayscale intensities."""

from .common import validate_roi

SPEC = {
    "version": "grayscale8-v1", "bits_per_pixel": 8, "bytes_per_eye": 512,
    "formula": "C(y,x) = I(y,x)", "symbols": "Original 0..255 intensity values.",
}


def encode(roi):
    validate_roi(roi)
    return roi.copy()
