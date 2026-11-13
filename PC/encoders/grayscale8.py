"""Lossless baseline: all original canonical grayscale intensities."""

import numpy as np

from .common import validate_roi

SPEC = {
    "version": "grayscale8-v1", "bits_per_pixel": 8, "bytes_per_eye": 512,
    "formula": "C(y,x) = I(y,x)", "symbols": "Original 0..255 intensity values.",
}


def encode(roi):
    validate_roi(roi)
    return roi.copy()


def normalize_contrast(roi):
    """Experimental per-eye stretch with limited gain for low-contrast crops."""
    validate_roi(roi)
    low, high = np.percentile(roi, [5, 95])
    center = (low + high) / 2
    gain = min(3.0, 192 / max(high - low, 1.0))
    return np.clip(np.rint((roi.astype(np.float32) - center) * gain + 127.5), 0, 255).astype(np.uint8)
