"""Task-independent Sobel L1 magnitude, uniformly quantized to four levels."""

import numpy as np

from .common import sobel_components, validate_roi

SPEC = {
    "version": "sobel2-v1", "bits_per_pixel": 2, "bytes_per_eye": 128,
    "derivatives": "3x3 Sobel, signed intermediates, scale=1, BORDER_REPLICATE",
    "formula": "M=abs(Gx)+abs(Gy); E=floor((M+3)/6); C=floor(E/64)",
    "normalization": "Fixed: the joint Sobel L1 maximum is 1530 for uint8, so E is 0..255.",
    "symbols": {"0": "M 0..380", "1": "M 381..764", "2": "M 765..1148", "3": "M 1149..1530"},
}


def encode(roi):
    validate_roi(roi)
    gx, gy = sobel_components(roi)
    magnitude = np.abs(gx) + np.abs(gy)
    edge8 = (magnitude + 3) // 6
    return np.ascontiguousarray(edge8 // 64, dtype=np.uint8)
