"""Proposed AEEE v0.1: a hypothesis, not an established novel/superior method.

Fixed central-band prior, per-image integer threshold, horizontal/diagonal
edge continuity and strong-edge polarity. No labels, eye state, landmarks,
iris, model scores or other encoder outputs enter this method.
"""

import numpy as np

from .common import sobel_components, validate_roi

SPEC = {
    "version": "aeee2-v0.1", "bits_per_pixel": 2, "bytes_per_eye": 128,
    "status": "Initial unvalidated research proposal; novelty and superiority unestablished.",
    "derivatives": "3x3 Sobel, signed intermediates, scale=1, BORDER_REPLICATE",
    "central_band": "W(y)=2 for y=3..12 inclusive, otherwise 1 (zero-based rows)",
    "score": "S=W*abs(Gy)",
    "threshold": "T=max(64, floor((sum(S)+511)/512))",
    "eligible": "S>=T and abs(Gy)>0 and 2*abs(Gy)>=abs(Gx)",
    "continuity": "Eligible neighbor at x-1 or x+1, with vertical offset -1,0,+1; outside neighbors are false.",
    "symbols": {"0": "not retained", "1": "retained, S<2*T",
                "2": "retained, S>=2*T and Gy>0", "3": "retained, S>=2*T and Gy<0"},
    "parameter_policy": "Frozen design priors, not selected with validation/test labels; no optimality claimed.",
}


def horizontal_support(eligible: np.ndarray) -> np.ndarray:
    padded = np.pad(eligible, 1, constant_values=False)
    support = np.zeros_like(eligible)
    for offset in (0, 1, 2):
        support |= padded[offset:offset + 16, :32]
        support |= padded[offset:offset + 16, 2:34]
    return support


def encode(roi):
    validate_roi(roi)
    gx, gy = sobel_components(roi)
    vertical = np.abs(gy)
    score = vertical.copy()
    score[3:13] *= 2
    threshold = max(64, (int(score.sum(dtype=np.int64)) + 511) // 512)
    eligible = (score >= threshold) & (vertical > 0) & (2 * vertical >= np.abs(gx))
    retained = eligible & horizontal_support(eligible)
    strong = retained & (score >= 2 * threshold)
    codes = np.zeros((16, 32), np.uint8)
    codes[retained] = 1
    codes[strong & (gy > 0)] = 2
    codes[strong & (gy < 0)] = 3
    return codes
