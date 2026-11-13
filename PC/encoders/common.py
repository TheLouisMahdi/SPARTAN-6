"""Input and byte-format contracts, without choosing an encoding method."""

import cv2
import numpy as np


def validate_roi(roi: np.ndarray):
    if (not isinstance(roi, np.ndarray) or roi.dtype != np.uint8
            or roi.shape != (16, 32) or not roi.flags.c_contiguous):
        raise ValueError("Encoder input must be a contiguous uint8 canonical ROI with shape (16,32).")


def sobel_components(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unscaled signed 3x3 derivatives; nearest-pixel replicated borders."""
    gx = cv2.Sobel(roi, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0,
                   borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(roi, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0,
                   borderType=cv2.BORDER_REPLICATE)
    return gx.astype(np.int32), gy.astype(np.int32)


def pack2(codes: np.ndarray) -> bytes:
    validate_roi(codes)
    if np.any(codes > 3):
        raise ValueError("2-bit symbols must be integers in 0..3.")
    pixels = codes.reshape(-1, 4)
    packed = (pixels[:, 0] << 6) | (pixels[:, 1] << 4) | (pixels[:, 2] << 2) | pixels[:, 3]
    return packed.tobytes()


def unpack2(payload: bytes) -> np.ndarray:
    if not isinstance(payload, bytes) or len(payload) != 128:
        raise ValueError("One 2-bit eye payload must be exactly 128 bytes.")
    packed = np.frombuffer(payload, np.uint8)
    result = np.empty((128, 4), np.uint8)
    for column, shift in enumerate((6, 4, 2, 0)):
        result[:, column] = (packed >> shift) & 3
    return result.reshape(16, 32)
