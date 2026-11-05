"""Independent, versioned encoders of the same canonical uint8 eye ROI."""

from . import aeee2, grayscale2, grayscale8, sobel2
from .common import pack2, unpack2, validate_roi

METHODS = {
    "grayscale8": grayscale8,
    "grayscale2": grayscale2,
    "sobel2": sobel2,
    "aeee2": aeee2,
}
