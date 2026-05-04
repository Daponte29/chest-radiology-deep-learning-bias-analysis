"""
Canny Edge Detection — shape-bias transform.

Extracts only the edges and contours of structures in the image, discarding
all texture and intensity information. A model trained on these images must
learn to classify from pure shape/contour features alone.

Thresholds are set automatically per image using the median pixel intensity
(the same sigma=0.33 heuristic used in the original CheXpert shape-bias work).
"""

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)


def apply(img: Image.Image) -> Image.Image:
    """
    Apply Canny edge detection to a PIL image.

    Args:
        img: Input image (any mode).

    Returns:
        Grayscale (mode 'L') binary edge map at TARGET_SIZE.
        White pixels are edges; background is black.
    """
    pil = img.convert("L")
    if pil.size != TARGET_SIZE:
        pil = pil.resize(TARGET_SIZE, Image.BILINEAR)
    gray = np.array(pil)

    # Auto-threshold: lo = 0.77 * median, hi = 1.33 * median
    median = np.median(gray)
    lo = max(int(0.77 * median), 0)
    hi = min(int(1.33 * median), 255)

    edges = cv2.Canny(gray, lo, hi)
    return Image.fromarray(edges)
