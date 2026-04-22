"""
Patch Rotation — shape-bias transform.

Divides the image into non-overlapping 32×32 patches and randomly rotates
each patch by 90°, 180°, or 270°. Local texture orientation within each patch
is disrupted while the global spatial layout of anatomical structures is
preserved — nudging the model toward shape-level features.
"""

import random

import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)
PATCH_SIZE  = 32   # must evenly divide TARGET_SIZE


def apply(img: Image.Image, patch_size: int = PATCH_SIZE) -> Image.Image:
    """
    Randomly rotate each non-overlapping patch by a multiple of 90°.

    Args:
        img:        Input image (any mode).
        patch_size: Side length of each square patch in pixels.

    Returns:
        Grayscale (mode 'L') patch-rotated image at TARGET_SIZE.
    """
    pil = img.convert("L")
    if pil.size != TARGET_SIZE:
        pil = pil.resize(TARGET_SIZE, Image.BILINEAR)
    gray = np.array(pil)
    H, W   = gray.shape
    canvas = np.zeros((H, W), dtype=np.uint8)

    for row in range(0, H, patch_size):
        for col in range(0, W, patch_size):
            patch  = gray[row : row + patch_size, col : col + patch_size]
            k      = random.choice([1, 2, 3])   # 90°, 180°, or 270°
            canvas[row : row + patch_size, col : col + patch_size] = np.rot90(patch, k=k)

    return Image.fromarray(canvas)
