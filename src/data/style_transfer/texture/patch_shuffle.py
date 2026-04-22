"""
Patch Shuffle — texture-bias transform.

Divides the image into non-overlapping 32×32 patches and randomly permutes
their positions. Global anatomical structure (shape) is destroyed while local
texture statistics within each patch are fully preserved. A model trained on
these images cannot learn shape-based features and must rely on texture.
"""

import numpy as np
from PIL import Image

TARGET_SIZE = (224, 224)
PATCH_SIZE  = 32   # must evenly divide TARGET_SIZE


def apply(img: Image.Image, patch_size: int = PATCH_SIZE) -> Image.Image:
    """
    Randomly shuffle non-overlapping patches of a PIL image.

    Args:
        img:        Input image (any mode).
        patch_size: Side length of each square patch in pixels.
                    Must evenly divide both image dimensions.

    Returns:
        Grayscale (mode 'L') patch-shuffled image at TARGET_SIZE.
    """
    gray = img.convert("L")
    if gray.size != TARGET_SIZE:
        gray = gray.resize(TARGET_SIZE, Image.BILINEAR)
    arr  = np.array(gray)           # (H, W)
    H, W = arr.shape

    n_h = H // patch_size
    n_w = W // patch_size
    n_patches = n_h * n_w

    # Reshape into a grid of patches: (n_h, n_w, patch_size, patch_size)
    patches = (
        arr
        .reshape(n_h, patch_size, n_w, patch_size)
        .transpose(0, 2, 1, 3)             # (n_h, n_w, ph, pw)
        .reshape(n_patches, patch_size, patch_size)
    )

    # Permute patch order
    patches = patches[np.random.permutation(n_patches)]

    # Reconstruct image from shuffled patches
    result = (
        patches
        .reshape(n_h, n_w, patch_size, patch_size)
        .transpose(0, 2, 1, 3)             # (n_h, ph, n_w, pw)
        .reshape(H, W)
    )

    return Image.fromarray(result)
