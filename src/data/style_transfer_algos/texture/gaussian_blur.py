"""
Gaussian Blur — texture-bias transform.

Blurs the image with a Gaussian kernel, suppressing high-frequency texture
details. Forces a model trained on these images to rely on low-frequency
shape and structural cues rather than fine-grained texture patterns.
"""

from PIL import Image, ImageFilter

TARGET_SIZE = (224, 224)
BLUR_RADIUS = 2


def apply(img: Image.Image, radius: int = BLUR_RADIUS) -> Image.Image:
    """
    Apply Gaussian blur to a PIL image.

    Args:
        img:    Input image (any mode).
        radius: Blur kernel radius in pixels.

    Returns:
        Grayscale (mode 'L') blurred image at TARGET_SIZE.
    """
    gray = img.convert("L")
    if gray.size != TARGET_SIZE:
        gray = gray.resize(TARGET_SIZE, Image.BILINEAR)
    return gray.filter(ImageFilter.GaussianBlur(radius=radius))
