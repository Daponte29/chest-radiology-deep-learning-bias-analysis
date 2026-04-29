"""
Tests for all 4 stylization transforms:
  - texture: gaussian_blur, patch_shuffle
  - shape:   canny_edge, patch_rotation
"""
import numpy as np
import pytest
from PIL import Image

from data.style_transfer.texture.gaussian_blur import apply as blur_apply
from data.style_transfer.texture.patch_shuffle import apply as shuffle_apply
from data.style_transfer.shape.canny_edge import apply as canny_apply
from data.style_transfer.shape.patch_rotation import apply as rotation_apply


TARGET_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_image(size=(256, 256)) -> Image.Image:
    arr = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _gray_image(size=(256, 256)) -> Image.Image:
    arr = np.random.randint(0, 256, size[::-1], dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _uniform_image(value: int = 128, size=(256, 256)) -> Image.Image:
    arr = np.full((*size[::-1], 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Gaussian Blur (texture bias)
# ---------------------------------------------------------------------------

class TestGaussianBlur:
    def test_returns_pil_image(self):
        assert isinstance(blur_apply(_rgb_image()), Image.Image)

    def test_output_size(self):
        assert blur_apply(_rgb_image()).size == TARGET_SIZE

    def test_output_mode_grayscale(self):
        assert blur_apply(_rgb_image()).mode == "L"

    def test_accepts_grayscale_input(self):
        out = blur_apply(_gray_image())
        assert out.size == TARGET_SIZE

    def test_accepts_already_target_size(self):
        out = blur_apply(_rgb_image(size=TARGET_SIZE))
        assert out.size == TARGET_SIZE

    def test_blur_reduces_high_freq(self):
        # Sharp checkerboard vs uniform: blurred checkerboard should have lower std
        checker = np.indices((224, 224)).sum(axis=0) % 2 * 255
        sharp_img = Image.fromarray(checker.astype(np.uint8), mode="L")
        blurred = blur_apply(sharp_img)
        assert np.std(np.array(blurred)) < np.std(checker)

    def test_uniform_image_stays_uniform(self):
        out = blur_apply(_uniform_image(128))
        arr = np.array(out)
        assert arr.min() == arr.max()

    def test_custom_radius_accepted(self):
        out = blur_apply(_rgb_image(), radius=5)
        assert out.size == TARGET_SIZE


# ---------------------------------------------------------------------------
# Patch Shuffle (texture bias)
# ---------------------------------------------------------------------------

class TestPatchShuffle:
    def test_returns_pil_image(self):
        assert isinstance(shuffle_apply(_rgb_image()), Image.Image)

    def test_output_size(self):
        assert shuffle_apply(_rgb_image()).size == TARGET_SIZE

    def test_output_mode_grayscale(self):
        assert shuffle_apply(_rgb_image()).mode == "L"

    def test_preserves_pixel_multiset(self):
        img = _rgb_image()
        out = shuffle_apply(img)
        original_gray = np.array(img.convert("L").resize(TARGET_SIZE, Image.BILINEAR))
        out_arr = np.array(out)
        assert sorted(original_gray.flatten()) == sorted(out_arr.flatten())

    def test_output_is_different_from_input(self):
        # With high probability a random image will be shuffled differently
        np.random.seed(42)
        img = _rgb_image()
        original_gray = np.array(img.convert("L").resize(TARGET_SIZE, Image.BILINEAR))
        out_arr = np.array(shuffle_apply(img))
        assert not np.array_equal(original_gray, out_arr)

    def test_custom_patch_size(self):
        out = shuffle_apply(_rgb_image(), patch_size=16)
        assert out.size == TARGET_SIZE

    def test_accepts_already_target_size(self):
        assert shuffle_apply(_rgb_image(size=TARGET_SIZE)).size == TARGET_SIZE


# ---------------------------------------------------------------------------
# Canny Edge (shape bias)
# ---------------------------------------------------------------------------

class TestCannyEdge:
    def test_returns_pil_image(self):
        assert isinstance(canny_apply(_rgb_image()), Image.Image)

    def test_output_size(self):
        assert canny_apply(_rgb_image()).size == TARGET_SIZE

    def test_output_mode_grayscale(self):
        assert canny_apply(_rgb_image()).mode == "L"

    def test_uniform_image_produces_no_edges(self):
        out = canny_apply(_uniform_image(128))
        assert np.array(out).sum() == 0

    def test_checkerboard_produces_edges(self):
        checker = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(np.uint8)
        img = Image.fromarray(checker, mode="L")
        out = canny_apply(img)
        assert np.array(out).sum() > 0

    def test_output_is_binary_valued(self):
        out = canny_apply(_rgb_image())
        unique = np.unique(np.array(out))
        assert set(unique).issubset({0, 255})

    def test_accepts_grayscale_input(self):
        out = canny_apply(_gray_image())
        assert out.size == TARGET_SIZE


# ---------------------------------------------------------------------------
# Patch Rotation (shape bias)
# ---------------------------------------------------------------------------

class TestPatchRotation:
    def test_returns_pil_image(self):
        assert isinstance(rotation_apply(_rgb_image()), Image.Image)

    def test_output_size(self):
        assert rotation_apply(_rgb_image()).size == TARGET_SIZE

    def test_output_mode_grayscale(self):
        assert rotation_apply(_rgb_image()).mode == "L"

    def test_preserves_pixel_multiset(self):
        img = _rgb_image()
        out = rotation_apply(img)
        original_gray = np.array(img.convert("L").resize(TARGET_SIZE, Image.BILINEAR))
        out_arr = np.array(out)
        # Rotation preserves all pixel values, just rearranges within each patch
        assert sorted(original_gray.flatten()) == sorted(out_arr.flatten())

    def test_uniform_image_unchanged(self):
        img = _uniform_image(100)
        out = rotation_apply(img)
        assert np.array(out).min() == np.array(out).max() == 100

    def test_custom_patch_size(self):
        out = rotation_apply(_rgb_image(), patch_size=16)
        assert out.size == TARGET_SIZE

    def test_accepts_grayscale_input(self):
        out = rotation_apply(_gray_image())
        assert out.size == TARGET_SIZE

    def test_accepts_already_target_size(self):
        assert rotation_apply(_rgb_image(size=TARGET_SIZE)).size == TARGET_SIZE


# ---------------------------------------------------------------------------
# Cross-transform consistency
# ---------------------------------------------------------------------------

class TestCrossTransform:
    @pytest.mark.parametrize("fn", [blur_apply, shuffle_apply, canny_apply, rotation_apply])
    def test_all_produce_target_size(self, fn):
        out = fn(_rgb_image())
        assert out.size == TARGET_SIZE

    @pytest.mark.parametrize("fn", [blur_apply, shuffle_apply, canny_apply, rotation_apply])
    def test_all_produce_grayscale(self, fn):
        out = fn(_rgb_image())
        assert out.mode == "L"

    @pytest.mark.parametrize("fn", [blur_apply, shuffle_apply, canny_apply, rotation_apply])
    def test_all_handle_rgb_input(self, fn):
        out = fn(_rgb_image(size=(300, 400)))
        assert out.size == TARGET_SIZE
