"""
Tests for src/data/chexpert_dataset.py — CheXpertDataset
Run with: pytest src/tests/test_dataset.py -v
"""
import io
import numpy as np
import polars as pl
import pytest
import torch
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock
from torchvision import transforms

from data.chexpert_dataset import CheXpertDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_LABELS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'No Finding', 'Support Devices',
]


def _make_parquet(
    tmp_path: Path,
    rows: int = 5,
    label_cols: list[str] | None = None,
    extra_cols: dict | None = None,
) -> Path:
    """Write a minimal valid parquet manifest and return its path."""
    cols = label_cols or DEFAULT_LABELS
    data = {"Path": [f"train/patient{i:05d}/study1/view1_frontal.jpg" for i in range(rows)]}
    for col in cols:
        data[col] = [i % 2 for i in range(rows)]  # alternating 0/1
    if extra_cols:
        data.update(extra_cols)
    df = pl.DataFrame(data)
    out = tmp_path / "manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    return out


def _make_dummy_image(width: int = 64, height: int = 64) -> Image.Image:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _patch_image_open(img: Image.Image | None = None):
    """Context manager: patches PIL.Image.open to return a dummy image."""
    target_img = img or _make_dummy_image()
    return patch("src.data.chexpert_dataset.Image.open", return_value=target_img)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def parquet_path(tmp_path):
    return _make_parquet(tmp_path, rows=10)


@pytest.fixture()
def dataset(parquet_path, tmp_path):
    with _patch_image_open():
        return CheXpertDataset(
            manifest_path=str(parquet_path),
            image_root_dir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_loads_default_labels(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path))
        assert ds.target_cols == DEFAULT_LABELS

    def test_loads_custom_labels(self, tmp_path):
        custom = ['Edema', 'Fracture']
        pq = _make_parquet(tmp_path, label_cols=custom)
        ds = CheXpertDataset(str(pq), str(tmp_path), target_cols=custom)
        assert ds.target_cols == custom

    def test_len_matches_manifest(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path))
        assert len(ds) == 10

    def test_targets_shape(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path))
        assert ds.targets.shape == (10, len(DEFAULT_LABELS))

    def test_targets_dtype_float32(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path))
        assert ds.targets.dtype == np.float32

    def test_paths_length_matches(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path))
        assert len(ds.paths) == 10

    def test_missing_label_column_raises(self, tmp_path):
        """Parquet missing a required label column should raise at target extraction."""
        pq = _make_parquet(tmp_path, label_cols=['Edema'])  # only 1 col
        with pytest.raises(Exception):
            CheXpertDataset(str(pq), str(tmp_path), target_cols=['Edema', 'Fracture'])

    def test_missing_parquet_raises(self, tmp_path):
        with pytest.raises(Exception):
            CheXpertDataset(str(tmp_path / "nonexistent.parquet"), str(tmp_path))


# ---------------------------------------------------------------------------
# 2. __len__
# ---------------------------------------------------------------------------

class TestLen:
    def test_returns_int(self, dataset):
        assert isinstance(len(dataset), int)

    def test_nonzero(self, dataset):
        assert len(dataset) > 0

    @pytest.mark.parametrize("n", [1, 3, 7])
    def test_various_sizes(self, tmp_path, n):
        pq = _make_parquet(tmp_path / str(n), rows=n)
        ds = CheXpertDataset(str(pq), str(tmp_path))
        assert len(ds) == n


# ---------------------------------------------------------------------------
# 3. __getitem__ — image found
# ---------------------------------------------------------------------------

class TestGetItemFound:
    def test_returns_tuple_of_two(self, dataset):
        with _patch_image_open():
            out = dataset[0]
        assert len(out) == 2

    def test_label_is_tensor(self, dataset):
        with _patch_image_open():
            _, label = dataset[0]
        assert isinstance(label, torch.Tensor)

    def test_label_shape(self, dataset):
        with _patch_image_open():
            _, label = dataset[0]
        assert label.shape == (len(DEFAULT_LABELS),)

    def test_label_dtype_float(self, dataset):
        with _patch_image_open():
            _, label = dataset[0]
        assert label.dtype == torch.float32

    def test_label_values_binary(self, dataset):
        with _patch_image_open():
            _, label = dataset[0]
        unique_vals = label.unique().tolist()
        assert all(v in (0.0, 1.0) for v in unique_vals)

    def test_image_without_transform_is_pil(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), transform=None)
        with _patch_image_open():
            image, _ = ds[0]
        # Without a transform the dataset returns the raw PIL image
        assert isinstance(image, Image.Image)

    def test_image_with_transform_is_tensor(self, parquet_path, tmp_path):
        tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), transform=tfm)
        with _patch_image_open():
            image, _ = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 32, 32)

    def test_rgb_conversion(self, parquet_path, tmp_path):
        """Grayscale source images should be converted to RGB."""
        gray_img = Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L")
        tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), transform=tfm)
        with patch("src.data.chexpert_dataset.Image.open", return_value=gray_img):
            image, _ = ds[0]
        assert image.shape[0] == 3  # RGB → 3 channels

    def test_index_boundary_last(self, dataset):
        with _patch_image_open():
            out = dataset[len(dataset) - 1]
        assert out is not None


# ---------------------------------------------------------------------------
# 4. __getitem__ — image missing (fallback path)
# ---------------------------------------------------------------------------

class TestGetItemMissing:
    def test_missing_image_returns_zero_tensor(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path))
        # Don't patch → Image.open will raise FileNotFoundError on the fake path
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert torch.all(image == 0)

    def test_missing_image_label_is_zero_tensor(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path))
        _, label = ds[0]
        assert isinstance(label, torch.Tensor)
        assert label.shape == (len(DEFAULT_LABELS),)
        assert torch.all(label == 0)

    def test_missing_image_label_length_matches_target_cols(self, tmp_path):
        custom = ['Edema', 'Fracture']
        pq = _make_parquet(tmp_path, label_cols=custom)
        ds = CheXpertDataset(str(pq), str(tmp_path), target_cols=custom)
        _, label = ds[0]
        assert label.shape == (len(custom),)


# ---------------------------------------------------------------------------
# 5. Transform behaviour
# ---------------------------------------------------------------------------

class TestTransform:
    def test_no_transform_does_not_crash(self, parquet_path, tmp_path):
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), transform=None)
        with _patch_image_open():
            ds[0]  # should not raise

    def test_transform_is_called(self, parquet_path, tmp_path):
        mock_tfm = MagicMock(return_value=torch.zeros(3, 32, 32))
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), transform=mock_tfm)
        with _patch_image_open():
            ds[0]
        mock_tfm.assert_called_once()

    def test_transform_receives_pil_image(self, parquet_path, tmp_path):
        received = []
        def capturing_tfm(img):
            received.append(img)
            return img
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), transform=capturing_tfm)
        with _patch_image_open():
            ds[0]
        assert isinstance(received[0], Image.Image)


# ---------------------------------------------------------------------------
# 6. Custom target_cols
# ---------------------------------------------------------------------------

class TestCustomTargetCols:
    def test_single_col(self, tmp_path):
        pq = _make_parquet(tmp_path, label_cols=['Edema'])
        ds = CheXpertDataset(str(pq), str(tmp_path), target_cols=['Edema'])
        with _patch_image_open():
            _, label = ds[0]
        assert label.shape == (1,)

    def test_subset_of_default(self, parquet_path, tmp_path):
        subset = ['Edema', 'Fracture', 'No Finding']
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), target_cols=subset)
        with _patch_image_open():
            _, label = ds[0]
        assert label.shape == (3,)

    def test_target_cols_stored(self, parquet_path, tmp_path):
        cols = ['Edema', 'Fracture']
        ds = CheXpertDataset(str(parquet_path), str(tmp_path), target_cols=cols)
        assert ds.target_cols == cols
