import pytest
from src.config import DataConfig


# TODO: replace stubs with real dataset tests once MyDataset is implemented

def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.raw_dir == "data/raw"
    assert cfg.processed_dir == "data/processed"


# Example structure for dataset tests:
# def test_dataset_len():
#     cfg = DataConfig()
#     ds = MyDataset("train", cfg)
#     assert len(ds) > 0
#
# def test_dataset_item_shape():
#     cfg = DataConfig()
#     ds = MyDataset("train", cfg)
#     x, y = ds[0]
#     assert x.shape == (C, H, W)  # TODO: set expected shape
