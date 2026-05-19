from __future__ import annotations

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.config import DataConfig


class MyDataset(Dataset):
    """TODO: replace with your dataset logic."""

    def __init__(self, split: str, cfg: DataConfig):
        self.cfg = cfg
        self.split = split
        self.samples = self._load(split)

    def _load(self, split: str) -> list:
        # TODO: load file paths / labels from processed_dir
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # TODO: load and return (input_tensor, label)
        raise NotImplementedError


def build_loaders(cfg: DataConfig, batch_size: int, num_workers: int = 4):
    train_ds = MyDataset("train", cfg)
    val_ds = MyDataset("val", cfg)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
