import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import polars as pl


class CheXpertDataset(Dataset):
    """Loads CheXpert images and labels from a Parquet manifest."""

    DEFAULT_LABELS = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "No Finding",
        "Support Devices",
    ]

    def __init__(self, manifest_path, image_root_dir, transform=None, target_cols=None):
        self.image_root_dir = Path(image_root_dir)
        self.transform      = transform
        self.target_cols    = target_cols or self.DEFAULT_LABELS

        print(f"Loading manifest from {manifest_path}...")
        df = pl.read_parquet(manifest_path)

        # Manifest paths start with "CheXpert-v1.0-small/..." — strip that prefix
        # so paths resolve correctly under image_root_dir
        self.paths   = [str(Path(*Path(p).parts[1:])) for p in df["Path"].to_list()]
        self.targets = df.select(self.target_cols).to_numpy().astype("float32")

        print(f"Dataset loaded. Size: {len(self.paths)} images. Labels: {self.target_cols}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.image_root_dir / self.paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}")
            return torch.zeros((3, 224, 224)), torch.zeros(len(self.target_cols))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.targets[idx])
