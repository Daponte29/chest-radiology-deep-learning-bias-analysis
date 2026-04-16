import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import polars as pl


class CheXpertDataset(Dataset):
    """
    Custom Dataset for CheXpert that loads from a processed Parquet manifest.
    Supports standard images as well as stylized variants for shape/texture bias analysis.
    """

    DEFAULT_LABELS = [
        'Enlarged Cardiomediastinum', # 1
        'Cardiomegaly', # 2
        'Lung Opacity', # 3
        'Lung Lesion',  # 4
        'Edema',         # 5
        'Consolidation', # 6
        'Pneumonia',     # 7
        'Atelectasis',   # 8
        'Pneumothorax',  # 9
        'Pleural Effusion', # 10
        'Pleural Other', # 11
        'Fracture',      # 12
        'No Finding',    # 13
        'Support Devices', # 14
    ]

    def __init__(
        self,
        manifest_path,
        image_root_dir,
        transform=None,
        target_cols=None,
    ):
        """
        Args:
            manifest_path (str): Path to the parquet manifest file.
            image_root_dir (str): Root directory where images are stored.
            transform (callable, optional): PyTorch transform pipeline.
            target_cols (list, optional): Label columns to use as targets.
                Defaults to all 14 CheXpert pathology columns.
        """
        self.image_root_dir = Path(image_root_dir)
        self.transform = transform
        self.target_cols = target_cols or self.DEFAULT_LABELS

        print(f"Loading manifest from {manifest_path}...")
        self.df = pl.read_parquet(manifest_path)
        # Manifest paths are stored as 'CheXpert-v1.0-small/train/...' but images
        # are rooted directly at image_root_dir, so drop the first path component.
        self.paths = [str(Path(*Path(p).parts[1:])) for p in self.df["Path"].to_list()]
        self.targets = self.df.select(self.target_cols).to_numpy().astype("float32")
        print(f"Dataset loaded. Size: {len(self.paths)} images. Targets: {self.target_cols}")

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
