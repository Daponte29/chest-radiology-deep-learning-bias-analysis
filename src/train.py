"""
train.py — CheXpert DenseNet Training Script
=============================================
Goal: train a DenseNet on CheXpert chest X-rays for multi-label pathology classification.


What this script does, step by step:
    1. Load config (YAML)
    2. Set random seed for reproducibility
    3. Build train and validation datasets + dataloaders
    4. Build DenseNet model
    5. Define loss function and optimizer
    6. Loop over epochs:
         - train for one epoch
         - evaluate on validation set (loss + AUROC per label)
         - save the best model checkpoint
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.chexpert_dataset import CheXpertDataset
from src.models.densenet import DenseNetClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size: int, is_train: bool) -> transforms.Compose:
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Validation: loss + AUROC per label
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, device, label_names):
    model.eval()
    total_loss  = 0.0
    all_labels  = []
    all_probs   = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.vstack(all_labels)   # (N, num_classes)
    all_probs  = np.vstack(all_probs)

    # AUROC per label (skip labels with no positive examples)
    aurocs = []
    for i, name in enumerate(label_names):
        try:
            score = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            score = float("nan")
        aurocs.append((name, score))

    valid_scores = [s for _, s in aurocs if not np.isnan(s)]
    mean_auroc   = float(np.mean(valid_scores)) if valid_scores else float("nan")
    val_loss     = total_loss / len(loader.dataset)

    return val_loss, mean_auroc, aurocs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="REPLICATE/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Datasets and dataloaders
    # ------------------------------------------------------------------
    img_size    = cfg["training"]["img_size"]
    num_workers = cfg["training"]["num_workers"]
    batch_size  = cfg["training"]["batch_size"]
    image_root  = cfg["paths"]["image_root"]

    train_dataset = CheXpertDataset(
        manifest_path  = cfg["paths"]["train_manifest"],
        image_root_dir = image_root,
        transform      = build_transforms(img_size, is_train=True),
    )
    valid_dataset = CheXpertDataset(
        manifest_path  = cfg["paths"]["valid_manifest"],
        image_root_dir = image_root,
        transform      = build_transforms(img_size, is_train=False),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    label_names = train_dataset.target_cols
    num_classes = len(label_names)
    print(f"Labels ({num_classes}): {label_names}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = DenseNetClassifier(
        num_classes = num_classes,
        pretrained  = cfg["model"]["pretrained"],
        variant     = cfg["model"]["name"],
    ).to(device)

    # ------------------------------------------------------------------
    # Loss and optimizer
    # ------------------------------------------------------------------
    # BCEWithLogitsLoss: handles sigmoid + binary cross-entropy in one
    # numerically stable operation. Correct for multi-label classification.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["training"]["learning_rate"],
        betas        = tuple(cfg["training"]["betas"]),
        eps          = cfg["training"]["eps"],
        weight_decay = cfg["training"]["weight_decay"],
    )

    # ------------------------------------------------------------------
    # Output directory for checkpoints
    # ------------------------------------------------------------------
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    num_epochs = cfg["training"]["num_epochs"]
    best_auroc = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mean_auroc, aurocs = evaluate(model, valid_loader, criterion, device, label_names)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auroc={mean_auroc:.4f} | "
            f"{elapsed:.0f}s"
        )

        # Print per-label AUROC
        for name, score in aurocs:
            marker = " <-- best" if score == max(s for _, s in aurocs if not np.isnan(s)) else ""
            print(f"  {name:<35} {score:.4f}{marker}")

        # Save best checkpoint
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            ckpt_path  = output_dir / "best_model.pth"
            torch.save({
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc":  mean_auroc,
                "val_loss":   val_loss,
            }, ckpt_path)
            print(f"  ** New best saved -> {ckpt_path} (auroc={mean_auroc:.4f})")

    print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")
    print(f"Checkpoint saved to: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
