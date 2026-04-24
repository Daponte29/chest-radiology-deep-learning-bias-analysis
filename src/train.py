"""
train.py — CheXpert DenseNet Training Script
=============================================
Goal: train a DenseNet on CheXpert chest X-rays for multi-label pathology classification.

What this script does, step by step:
    1. Load config (YAML)
    2. Set random seed for reproducibility
    3. Build train and validation datasets + dataloaders
    4. Build DenseNet model (full backbone unfrozen, pretrained ImageNet weights)
    5. Define loss (BCEWithLogitsLoss), Adam optimizer, cosine LR scheduler
    6. Loop over epochs:
         - train for one epoch (optional AMP for speed on CUDA)
         - evaluate on validation set (loss + per-label AUROC)
         - step the LR scheduler
         - save best checkpoint (by mean AUROC)
         - append epoch metrics to training_history.parquet

Usage:
    python -m src.train --config src/configs/train_original.yaml
    python -m src.train --config src/configs/train_original.yaml --resume results/original/best_model.pth
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows: prevents OMP duplicate-library crash

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

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

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Validation: loss + AUROC per label
# ---------------------------------------------------------------------------

def evaluate(model, loader, criterion, device, label_names):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False, dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.vstack(all_labels)  # (N, num_classes)
    all_probs  = np.vstack(all_probs)

    # Per-label AUROC; skip labels where only one class is present in the batch
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

    # CheXpert benchmark: mean AUROC over the 5 official competition labels
    competition_labels = {"Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"}
    comp_scores = [s for n, s in aurocs if n in competition_labels and not np.isnan(s)]
    comp_auroc  = float(np.mean(comp_scores)) if comp_scores else float("nan")

    return val_loss, mean_auroc, comp_auroc, aurocs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DenseNet on CheXpert")
    parser.add_argument(
        "--config", default="src/configs/train_original.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", default=None, metavar="CKPT",
        help="Path to checkpoint (.pth) to resume training from",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    # ------------------------------------------------------------------
    # Datasets and dataloaders
    # ------------------------------------------------------------------
    t_cfg       = cfg["training"]
    img_size    = t_cfg["img_size"]
    num_workers = t_cfg["num_workers"]
    batch_size  = t_cfg["batch_size"]
    image_root  = cfg["paths"]["image_root"]

    train_dataset = CheXpertDataset(
        manifest_path  = cfg["paths"]["train_parquet"],
        image_root_dir = image_root,
        transform      = build_transforms(img_size, is_train=True),
    )
    valid_dataset = CheXpertDataset(
        manifest_path  = cfg["paths"]["valid_parquet"],
        image_root_dir = image_root,
        transform      = build_transforms(img_size, is_train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,  num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    label_names = train_dataset.target_cols
    num_classes = len(label_names)
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Valid samples: {len(valid_dataset):,}")
    print(f"Labels ({num_classes}): {label_names}")

    # ------------------------------------------------------------------
    # Model  (full backbone unfrozen — all params trainable)
    # ------------------------------------------------------------------
    model = DenseNetClassifier(
        num_classes = num_classes,
        pretrained  = cfg["model"]["pretrained"],
        variant     = cfg["model"]["name"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters  : {total_params:,}")

    # ------------------------------------------------------------------
    # Loss, optimizer, cosine LR scheduler
    # ------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()

    betas     = tuple(t_cfg.get("betas", [0.9, 0.999]))
    eps       = t_cfg.get("eps", 1e-8)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = t_cfg["learning_rate"],
        betas        = betas,
        eps          = eps,
        weight_decay = t_cfg["weight_decay"],
    )

    num_epochs = t_cfg["num_epochs"]
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )

    # Mixed-precision scaler (CUDA only; skipped on CPU)
    use_amp = device.type == "cuda" and t_cfg.get("amp", True)
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"AMP enabled : {use_amp}")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optionally resume from a previous checkpoint
    # ------------------------------------------------------------------
    start_epoch = 1
    best_auroc  = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auroc  = ckpt.get("val_auroc_5", ckpt.get("val_auroc", 0.0))
        # Advance scheduler state to match the resumed epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from {args.resume}  (epoch {ckpt['epoch']}, auroc={best_auroc:.4f})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history_path = output_dir / "training_history.parquet"
    history_rows = []

    wall_start = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler,
        )
        val_loss, mean_auroc, comp_auroc, aurocs = evaluate(
            model, valid_loader, criterion, device, label_names,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"auroc_5={comp_auroc:.4f} | "   # CheXpert benchmark (5 labels)
            f"auroc_14={mean_auroc:.4f} | "  # all 14 labels
            f"lr={current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        # Per-label AUROC breakdown
        valid_scores = [s for _, s in aurocs if not np.isnan(s)]
        best_label   = max(valid_scores) if valid_scores else float("nan")
        for name, score in aurocs:
            marker = " <--" if score == best_label else ""
            print(f"  {name:<35} {score:.4f}{marker}")

        # Save best checkpoint (tracked by 5-label competition AUROC)
        if comp_auroc > best_auroc:
            best_auroc = comp_auroc
            ckpt_path  = output_dir / "best_model.pth"
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc_5":          comp_auroc,
                "val_auroc_14":         mean_auroc,
                "val_loss":             val_loss,
                "config":               cfg,
            }, ckpt_path)
            print(f"  ** New best -> {ckpt_path}  (auroc_5={comp_auroc:.4f})")

        history_rows.append({
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_loss, 6),
            "val_auroc_5":  round(comp_auroc, 6),
            "val_auroc_14": round(mean_auroc, 6),
            "lr":           current_lr,
        })

    pl.DataFrame(history_rows).write_parquet(history_path)
    print(f"Training history: {history_path}")

    total_mins = (time.time() - wall_start) / 60
    print(f"\nTotal training time: {total_mins:.1f} min")
    print(f"\nTraining complete. Best val AUROC (5-label benchmark): {best_auroc:.4f}")
    print(f"Best checkpoint : {output_dir / 'best_model.pth'}")
    print(f"Training history: {history_path}")

    # ------------------------------------------------------------------
    # Final test-set evaluation using the best checkpoint
    # ------------------------------------------------------------------
    test_parquet = cfg["paths"].get("test_parquet")
    if test_parquet and Path(test_parquet).exists():
        print("\n--- Final test-set evaluation (best checkpoint) ---")
        best_ckpt = torch.load(output_dir / "best_model.pth", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_dataset = CheXpertDataset(
            manifest_path  = test_parquet,
            image_root_dir = image_root,
            transform      = build_transforms(img_size, is_train=False),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

        _, test_mean_auroc, test_comp_auroc, test_aurocs = evaluate(
            model, test_loader, criterion, device, label_names,
        )

        competition_labels = {"Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"}
        print(f"\n{'Label':<35} {'Test AUROC':>10}")
        print("-" * 47)
        for name, score in test_aurocs:
            tag = " *" if name in competition_labels else ""
            print(f"  {name:<35} {score:.4f}{tag}")
        print("-" * 47)
        print(f"  {'Mean (5 competition labels)*':<35} {test_comp_auroc:.4f}")
        print(f"  {'Mean (all 14 labels)':<35} {test_mean_auroc:.4f}")
        print("  * = CheXpert leaderboard labels")

        # Save test results to JSON for notebooks / reporting
        import json
        test_results = {
            "auroc_5_competition": round(test_comp_auroc, 6),
            "auroc_14_mean":       round(test_mean_auroc, 6),
            "per_label": {name: round(score, 6) for name, score in test_aurocs},
        }
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest results saved to: {results_path}")
    else:
        print("\nNo test_parquet found in config — skipping final test evaluation.")


if __name__ == "__main__":
    main()
