"""
train.py — CheXpert DenseNet Training Script
=============================================
What this script does:
    1. Load config (YAML)
    2. Set random seed for reproducibility
    3. Build train and validation datasets + dataloaders
       - Optional WeightedRandomSampler for rare-label balancing
       - Optional smoke-test mode (overfit N images to verify pipeline)
    4. Build DenseNet model (full backbone unfrozen, pretrained ImageNet weights)
    5. Define loss (Focal Loss), Adam optimizer, cosine LR scheduler
    6. Loop over epochs:
         - train for one epoch (optional AMP for speed on CUDA)
         - evaluate on validation set (loss + per-label AUROC)
         - step the LR scheduler
         - save best checkpoint (by 5-label competition AUROC)
         - early stop if val AUROC has not improved for `patience` epochs
         - append epoch metrics to training_history.parquet

Usage:
    python -m src.train --config src/configs/train_original.yaml
    python -m src.train --config src/configs/train_original.yaml --resume results/original/best_model.pth
    python -m src.train --config src/configs/train_original.yaml --smoke-test 50
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
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


def build_transforms(img_size: int, is_train: bool, aug_cfg: dict | None = None) -> transforms.Compose:
    steps = [transforms.Resize((img_size, img_size))]
    if is_train and aug_cfg:
        if aug_cfg.get("horizontal_flip", True):
            steps.append(transforms.RandomHorizontalFlip())
        jitter = aug_cfg.get("color_jitter", 0.15)
        if jitter > 0:
            steps.append(transforms.ColorJitter(brightness=jitter, contrast=jitter))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-label focal loss. gamma=0 reduces to standard BCE."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t    = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
        return ((1 - p_t) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# WeightedRandomSampler
# ---------------------------------------------------------------------------

def get_targets(dataset) -> np.ndarray:
    """Extract (N, C) targets array from any dataset type."""
    if isinstance(dataset, CheXpertDataset):
        return dataset.targets
    if isinstance(dataset, Subset):
        return dataset.dataset.targets[np.array(dataset.indices)]
    if isinstance(dataset, ConcatDataset):
        return np.vstack([get_targets(d) for d in dataset.datasets])
    raise TypeError(f"Cannot extract targets from {type(dataset)}")


def get_label_names(dataset) -> list:
    """Extract label names from any dataset type."""
    if isinstance(dataset, CheXpertDataset):
        return dataset.target_cols
    if isinstance(dataset, Subset):
        return dataset.dataset.target_cols
    if isinstance(dataset, ConcatDataset):
        return get_label_names(dataset.datasets[0])
    raise TypeError(f"Cannot get label names from {type(dataset)}")


def build_sampler(targets: np.ndarray) -> WeightedRandomSampler:
    """Return a sampler that upsamples rare-pathology images.

    Weight per sample = sum of inverse-frequency weights for its positive labels.
    """
    n             = len(targets)
    n_pos         = np.where(targets.sum(axis=0) == 0, 1, targets.sum(axis=0))
    label_weights = n / n_pos
    sample_weights = torch.from_numpy((targets * label_weights).sum(axis=1)).float()
    return WeightedRandomSampler(weights=sample_weights, num_samples=n, replacement=True)


def build_train_dataset(cfg: dict, t_cfg: dict, image_root: str,
                        target_cols, transform, seed: int):
    """Build the training dataset, optionally blending stylized + original images.

    blend_ratio controls the fraction of stylized images (default 1.0 = all stylized).
    Requires paths.train_parquet_blend to point to the original manifest.
    Total dataset size stays equal to len(stylized dataset) regardless of ratio.
    """
    train_parquet = cfg["paths"]["train_parquet"]
    blend_parquet = cfg["paths"].get("train_parquet_blend")
    blend_ratio   = float(t_cfg.get("blend_ratio", 1.0))

    if blend_ratio < 1.0 and blend_parquet:
        stylized_ds = CheXpertDataset(train_parquet, image_root, transform, target_cols)
        original_ds = CheXpertDataset(blend_parquet,  image_root, transform, target_cols)

        n_total    = len(stylized_ds)
        n_stylized = int(n_total * blend_ratio)
        n_original = n_total - n_stylized

        rng     = np.random.default_rng(seed)
        sty_idx = rng.choice(len(stylized_ds), size=n_stylized, replace=False).tolist()
        ori_idx = rng.choice(len(original_ds), size=min(n_original, len(original_ds)), replace=False).tolist()

        print(f"Blend       : {blend_ratio:.0%} stylized ({n_stylized:,}) + "
              f"{1-blend_ratio:.0%} original ({len(ori_idx):,})")
        return ConcatDataset([Subset(stylized_ds, sty_idx), Subset(original_ds, ori_idx)])

    return CheXpertDataset(train_parquet, image_root, transform, target_cols)


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

    all_labels = np.vstack(all_labels)
    all_probs  = np.vstack(all_probs)

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

    # comp_auroc = mean over all selected labels (used for checkpoint tracking)
    comp_auroc = mean_auroc

    return val_loss, mean_auroc, comp_auroc, aurocs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DenseNet on CheXpert")
    parser.add_argument("--config",      default="src/configs/train_original.yaml")
    parser.add_argument("--resume",      default=None, metavar="CKPT")
    parser.add_argument("--smoke-test",  default=None, type=int, metavar="N",
                        help="Overfit on N images to verify the pipeline end-to-end")
    # Quick overrides — these take precedence over whatever is in the YAML
    parser.add_argument("--lr",          default=None, type=float, help="Override learning_rate")
    parser.add_argument("--epochs",      default=None, type=int,   help="Override num_epochs")
    parser.add_argument("--blend-ratio", default=None, type=float, help="Override blend_ratio (0.0-1.0)")
    parser.add_argument("--loss",        default=None, choices=["focal", "bce"], help="Override loss")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides on top of YAML
    if args.lr          is not None: cfg["training"]["learning_rate"] = args.lr
    if args.epochs      is not None: cfg["training"]["num_epochs"]    = args.epochs
    if args.blend_ratio is not None: cfg["training"]["blend_ratio"]   = args.blend_ratio
    if args.loss        is not None: cfg["training"]["loss"]          = args.loss

    set_seed(cfg["seed"])

    smoke_test = args.smoke_test is not None
    if smoke_test:
        print(f"\n*** SMOKE TEST MODE — {args.smoke_test} images ***\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    t_cfg       = cfg["training"]
    img_size    = t_cfg["img_size"]
    num_workers = 0 if smoke_test else t_cfg["num_workers"]
    batch_size  = t_cfg["batch_size"]
    image_root  = cfg["paths"]["image_root"]

    target_cols = cfg.get("labels", None)
    aug_cfg     = t_cfg.get("augmentation", {})

    train_dataset = build_train_dataset(
        cfg, t_cfg, image_root, target_cols,
        build_transforms(img_size, is_train=True, aug_cfg=aug_cfg), cfg["seed"],
    )
    valid_dataset = CheXpertDataset(
        manifest_path  = cfg["paths"]["valid_parquet"],
        image_root_dir = image_root,
        transform      = build_transforms(img_size, is_train=False),
        target_cols    = target_cols,
    )

    # Smoke test: subset to N images, no sampler
    if smoke_test:
        n = min(args.smoke_test, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(n)))
        valid_dataset = Subset(valid_dataset, list(range(min(n, len(valid_dataset)))))

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    use_sampler = t_cfg.get("weighted_sampler", True) and not smoke_test
    if use_sampler:
        sampler = build_sampler(get_targets(train_dataset))
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    label_names  = get_label_names(train_dataset)
    num_classes  = len(label_names)
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Valid samples: {len(valid_dataset):,}")
    print(f"Labels ({num_classes}): {label_names}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = DenseNetClassifier(
        num_classes = num_classes,           # derived from dataset, not hardcoded
        pretrained  = cfg["model"]["pretrained"],
        variant     = cfg["model"]["name"],
    ).to(device)
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Loss — selected from config: focal | bce
    # ------------------------------------------------------------------
    loss_type = t_cfg.get("loss", "focal")
    if loss_type == "focal":
        gamma     = t_cfg.get("focal_gamma", 2.0)
        criterion = FocalLoss(gamma=gamma)
        print(f"Loss        : FocalLoss (gamma={gamma})")
    elif loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
        print(f"Loss        : BCEWithLogitsLoss")
    else:
        raise ValueError(f"Unknown loss '{loss_type}' — choose 'focal' or 'bce'")

    # ------------------------------------------------------------------
    # Optimizer — selected from config: adam | sgd
    # ------------------------------------------------------------------
    num_epochs   = t_cfg["num_epochs"]
    opt_type     = t_cfg.get("optimizer", "adam")
    lr           = t_cfg["learning_rate"]
    weight_decay = t_cfg["weight_decay"]

    if opt_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = lr,
            betas        = tuple(t_cfg.get("betas", [0.9, 0.999])),
            eps          = t_cfg.get("eps", 1e-8),
            weight_decay = weight_decay,
        )
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr           = lr,
            momentum     = t_cfg.get("momentum", 0.9),
            weight_decay = weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer '{opt_type}' — choose 'adam' or 'sgd'")
    print(f"Optimizer   : {opt_type}  (lr={lr})")

    # ------------------------------------------------------------------
    # LR Scheduler — selected from config: cosine | step
    # ------------------------------------------------------------------
    sched_type = t_cfg.get("scheduler", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=t_cfg.get("eta_min", 1e-6),
        )
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = t_cfg.get("step_size", 3),
            gamma     = t_cfg.get("step_gamma", 0.1),
        )
    else:
        raise ValueError(f"Unknown scheduler '{sched_type}' — choose 'cosine' or 'step'")
    print(f"Scheduler   : {sched_type}")

    use_amp = device.type == "cuda" and t_cfg.get("amp", True)
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"AMP enabled : {use_amp}")

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    patience        = t_cfg.get("early_stopping_patience", 3)
    epochs_no_improv = 0
    print(f"Early stop  : patience={patience} epochs")

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / "config.yaml")

    # ------------------------------------------------------------------
    # Optionally resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 1
    best_auroc  = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_auroc  = ckpt.get("val_auroc_5", ckpt.get("val_auroc", 0.0))
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from {args.resume}  (epoch {ckpt['epoch']}, auroc={best_auroc:.4f})")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history_path = output_dir / "training_history.parquet"
    history_rows = []
    wall_start   = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler,
        )
        val_loss, comp_auroc, _, aurocs = evaluate(
            model, valid_loader, criterion, device, label_names,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auroc={comp_auroc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        valid_scores = [s for _, s in aurocs if not np.isnan(s)]
        best_label   = max(valid_scores) if valid_scores else float("nan")
        for name, score in aurocs:
            marker = " <--" if score == best_label else ""
            print(f"  {name:<35} {score:.4f}{marker}")

        # Best checkpoint
        if comp_auroc > best_auroc:
            best_auroc       = comp_auroc
            epochs_no_improv = 0
            ckpt_path        = output_dir / "best_model.pth"
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc":            comp_auroc,
                "val_loss":             val_loss,
                "config":               cfg,
            }, ckpt_path)
            print(f"  ** New best -> {ckpt_path}  (auroc_5={comp_auroc:.4f})")
        else:
            epochs_no_improv += 1
            print(f"  No improvement ({epochs_no_improv}/{patience})")

        history_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_auroc":  round(comp_auroc, 6),
            "lr":         current_lr,
        })

        # Early stopping
        if epochs_no_improv >= patience:
            print(f"\nEarly stopping — no improvement for {patience} epochs.")
            break

    pl.DataFrame(history_rows).write_parquet(history_path)

    total_mins = (time.time() - wall_start) / 60
    print(f"\nTotal training time: {total_mins:.1f} min")
    print(f"Training complete. Best val AUROC (5-label): {best_auroc:.4f}")
    print(f"Best checkpoint : {output_dir / 'best_model.pth'}")
    print(f"Training history: {history_path}")

    if smoke_test:
        print("\nSmoke test passed — pipeline is working correctly.")
        return

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
            target_cols    = label_names,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=t_cfg["num_workers"], pin_memory=True,
            persistent_workers=(t_cfg["num_workers"] > 0),
        )

        _, test_mean_auroc, _, test_aurocs = evaluate(
            model, test_loader, criterion, device, label_names,
        )

        print(f"\n{'Label':<35} {'Test AUROC':>10}")
        print("-" * 47)
        for name, score in test_aurocs:
            print(f"  {name:<35} {score:.4f}")
        print("-" * 47)
        print(f"  {'Mean (all selected labels)':<35} {test_mean_auroc:.4f}")

        import json
        test_results = {
            "auroc_mean": round(test_mean_auroc, 6),
            "per_label": {name: round(score, 6) for name, score in test_aurocs},
        }
        results_path = output_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2)
        print(f"\nTest results saved to: {results_path}")


if __name__ == "__main__":
    main()
