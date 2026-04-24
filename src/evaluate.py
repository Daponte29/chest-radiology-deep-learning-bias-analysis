"""
evaluate.py — Run a saved checkpoint on the test set.

Usage:
    python -m src.evaluate --config src/configs/train_original.yaml
    python -m src.evaluate --config src/configs/train_original.yaml --checkpoint results/original/best_model.pth
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.data.chexpert_dataset import CheXpertDataset
from src.models.densenet import DenseNetClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

COMPETITION_LABELS = {"Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def run_evaluation(model, loader, device, label_names):
    model.eval()
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_labels = np.vstack(all_labels)
    all_probs  = np.vstack(all_probs)

    aurocs = []
    for i, name in enumerate(label_names):
        try:
            score = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            score = float("nan")
        aurocs.append((name, score))

    return aurocs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("--config",     default="src/configs/train_original.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pth checkpoint. Defaults to best_model.pth in output_dir.")
    parser.add_argument("--split",      default="test", choices=["test", "valid"],
                        help="Which split to evaluate on (default: test)")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    t_cfg  = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve checkpoint path
    ckpt_path = args.checkpoint or str(Path(cfg["paths"]["output_dir"]) / "best_model.pth")
    print(f"Checkpoint: {ckpt_path}")

    # Resolve dataset split
    parquet_key = "test_parquet" if args.split == "test" else "valid_parquet"
    parquet_path = cfg["paths"][parquet_key]
    print(f"Split: {args.split}  ({parquet_path})")

    # Dataset & loader
    dataset = CheXpertDataset(
        manifest_path  = parquet_path, # The path to the test data needed for object initialization.
        image_root_dir = cfg["paths"]["image_root"],
        transform      = build_transform(t_cfg["img_size"]),
    )
    loader = DataLoader(
        dataset, batch_size=t_cfg["batch_size"],
        shuffle=False, num_workers=t_cfg["num_workers"], pin_memory=True,
        persistent_workers=(t_cfg["num_workers"] > 0),
    )

    label_names = dataset.target_cols
    num_classes = len(label_names)

    # Load model
    model = DenseNetClassifier(
        num_classes = num_classes,
        pretrained  = False,
        variant     = cfg["model"]["name"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    saved_epoch = ckpt.get("epoch", "?")
    print(f"Loaded weights from epoch {saved_epoch}\n")

    # Run
    aurocs = run_evaluation(model, loader, device, label_names)

    # Print results
    comp_scores = [s for n, s in aurocs if n in COMPETITION_LABELS and not np.isnan(s)]
    all_scores  = [s for _, s in aurocs if not np.isnan(s)]
    comp_mean   = float(np.mean(comp_scores)) if comp_scores else float("nan")
    all_mean    = float(np.mean(all_scores))  if all_scores  else float("nan")

    print(f"\n{'Label':<35} {'AUROC':>8}")
    print("-" * 45)
    for name, score in aurocs:
        tag = "  *" if name in COMPETITION_LABELS else ""
        print(f"  {name:<35} {score:.4f}{tag}")
    print("-" * 45)
    print(f"  {'Mean — 5 competition labels *':<35} {comp_mean:.4f}")
    print(f"  {'Mean — all 14 labels':<35} {all_mean:.4f}")
    print("\n  * = CheXpert leaderboard labels (Atelectasis, Cardiomegaly,")
    print("      Consolidation, Edema, Pleural Effusion)")

    # Save JSON
    output_dir   = Path(cfg["paths"]["output_dir"])
    results_path = output_dir / f"{args.split}_results.json"
    results = {
        "checkpoint":           ckpt_path,
        "epoch":                saved_epoch,
        "split":                args.split,
        "auroc_5_competition":  round(comp_mean, 6),
        "auroc_14_mean":        round(all_mean,  6),
        "per_label":            {name: round(score, 6) for name, score in aurocs},
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
