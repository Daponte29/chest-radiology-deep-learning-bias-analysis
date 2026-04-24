"""
bias_eval.py — Run all 4 biased models against all 5 test sets.

Produces a 4x5 AUC matrix, then computes matching/opposing reliance ratios
for each model. Results saved to results/bias_eval/.

Usage:
    python -m src.bias_eval
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from src.data.chexpert_dataset import CheXpertDataset
from src.evaluate import build_transform, run_evaluation, COMPETITION_LABELS
from src.models.densenet import DenseNetClassifier
from src.utils.reliance import compute_reliance

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

IMAGE_ROOT = "src/data/1"
IMG_SIZE   = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

BIASED_MODELS = {
    "gb": "results/gb/best_model.pth",
    "ps": "results/ps/best_model.pth",
    "ce": "results/ce/best_model.pth",
    "pr": "results/pr/best_model.pth",
}

TEST_SETS = {
    "original": "src/data/test_manifest.parquet",
    "gb":       "src/data/test_manifest_gb.parquet",
    "ps":       "src/data/test_manifest_ps.parquet",
    "ce":       "src/data/test_manifest_ce.parquet",
    "pr":       "src/data/test_manifest_pr.parquet",
}

LABEL_NAMES = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "No Finding", "Support Devices",
]

OUTPUT_DIR = Path("results/bias_eval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auroc5(aurocs: list[tuple[str, float]]) -> float:
    scores = [s for n, s in aurocs if n in COMPETITION_LABELS and not np.isnan(s)]
    return float(np.mean(scores)) if scores else float("nan")


def build_loader(parquet: str) -> DataLoader:
    dataset = CheXpertDataset(
        manifest_path  = parquet,
        image_root_dir = IMAGE_ROOT,
        transform      = build_transform(IMG_SIZE),
    )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


def load_model(ckpt_path: str, device: torch.device) -> DenseNetClassifier:
    model = DenseNetClassifier(
        num_classes = len(LABEL_NAMES),
        pretrained  = False,
        variant     = "densenet121",
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {ckpt_path}  (epoch {ckpt.get('epoch', '?')})")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-build all test loaders once (no need to reload per model)
    print("Building test loaders...")
    loaders = {name: build_loader(parquet) for name, parquet in TEST_SETS.items()}

    # AUC matrix: model -> {test_set -> auroc_5}
    auc_matrix: dict[str, dict[str, float]] = {}
    per_label_results: dict[str, dict[str, dict]] = {}

    for model_name, ckpt_path in BIASED_MODELS.items():
        print(f"\n{'='*55}")
        print(f"Model: {model_name}  |  {ckpt_path}")
        print(f"{'='*55}")

        if not Path(ckpt_path).exists():
            print(f"  SKIPPING — checkpoint not found")
            continue

        model = load_model(ckpt_path, device)
        auc_matrix[model_name] = {}
        per_label_results[model_name] = {}

        for test_name, loader in loaders.items():
            print(f"  Running test set: {test_name}")
            aurocs = run_evaluation(model, loader, device, LABEL_NAMES)
            score  = auroc5(aurocs)
            auc_matrix[model_name][test_name] = round(score, 6)
            per_label_results[model_name][test_name] = {
                n: round(s, 6) for n, s in aurocs
            }
            print(f"    auroc_5 = {score:.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Reliance ratios
    print(f"\n{'='*55}")
    print("Reliance ratios  (stylized_AUC / original_AUC)")
    print(f"{'='*55}")

    reliance_results = []
    for model_name, aucs in auc_matrix.items():
        r = compute_reliance(model_name, aucs)
        reliance_results.append(r)
        print(f"\n  Model: {model_name}  ({r['bias_type']} bias)")
        print(f"    Baseline AUC on original test : {r['auroc_original_test']:.4f}")
        print(f"    Matching  reliance {r['matching']}")
        print(f"    Opposing  reliance {r['opposing']}")
        print(f"    Mean matching : {r['mean_matching_reliance']:.4f}")
        print(f"    Mean opposing : {r['mean_opposing_reliance']:.4f}")

    # Save outputs
    auc_rows = [
        {"model": m, "test_set": t, "auroc_5": s}
        for m, tests in auc_matrix.items()
        for t, s in tests.items()
    ]
    pl.DataFrame(auc_rows).write_parquet(OUTPUT_DIR / "auc_matrix.parquet")

    with open(OUTPUT_DIR / "reliance.json", "w") as f:
        json.dump(reliance_results, f, indent=2)

    with open(OUTPUT_DIR / "per_label.json", "w") as f:
        json.dump(per_label_results, f, indent=2)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print("  auc_matrix.parquet  — raw 4x5 AUC scores")
    print("  reliance.json       — matching/opposing ratios per model")
    print("  per_label.json      — full per-label breakdown")


if __name__ == "__main__":
    main()
