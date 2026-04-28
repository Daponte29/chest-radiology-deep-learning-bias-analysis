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
from src.evaluate import build_transform, run_evaluation
from src.models.densenet import DenseNetClassifier
from src.utils.reliance import compute_reliance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE_ROOT  = "src/data/1"
IMG_SIZE    = 224
BATCH_SIZE  = 16
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

DEFAULT_RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_auroc(aurocs: list[tuple[str, float]]) -> float:
    scores = [s for _, s in aurocs if not np.isnan(s)]
    return float(np.mean(scores)) if scores else float("nan")


def read_ckpt_labels(ckpt_path: str, device: torch.device) -> tuple[list[str], str]:
    """Return (label_names, model_variant) from a saved checkpoint's config."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get("config", {})
    labels  = cfg.get("labels") or CheXpertDataset.DEFAULT_LABELS
    variant = cfg.get("model", {}).get("name", "densenet121")
    return labels, variant


def build_loader(parquet: str, label_names: list[str]) -> DataLoader:
    dataset = CheXpertDataset(
        manifest_path  = parquet,
        image_root_dir = IMAGE_ROOT,
        transform      = build_transform(IMG_SIZE),
        target_cols    = label_names,
    )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
    )


def load_model(ckpt_path: str, device: torch.device,
               label_names: list[str], variant: str) -> DenseNetClassifier:
    model = DenseNetClassifier(
        num_classes = len(label_names),
        pretrained  = False,
        variant     = variant,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded {ckpt_path}  (epoch {ckpt.get('epoch', '?')}, labels={len(label_names)})")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None,
                        help="Root results folder, e.g. results/nick. "
                             "Looks for checkpoints at <results-dir>/{gb,ps,ce,pr}/best_model.pth "
                             "and saves bias_eval output to <results-dir>/bias_eval/. "
                             "Defaults to results/")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    output_dir  = results_dir / "bias_eval"

    biased_models = {
        name: str(results_dir / name / "best_model.pth")
        for name in ["gb", "ps", "ce", "pr"]
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}\n")
    output_dir.mkdir(parents=True, exist_ok=True)

    auc_matrix:       dict[str, dict[str, float]] = {}
    per_label_results: dict[str, dict[str, dict]] = {}

    for model_name, ckpt_path in biased_models.items():
        print(f"\n{'='*55}")
        print(f"Model: {model_name}  |  {ckpt_path}")
        print(f"{'='*55}")

        if not Path(ckpt_path).exists():
            print("  SKIPPING — checkpoint not found")
            continue

        label_names, variant = read_ckpt_labels(ckpt_path, device)
        print(f"  Labels ({len(label_names)}): {label_names}")

        print("  Building test loaders...")
        loaders = {name: build_loader(parquet, label_names) for name, parquet in TEST_SETS.items()}

        model = load_model(ckpt_path, device, label_names, variant)
        auc_matrix[model_name]        = {}
        per_label_results[model_name] = {}

        for test_name, loader in loaders.items():
            print(f"  Running test set: {test_name}")
            aurocs = run_evaluation(model, loader, device, label_names)
            score  = mean_auroc(aurocs)
            auc_matrix[model_name][test_name]        = round(score, 6)
            per_label_results[model_name][test_name] = {n: round(s, 6) for n, s in aurocs}
            print(f"    mean_auroc = {score:.4f}")

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

    # Save
    auc_rows = [
        {"model": m, "test_set": t, "auroc": s}
        for m, tests in auc_matrix.items()
        for t, s in tests.items()
    ]
    pl.DataFrame(auc_rows).write_parquet(output_dir / "auc_matrix.parquet")

    with open(output_dir / "reliance.json", "w") as f:
        json.dump(reliance_results, f, indent=2)

    with open(output_dir / "per_label.json", "w") as f:
        json.dump(per_label_results, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print("  auc_matrix.parquet  — raw 4x5 AUC scores")
    print("  reliance.json       — matching/opposing ratios per model")
    print("  per_label.json      — full per-label breakdown")


if __name__ == "__main__":
    main()
