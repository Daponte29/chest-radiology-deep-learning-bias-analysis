"""
generate_manifests.py — Generate all Parquet manifests for training experiments.

Split strategy
--------------
- src/data/1/valid.csv  ->  test_manifest.parquet        (held-out, never trained on)
- src/data/1/train.csv  ->  train_manifest.parquet (97%)
                        ->  valid_manifest.parquet  (3%, same split for ALL experiments)

The SAME patient split is applied to every stylized variant so all experiments
share identical train / valid / test patient groups.

Manifests generated
-------------------
  Original (baseline):
    train_manifest.parquet
    valid_manifest.parquet      <- original images, used for ALL model checkpoint selection
    test_manifest.parquet       <- original images, used for final baseline evaluation

  Per stylization (gb / ps / ce / pr):
    train_manifest_{s}.parquet  <- stylized train images, used to train each biased model
    test_manifest_{s}.parquet   <- stylized test images, used for cross-evaluation AUROCs

  (stylized valid manifests are intentionally omitted — validation always uses originals)

Usage
-----
    python -m src.data.generate_manifests
"""

import re
import sys
from pathlib import Path

# Allow running directly as a script as well as via python -m
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEXPERT_LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
]

# Suffix -> human-readable name
STYLIZATIONS = {
    "_gb": "gaussian_blur",
    "_ps": "patch_shuffle",
    "_ce": "canny_edge",
    "_pr": "patch_rotation",
}

_PATIENT_RE = re.compile(r"(patient\d+)")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_csv(csv_path: Path) -> pl.DataFrame:
    """Load a raw CheXpert CSV and apply the standard ETL pipeline.

    1. Filter for frontal views only.
    2. Record whether original label was -1 in a {label}_uncertain column.
    3. Apply U-Zero policy: -1 -> 0, NaN -> 0, cast to Int8.
    """
    print(f"  Loading {csv_path.name} ...")
    df = pl.read_csv(str(csv_path))
    initial = len(df)

    df = df.filter(pl.col("Path").str.contains(r"view.*_frontal", literal=False))
    print(f"  Frontal filter: {initial:,} -> {len(df):,} rows")

    for label in CHEXPERT_LABELS:
        if label not in df.columns:
            continue
        df = df.with_columns(
            (pl.col(label) == -1).fill_null(False).cast(pl.Int8)
            .alias(f"{label}_uncertain")
        )
        df = df.with_columns(
            pl.col(label).fill_null(0).replace(-1, 0).cast(pl.Int8).alias(label)
        )

    return df


def _patient_id(path: str) -> str:
    m = _PATIENT_RE.search(path)
    return m.group(1) if m else path


def _patient_split(
    df: pl.DataFrame,
    valid_fraction: float = 0.03,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split df into (train, valid) at the patient level — no leakage."""
    df = df.with_columns(
        pl.col("Path")
        .map_elements(_patient_id, return_dtype=pl.Utf8)
        .alias("_patient_id")
    )

    unique_patients = df.select("_patient_id").unique().sort("_patient_id")
    n_valid = max(1, round(len(unique_patients) * valid_fraction))
    valid_patients = set(
        unique_patients.sample(n=n_valid, seed=seed)["_patient_id"].to_list()
    )

    train_df = df.filter(~pl.col("_patient_id").is_in(valid_patients)).drop("_patient_id")
    valid_df = df.filter( pl.col("_patient_id").is_in(valid_patients)).drop("_patient_id")

    n_total = len(unique_patients)
    print(
        f"  Patient split: {n_total:,} patients -> "
        f"{n_total - n_valid:,} train / {n_valid:,} valid "
        f"({100 * n_valid / n_total:.1f}% valid)"
    )
    return train_df, valid_df


def _stylize_paths(df: pl.DataFrame, suffix: str) -> pl.DataFrame:
    """Return a copy of df with Path column pointing to stylized image filenames."""
    return df.with_columns(
        pl.col("Path").str.replace(r"\.jpg$", suffix + ".jpg").alias("Path")
    )


def _save(df: pl.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(str(path))
    print(f"    {label:<35} {len(df):>7,} rows  ->  {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent       # src/data/
    data_dir = base_dir / "1"                        # src/data/1/

    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"

    for csv in (train_csv, valid_csv):
        if not csv.exists():
            raise FileNotFoundError(f"Expected CSV not found: {csv}")

    # ------------------------------------------------------------------
    # 1. Process source CSVs
    # ------------------------------------------------------------------
    print("\n[1/3] Processing valid.csv (test set) ...")
    test_df = _process_csv(valid_csv)

    print("\n[2/3] Processing train.csv + patient-level 97/3 split ...")
    full_train_df      = _process_csv(train_csv)
    train_df, valid_df = _patient_split(full_train_df, valid_fraction=0.03, seed=42)

    # ------------------------------------------------------------------
    # 2. Save original manifests
    # ------------------------------------------------------------------
    print("\n[3/3] Writing manifests ...")
    print("  Original:")
    _save(train_df, base_dir / "train_manifest.parquet",  "train")
    _save(valid_df, base_dir / "valid_manifest.parquet",  "valid (checkpoint monitor)")
    _save(test_df,  base_dir / "test_manifest.parquet",   "test")

    # ------------------------------------------------------------------
    # 3. Save stylized manifests (same split, suffixed paths)
    # ------------------------------------------------------------------
    for suffix, name in STYLIZATIONS.items():
        print(f"\n  {name} ({suffix}):")
        _save(_stylize_paths(train_df, suffix),
              base_dir / f"train_manifest{suffix}.parquet",
              f"train{suffix}")
        _save(_stylize_paths(test_df, suffix),
              base_dir / f"test_manifest{suffix}.parquet",
              f"test{suffix}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "-" * 55)
    print(f"  train          : {len(train_df):>7,} rows (original)")
    print(f"  valid          : {len(valid_df):>7,} rows (original, all experiments)")
    print(f"  test           : {len(test_df):>7,} rows (original)")
    for suffix, name in STYLIZATIONS.items():
        print(f"  train{suffix:<4}       : {len(train_df):>7,} rows ({name})")
        print(f"  test{suffix:<4}        : {len(test_df):>7,} rows ({name})")
    print(f"\nAll manifests written to: {base_dir}")
