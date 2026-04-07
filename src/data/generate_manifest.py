"""
Generate the three Parquet manifests used for training.

Split strategy
--------------
- valid.csv (Kaggle held-out set)  ->  test_manifest.parquet   (never touched during training)
- train.csv, 97% of patients       ->  train_manifest.parquet
- train.csv,  3% of patients       ->  valid_manifest.parquet  (used for epoch-level monitoring)

The train/valid split is patient-level: all studies from a given patient stay
in the same split, preventing any patient-level data leakage.

Usage
-----
    python -m src.data.generate_manifest
    # or directly:
    python src/data/generate_manifest.py
"""
import re
from pathlib import Path

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

_PATIENT_RE = re.compile(r"(patient\d+)")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_csv(input_csv_path: str) -> pl.DataFrame:
    """Load a raw CheXpert CSV and apply the standard ETL pipeline.

    Steps
    -----
    1. Filter for frontal views only (view*_frontal).
    2. For each label, record whether the original value was -1 in a
       ``{label}_uncertain`` column (Int8, 1 = was uncertain).
    3. Apply U-Zero policy: -1 -> 0, NaN -> 0, cast to Int8.

    Returns the cleaned DataFrame (does NOT save to disk).
    """
    print(f"  Loading {input_csv_path}...")
    df = pl.read_csv(input_csv_path)
    initial = len(df)

    df = df.filter(pl.col("Path").str.contains(r"view.*_frontal", literal=False))
    print(f"  Frontal filter: {initial} -> {len(df)} rows")

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


def _save(df: pl.DataFrame, output_path: Path, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    print(f"  Saved {label}: {len(df):,} rows -> {output_path}")


def _patient_id(path: str) -> str:
    m = _PATIENT_RE.search(path)
    return m.group(1) if m else path


def _patient_split(
    df: pl.DataFrame,
    valid_fraction: float = 0.03,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split df into (train, valid) at the patient level.

    Args:
        df:               Processed train DataFrame.
        valid_fraction:   Fraction of *patients* (not images) held out for validation.
        seed:             Random seed for reproducibility.

    Returns:
        (train_df, valid_df) — no patient appears in both splits.
    """
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

    n_total_patients = len(unique_patients)
    print(
        f"  Patient split: {n_total_patients:,} total patients -> "
        f"{n_total_patients - n_valid:,} train / {n_valid:,} valid  "
        f"({100*n_valid/n_total_patients:.1f}% valid)"
    )
    return train_df, valid_df


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def create_manifest(input_csv_path: str, output_parquet_path: str) -> pl.DataFrame:
    """Process a single CSV and save it as a Parquet manifest.

    Convenience wrapper used when no split is needed (e.g. the test set).
    """
    df = _process_csv(input_csv_path)
    _save(df, Path(output_parquet_path), label=Path(output_parquet_path).stem)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dir  = Path(__file__).resolve().parent   # src/data/
    data_dir  = base_dir / "1"                    # src/data/1/

    train_csv = data_dir / "train.csv"
    valid_csv = data_dir / "valid.csv"

    for csv in (train_csv, valid_csv):
        if not csv.exists():
            raise FileNotFoundError(f"Expected CSV not found: {csv}")

    # ------------------------------------------------------------------
    # 1. Kaggle valid.csv -> test_manifest.parquet  (held-out, never trained on)
    # ------------------------------------------------------------------
    print("\n[1/2] Processing valid.csv -> test_manifest.parquet")
    test_df = _process_csv(str(valid_csv))
    _save(test_df, base_dir / "test_manifest.parquet", "test")

    # ------------------------------------------------------------------
    # 2. train.csv -> patient-level 97/3 split
    #    -> train_manifest.parquet + valid_manifest.parquet
    # ------------------------------------------------------------------
    print("\n[2/2] Processing train.csv -> train / valid manifests (97% / 3% patient split)")
    full_train_df = _process_csv(str(train_csv))
    train_df, valid_df = _patient_split(full_train_df, valid_fraction=0.03, seed=42)

    _save(train_df, base_dir / "train_manifest.parquet", "train")
    _save(valid_df, base_dir / "valid_manifest.parquet", "valid")

    print("\nDone. Manifests written to", base_dir)
    print(f"  train : {len(train_df):>7,} rows")
    print(f"  valid : {len(valid_df):>7,} rows")
    print(f"  test  : {len(test_df):>7,} rows")
