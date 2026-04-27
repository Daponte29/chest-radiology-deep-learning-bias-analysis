import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("results")
EXPERIMENT_NAMES    = ["original", "gb", "ps", "ce", "pr"]

LABELS = {
    "original": "Baseline (original)",
    "gb":       "Gaussian Blur (shape)",
    "ps":       "Patch Shuffle (texture)",
    "ce":       "Canny Edge (shape)",
    "pr":       "Patch Rotation (texture)",
}

COLORS = {
    "original": "black",
    "gb":       "steelblue",
    "ps":       "cornflowerblue",
    "ce":       "tomato",
    "pr":       "salmon",
}


def _load_history(folder: Path) -> pl.DataFrame | None:
    """Load training history from parquet or CSV, normalising column names."""
    parquet = folder / "training_history.parquet"
    csv     = folder / "training_history.csv"

    if parquet.exists():
        return pl.read_parquet(parquet)
    if csv.exists():
        df = pl.read_csv(csv)
        # old CSV used 'val_auroc' — rename to match new column names
        if "val_auroc" in df.columns and "val_auroc_5" not in df.columns:
            df = df.rename({"val_auroc": "val_auroc_5"})
        return df
    return None


def plot_training_curves(
    metric: str = "val_auroc",
    out: str | None = None,
    show: bool = True,
    results_dir: str | Path | None = None,
) -> None:
    """Plot one metric across all 5 models on a single chart.

    metric:      column to plot — val_auroc, train_loss, val_loss
    out:         save path (default: <results_dir>/training_curves.png)
    results_dir: root results folder, e.g. results/nick (default: results/)
    """
    rdir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    out  = out or str(rdir / "training_curves.png")
    experiments = {name: rdir / name for name in EXPERIMENT_NAMES}

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, folder in experiments.items():
        df = _load_history(folder)
        if df is None:
            print(f"Skipping {name} — no training history found in {folder}")
            continue
        if metric not in df.columns:
            print(f"Skipping {name} — column '{metric}' not in history")
            continue
        ax.plot(
            df["epoch"].to_list(),
            df[metric].to_list(),
            label=LABELS[name],
            color=COLORS[name],
            linewidth=2,
            marker="o",
            markersize=4,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title(f"{metric.replace('_', ' ').title()} — All Experiments")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if show:
        plt.show()


def plot_loss_curves(
    out: str | None = None,
    show: bool = True,
    results_dir: str | Path | None = None,
) -> None:
    """Train vs val loss — one subplot per model.

    results_dir: root results folder, e.g. results/nick (default: results/)
    out:         save path (default: <results_dir>/loss_curves.png)
    """
    rdir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    out  = out or str(rdir / "loss_curves.png")
    experiments = {name: rdir / name for name in EXPERIMENT_NAMES}

    histories = {
        name: _load_history(folder)
        for name, folder in experiments.items()
    }
    available = {k: v for k, v in histories.items() if v is not None}

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, available.items()):
        epochs = df["epoch"].to_list()
        ax.plot(epochs, df["train_loss"].to_list(),
                label="Train", color=COLORS[name], linewidth=2, marker="o", markersize=3)
        ax.plot(epochs, df["val_loss"].to_list(),
                label="Val", color=COLORS[name], linewidth=2, linestyle="--",
                marker="s", markersize=3, alpha=0.7)
        ax.set_title(LABELS[name], fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Train vs Val Loss — All Experiments", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    if show:
        plt.show()
