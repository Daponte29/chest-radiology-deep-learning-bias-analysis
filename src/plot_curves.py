"""
plot_curves.py — CLI entry point for training curve plots.

Usage:
    python -m src.plot_curves                          # val AUROC all models
    python -m src.plot_curves --metric val_loss        # single metric
    python -m src.plot_curves --loss                   # train vs val loss subplots
    python -m src.plot_curves --loss --out results/loss.png
"""

import argparse
from src.utils.plotting import plot_training_curves, plot_loss_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", action="store_true",
                        help="Plot train vs val loss subplots instead of a single metric")
    parser.add_argument("--metric", default="val_auroc_5",
                        choices=["val_auroc_5", "val_auroc_14", "train_loss", "val_loss"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.loss:
        plot_loss_curves(out=args.out)
    else:
        plot_training_curves(metric=args.metric, out=args.out)
