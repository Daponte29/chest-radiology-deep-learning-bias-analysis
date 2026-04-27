"""
plot_curves.py — CLI for training curve plots.

Usage:
    python -m src.plot_curves                                        # val AUROC, all models
    python -m src.plot_curves --results-dir results/nick            # Nick's results
    python -m src.plot_curves --loss --results-dir results/ed       # Ed's loss curves
    python -m src.plot_curves --metric train_loss --out my_plot.png
"""

import argparse
from src.utils.plotting import plot_training_curves, plot_loss_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None,
                        help="Root results folder, e.g. results/nick (default: results/)")
    parser.add_argument("--loss",   action="store_true",
                        help="Plot train vs val loss subplots instead of a single metric")
    parser.add_argument("--metric", default="val_auroc",
                        choices=["val_auroc", "train_loss", "val_loss"])
    parser.add_argument("--out",    default=None)
    args = parser.parse_args()

    if args.loss:
        plot_loss_curves(out=args.out, results_dir=args.results_dir)
    else:
        plot_training_curves(metric=args.metric, out=args.out, results_dir=args.results_dir)
