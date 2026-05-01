"""
plot.py — Unified plotting CLI for the shape-vs-texture bias project.

All three subcommands auto-discover configs from the archive directory so no
hardcoded paths or config-specific scripts are needed.

Subcommands
-----------
  curves    Training curves (val AUROC + loss) for one results directory
  compare   Multi-config AUROC bar chart + per-label heatmaps (all discovered configs)
  reliance  Matching vs opposing reliance (all discovered configs with bias_eval output)

Usage
-----
  python -m src.plot curves   --results-dir src/configs/archive_results_configs/config_1/results
  python -m src.plot compare  [--archive src/configs/archive_results_configs]
  python -m src.plot reliance [--archive src/configs/archive_results_configs]
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVE_DEFAULT = Path("src/configs/archive_results_configs")

MODELS = ["original", "gb", "ps", "ce", "pr"]
MODEL_LABELS = {
    "original": "Original",
    "gb":       "Gaussian Blur\n(texture bias)",
    "ps":       "Patch Shuffle\n(texture bias)",
    "ce":       "Canny Edge\n(shape bias)",
    "pr":       "Patch Rotation\n(shape bias)",
}

LABEL_ORDER = [
    "Support Devices", "No Finding", "Pleural Effusion", "Pleural Other",
    "Lung Opacity", "Edema", "Consolidation", "Atelectasis",
    "Pneumothorax", "Cardiomegaly", "Pneumonia",
    "Enlarged Cardiomediastinum", "Lung Lesion", "Fracture",
]

_PALETTE = ["#B07FE0", "#E07B39", "#4878CF", "#6ACC65",
            "#E84393", "#FFD700", "#00CED1", "#FF6B35"]

# Correct bias-type definitions (guards against stale reliance.json labels)
_CORRECT_BIAS = {"gb": "texture", "ps": "texture", "ce": "shape", "pr": "shape"}
_TEXTURE, _SHAPE = {"gb", "ps"}, {"ce", "pr"}

# Dark theme
DARK_BG    = "#0f1117"
PANEL_BG   = "#1a1d27"
TEXT_COLOR = "#e8eaf0"
GRID_COLOR = "#2a2d3a"
NAN_COLOR  = "#2e2e3a"

_cmap = LinearSegmentedColormap.from_list(
    "dark_rg", ["#8b0000", "#cc4400", "#e8b84b", "#4878CF", "#1a6e2e"]
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def discover_configs(archive: Path) -> list[dict]:
    """Return sorted list of config metadata dicts for every config_* folder."""
    configs = []
    for folder in sorted(archive.glob("config_*")):
        results_dir = folder / "results"
        if not results_dir.is_dir():
            continue
        meta = _parse_yaml_meta(folder)
        n = folder.name  # e.g. "config_1"
        idx = int(n.split("_")[-1]) - 1 if n.split("_")[-1].isdigit() else len(configs)
        configs.append({
            "key":         n,
            "label":       f"Config {idx + 1}",
            "results_dir": results_dir,
            "color":       _PALETTE[idx % len(_PALETTE)],
            **meta,
        })
    return configs


def _parse_yaml_meta(config_dir: Path) -> dict:
    """Extract display metadata from train_original.yaml."""
    yaml_path = config_dir / "train_original.yaml"
    if not yaml_path.exists():
        return {"loss_str": "?", "sampler": "?", "lr_str": "?", "wd_str": "?", "n_labels": "?"}
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    t   = cfg.get("training", {})
    loss    = t.get("loss", "bce")
    gamma   = t.get("focal_gamma", 2.0)
    loss_str = f"Focal γ={gamma}" if loss == "focal" else "BCE"
    lr  = t.get("learning_rate", 0)
    wd  = t.get("weight_decay", 0)
    return {
        "loss_str":  loss_str,
        "sampler":   "Yes" if t.get("weighted_sampler", False) else "No",
        "lr_str":    f"{lr:.0e}",
        "wd_str":    f"{wd:.0e}",
        "n_labels":  str(len(cfg.get("labels") or []) or 14),
    }


def _load_test_results(results_dir: Path) -> dict[str, dict]:
    """Load test_results.json for each of the 5 model variants."""
    out = {}
    for model in MODELS:
        p = results_dir / model / "test_results.json"
        if p.exists():
            with open(p) as f:
                out[model] = json.load(f)
    return out


def _load_reliance(results_dir: Path) -> dict[str, dict] | None:
    """Load and correct bias_eval/reliance.json. Returns None if not found."""
    p = results_dir / "bias_eval" / "reliance.json"
    if not p.exists():
        return None
    with open(p) as f:
        raw = json.load(f)

    result = {}
    for entry in raw:
        model = entry["model"]
        bias  = _CORRECT_BIAS[model]
        all_r: dict[str, float] = {}
        all_r.update(entry.get("matching", {}))
        all_r.update(entry.get("opposing", {}))
        match_keys = sorted(_TEXTURE if bias == "texture" else _SHAPE)
        opp_keys   = sorted(_SHAPE   if bias == "texture" else _TEXTURE)
        matching   = {k: all_r[k] for k in match_keys if k in all_r}
        opposing   = {k: all_r[k] for k in opp_keys   if k in all_r}
        result[model] = {
            "bias_type":     bias,
            "mean_matching": float(np.mean(list(matching.values()))) if matching else np.nan,
            "mean_opposing": float(np.mean(list(opposing.values()))) if opposing else np.nan,
            "matching":      matching,
            "opposing":      opposing,
            "auroc_original": entry["auroc_original_test"],
        }
    return result


# ---------------------------------------------------------------------------
# Subcommand: curves
# ---------------------------------------------------------------------------

def cmd_curves(args: argparse.Namespace) -> None:
    """Plot val AUROC and loss curves from training history parquet files."""
    rdir = Path(args.results_dir)
    out  = Path(args.out) if args.out else rdir.parent / "training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    loaded = 0
    for name in MODELS:
        hist_path = rdir / name / "training_history.parquet"
        if not hist_path.exists():
            continue
        df = pl.read_parquet(hist_path)
        color = _PALETTE[MODELS.index(name) % len(_PALETTE)]
        label = MODEL_LABELS[name].replace("\n", " ")
        kw = dict(color=color, linewidth=2, marker="o", markersize=3, label=label)

        if "val_auroc" in df.columns:
            axes[0].plot(df["epoch"].to_list(), df["val_auroc"].to_list(), **kw)
        if "val_loss" in df.columns:
            axes[1].plot(df["epoch"].to_list(), df["val_loss"].to_list(), **kw)
        loaded += 1

    if loaded == 0:
        print(f"No training_history.parquet files found under {rdir}")
        return

    for ax, title, ylabel in [
        (axes[0], "Validation AUROC",  "AUROC"),
        (axes[1], "Validation Loss",   "Loss"),
    ]:
        ax.set_facecolor(PANEL_BG)
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR)
        ax.legend(framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8)

    fig.suptitle(f"Training Curves — {rdir.parent.name}",
                 color=TEXT_COLOR, fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"Saved: {out}")
    if not args.no_show:
        plt.show()


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> None:
    """Multi-config AUROC bar chart + per-label heatmaps."""
    archive = Path(args.archive)
    configs = discover_configs(archive)
    if not configs:
        raise SystemExit(f"No config_* folders found under {archive}")

    # Load results
    for c in configs:
        c["data"] = _load_test_results(c["results_dir"])
        if not c["data"]:
            print(f"  WARNING: no test_results.json found for {c['key']}, skipping")
    configs = [c for c in configs if c.get("data")]

    n_configs = len(configs)
    n_models  = len(MODELS)

    cmap_nan = _cmap.copy()
    cmap_nan.set_bad(color=NAN_COLOR)

    fig = plt.figure(figsize=(22, 22))
    fig.patch.set_facecolor(DARK_BG)
    gs  = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35,
                   top=0.92, bottom=0.05, left=0.07, right=0.97,
                   height_ratios=[0.24, 0.38, 0.38])

    # ── Panel 0: Config summary table ────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0, :])
    ax_t.set_facecolor(PANEL_BG)
    ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1); ax_t.axis("off")

    col_headers = ["", "Loss Function", "Sampler", "Learning Rate", "Weight Decay", "Labels"]
    col_x       = [0.01, 0.22, 0.38, 0.51, 0.63, 0.74]
    row_ys      = np.linspace(0.82, 0.12, n_configs)
    row_height  = (0.82 - 0.12) / max(n_configs, 1)

    hdr_kw = dict(fontsize=9, color="#aab0c4", fontweight="bold",
                  ha="left", va="center", transform=ax_t.transAxes)
    for hdr, x in zip(col_headers, col_x):
        ax_t.text(x, 0.93, hdr, **hdr_kw)
    ax_t.axhline(0.87, color=GRID_COLOR, linewidth=0.8)

    for cfg, y in zip(configs, row_ys):
        color = cfg["color"]
        row = [
            f"{cfg['label']}  —  {cfg['loss_str']}, Sampler={cfg['sampler']}",
            cfg["loss_str"], cfg["sampler"], cfg["lr_str"], cfg["wd_str"], cfg["n_labels"],
        ]
        ax_t.add_patch(mpatches.FancyBboxPatch(
            (col_x[0] - 0.005, y - row_height * 0.42), 0.195, row_height * 0.84,
            boxstyle="round,pad=0.01", facecolor=color, alpha=0.18,
            transform=ax_t.transAxes, zorder=2,
        ))
        for val, x in zip(row, col_x):
            is_name = (x == col_x[0])
            ax_t.text(x, y, val,
                      fontsize=8.5 if not is_name else 9,
                      color=color if is_name else TEXT_COLOR,
                      fontweight="bold" if is_name else "normal",
                      ha="left", va="center", transform=ax_t.transAxes)
        ax_t.axhline(y - row_height * 0.46, color=GRID_COLOR, linewidth=0.5)

    ax_t.set_title("Training Configuration Comparison  —  Experiment Grid",
                   color=TEXT_COLOR, fontsize=13, fontweight="bold", loc="left", pad=8)

    # ── Panel 1: Grouped bar chart ────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor(PANEL_BG)
    ax_bar.tick_params(colors=TEXT_COLOR)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(GRID_COLOR)

    bar_w     = 0.17
    group_gap = 0.12
    x_base    = np.arange(n_models) * (n_configs * bar_w + group_gap + 0.08)

    for ci, cfg in enumerate(configs):
        means = [cfg["data"].get(m, {}).get("auroc_mean", np.nan) for m in MODELS]
        x_pos = x_base + ci * bar_w
        bars  = ax_bar.bar(x_pos, means, width=bar_w, color=cfg["color"],
                           alpha=0.88, label=cfg["label"], zorder=3,
                           edgecolor="white", linewidth=0.3)
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=6.8, color=TEXT_COLOR, fontweight="bold")

    center_offset = (n_configs * bar_w) / 2 - bar_w / 2
    ax_bar.set_xticks(x_base + center_offset)
    ax_bar.set_xticklabels([MODEL_LABELS[m] for m in MODELS], color=TEXT_COLOR, fontsize=9.5)
    ax_bar.set_ylabel("Mean AUROC (test set)", color=TEXT_COLOR, fontsize=10)
    ax_bar.set_ylim(0.55, 0.93)
    ax_bar.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax_bar.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax_bar.set_title("Mean AUROC by Model Variant and Training Config",
                     color=TEXT_COLOR, fontsize=12, fontweight="bold", loc="left")
    ax_bar.legend(framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=8.5, loc="lower right")

    # ── Panel 2: Per-label heatmap — original model across all configs ────────
    ax_h1 = fig.add_subplot(gs[2, 0])
    ax_h1.set_facecolor(PANEL_BG)

    heat  = []
    xlbls = []
    for cfg in configs:
        xlbls.append(f"{cfg['label']}\n{cfg['loss_str']}, S={cfg['sampler']}")
        per = cfg["data"].get("original", {}).get("per_label", {})
        heat.append([per.get(lbl, np.nan) for lbl in LABEL_ORDER])

    heat_arr = np.array(heat, dtype=float).T
    im = ax_h1.imshow(np.ma.masked_invalid(heat_arr), aspect="auto",
                      cmap=cmap_nan, vmin=0.0, vmax=1.0)
    ax_h1.set_xticks(range(n_configs))
    ax_h1.set_xticklabels(xlbls, color=TEXT_COLOR, fontsize=7.5)
    ax_h1.set_yticks(range(len(LABEL_ORDER)))
    ax_h1.set_yticklabels(LABEL_ORDER, color=TEXT_COLOR, fontsize=8)
    ax_h1.tick_params(colors=TEXT_COLOR)
    for i in range(len(LABEL_ORDER)):
        for j in range(n_configs):
            v = heat_arr[i, j]
            if np.isnan(v):
                ax_h1.text(j, i, "N/A", ha="center", va="center",
                           fontsize=7, color="#666888", fontweight="bold")
            else:
                ax_h1.text(j, i, f"{v:.3f}", ha="center", va="center",
                           fontsize=7, color="white" if v < 0.65 else "#111",
                           fontweight="bold")
    cb = fig.colorbar(im, ax=ax_h1, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cb.set_label("AUROC", color=TEXT_COLOR, fontsize=9)
    ax_h1.set_title("Per-Label AUROC — Original Model (all configs)",
                    color=TEXT_COLOR, fontsize=10, fontweight="bold", loc="left")

    # ── Panel 3: Best config per-label across all 5 model variants ────────────
    ax_h2 = fig.add_subplot(gs[2, 1])
    ax_h2.set_facecolor(PANEL_BG)

    best_cfg   = max(configs, key=lambda c: c["data"].get("original", {}).get("auroc_mean", 0))
    best_labels = [l for l in LABEL_ORDER
                   if not np.isnan(best_cfg["data"].get("original", {})
                                   .get("per_label", {}).get(l, np.nan))]

    heat2 = []
    for model in MODELS:
        per = best_cfg["data"].get(model, {}).get("per_label", {})
        heat2.append([per.get(l, np.nan) for l in best_labels])

    heat2_arr = np.array(heat2, dtype=float).T
    im2 = ax_h2.imshow(np.ma.masked_invalid(heat2_arr), aspect="auto",
                       cmap=cmap_nan, vmin=0.0, vmax=1.0)
    ax_h2.set_xticks(range(n_models))
    ax_h2.set_xticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODELS],
                           color=TEXT_COLOR, fontsize=8, rotation=15, ha="right")
    ax_h2.set_yticks(range(len(best_labels)))
    ax_h2.set_yticklabels(best_labels, color=TEXT_COLOR, fontsize=8)
    ax_h2.tick_params(colors=TEXT_COLOR)
    for i in range(len(best_labels)):
        for j in range(n_models):
            v = heat2_arr[i, j]
            if np.isnan(v):
                ax_h2.text(j, i, "N/A", ha="center", va="center",
                           fontsize=7, color="#666888", fontweight="bold")
            else:
                ax_h2.text(j, i, f"{v:.3f}", ha="center", va="center",
                           fontsize=7, color="white" if v < 0.65 else "#111",
                           fontweight="bold")
    cb2 = fig.colorbar(im2, ax=ax_h2, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
    cb2.set_label("AUROC", color=TEXT_COLOR, fontsize=9)
    ax_h2.set_title(f"{best_cfg['label']} — Per-Label AUROC Across All 5 Model Variants\n"
                    f"(best-performing config, {best_cfg['n_labels']} labels)",
                    color=TEXT_COLOR, fontsize=10, fontweight="bold", loc="left")

    fig.suptitle("Shape vs. Texture Bias — DenseNet121 on CheXpert  |  Training Config Ablation Study",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.975)

    out = archive / "config_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"Saved: {out}")
    if not args.no_show:
        plt.show()


# ---------------------------------------------------------------------------
# Subcommand: reliance
# ---------------------------------------------------------------------------

def cmd_reliance(args: argparse.Namespace) -> None:
    """Matching vs opposing reliance chart across all configs with bias_eval output."""
    archive = Path(args.archive)
    configs = discover_configs(archive)

    # Filter to configs that have bias_eval reliance data
    loaded = []
    for c in configs:
        rel = _load_reliance(c["results_dir"])
        if rel is None:
            print(f"  SKIP {c['key']}: no bias_eval/reliance.json")
            continue
        c["reliance"] = rel
        loaded.append(c)

    if not loaded:
        raise SystemExit("No reliance.json files found. Run bias_eval.py first.")

    n_configs = len(loaded)
    biased    = ["gb", "ps", "ce", "pr"]
    bar_w     = 0.28
    x_base    = np.arange(n_configs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(DARK_BG)

    for ax_idx, model in enumerate(biased):
        ax = axes[ax_idx // 2, ax_idx % 2]
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)

        match_vals, opp_vals, colors = [], [], []
        for c in loaded:
            d = c["reliance"].get(model, {})
            match_vals.append(d.get("mean_matching", np.nan))
            opp_vals.append(d.get("mean_opposing", np.nan))
            colors.append(c["color"])

        for i, (mv, ov, col) in enumerate(zip(match_vals, opp_vals, colors)):
            b1 = ax.bar(x_base[i] - bar_w / 2, mv, width=bar_w, color=col,
                        alpha=0.90, zorder=3, edgecolor="white", linewidth=0.5)
            b2 = ax.bar(x_base[i] + bar_w / 2, ov, width=bar_w, color=col,
                        alpha=0.40, zorder=3, edgecolor=col, linewidth=1.0, hatch="///")
            for bar, val in [(b1[0], mv), (b2[0], ov)]:
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.006, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=8,
                            color=TEXT_COLOR, fontweight="bold")

        ax.axhline(1.0, color="#aab0c4", linewidth=1.0, linestyle="--", zorder=2, alpha=0.6)
        ax.set_xticks(x_base)
        ax.set_xticklabels([c["label"] for c in loaded], color=TEXT_COLOR, fontsize=8.5)
        ax.set_ylabel("Reliance ratio  (stylized AUC / original AUC)", color=TEXT_COLOR, fontsize=8)
        ax.set_ylim(0.60, 1.08)
        ax.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
        ax.set_title(MODEL_LABELS[model], color=TEXT_COLOR,
                     fontsize=10.5, fontweight="bold", loc="left")

        mp = mpatches.Patch(facecolor="#aaaaaa", alpha=0.9,
                            label="Matching reliance\n(same bias test set)")
        op = mpatches.Patch(facecolor="#aaaaaa", alpha=0.4, hatch="///",
                            label="Opposing reliance\n(opposite bias test set)")
        ax.legend(handles=[mp, op], framealpha=0.2, facecolor=PANEL_BG,
                  edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=7.5,
                  loc="lower right")

    legend_handles = [
        mpatches.Patch(facecolor=c["color"], alpha=0.85, label=c["label"])
        for c in loaded
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=n_configs,
               framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    config_names = ", ".join(c["label"] for c in loaded)
    fig.suptitle(f"Bias-Eval Reliance Ratios — {config_names}  |  Matching vs Opposing",
                 color=TEXT_COLOR, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    out = archive / "reliance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"Saved: {out}")
    if not args.no_show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plotting tools for the shape-vs-texture bias project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--no-show", action="store_true",
                        help="Save figure without calling plt.show()")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # curves
    p_curves = sub.add_parser("curves", help="Training curves for one results dir")
    p_curves.add_argument("--results-dir", required=True,
                          help="Path to a config's results/ folder, e.g. "
                               "src/configs/archive_results_configs/config_1/results")
    p_curves.add_argument("--out", default=None, help="Override output path")

    # compare
    p_cmp = sub.add_parser("compare", help="Multi-config AUROC comparison (auto-discovers archive)")
    p_cmp.add_argument("--archive", default=str(ARCHIVE_DEFAULT),
                       help=f"Root archive folder (default: {ARCHIVE_DEFAULT})")

    # reliance
    p_rel = sub.add_parser("reliance", help="Matching vs opposing reliance (auto-discovers archive)")
    p_rel.add_argument("--archive", default=str(ARCHIVE_DEFAULT),
                       help=f"Root archive folder (default: {ARCHIVE_DEFAULT})")

    args = parser.parse_args()
    {"curves": cmd_curves, "compare": cmd_compare, "reliance": cmd_reliance}[args.cmd](args)


if __name__ == "__main__":
    main()
