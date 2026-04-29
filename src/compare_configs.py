"""
compare_configs.py — Visual comparison of all training configs (Ed + Nick 1/2/3).

Reads test_results.json from:
  - ed_share/results/{model}/          (Ed's config)
  - ARCHIVE_NICK/{1,2,3}/{model}/      (Nick's configs)

Produces a publication-quality multi-panel figure saved to results/ARCHIVE_NICK/.

Usage:
    python -m src.compare_configs
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Compare training configs from archived results")
parser.add_argument(
    "--archive-dir", default="results/ARCHIVE_NICK",
    help="Root folder containing Nick's numbered subfolders. Default: results/ARCHIVE_NICK",
)
parser.add_argument(
    "--ed-dir", default="ed_share/results",
    help="Folder containing Ed's per-model result subdirs. Default: ed_share/results",
)
args = parser.parse_args()

ARCHIVE_ROOT = Path(args.archive_dir)
ED_ROOT      = Path(args.ed_dir)

# Ed first (best baseline), then Nick's three configs in run order.
CONFIGS = {
    "Ed's Config\nBCE, No Sampler": {
        "base_dir": ED_ROOT,       # direct parent of model subdirs
        "loss": "BCE",
        "sampler": "No",
        "lr": "1e-4",
        "weight_decay": "1e-5",
        "labels": "11",
        "color": "#B07FE0",
    },
    "Nick Config 1\nFocal γ=1.5 + Sampler": {
        "base_dir": ARCHIVE_ROOT / "1",
        "loss": "Focal  γ=1.5",
        "sampler": "Yes",
        "lr": "5e-5",
        "weight_decay": "5e-5",
        "labels": "14",
        "color": "#E07B39",
    },
    "Nick Config 2\nBCE + Sampler": {
        "base_dir": ARCHIVE_ROOT / "2",
        "loss": "BCE",
        "sampler": "Yes",
        "lr": "1e-4",
        "weight_decay": "1e-5",
        "labels": "14",
        "color": "#4878CF",
    },
    "Nick Config 3\nFocal γ=2.0, No Sampler": {
        "base_dir": ARCHIVE_ROOT / "3",
        "loss": "Focal  γ=2.0",
        "sampler": "No",
        "lr": "1e-4",
        "weight_decay": "1e-5",
        "labels": "14",
        "color": "#6ACC65",
    },
}

MODELS = ["original", "gb", "ps", "ce", "pr"]
MODEL_LABELS = {
    "original": "Original",
    "gb":       "Gaussian Blur\n(texture bias)",
    "ps":       "Patch Shuffle\n(texture bias)",
    "ce":       "Canny Edge\n(shape bias)",
    "pr":       "Patch Rotation\n(shape bias)",
}

# All 14 CheXpert labels minus Fracture (always NaN in all runs).
# Labels Ed didn't train on (Lung Lesion, No Finding) will appear as N/A in heatmaps.
LABEL_ORDER = [
    "Support Devices", "No Finding", "Pleural Effusion", "Pleural Other",
    "Lung Opacity", "Edema", "Consolidation", "Atelectasis",
    "Pneumothorax", "Cardiomegaly", "Pneumonia",
    "Enlarged Cardiomediastinum", "Lung Lesion",
]


def load_results(meta: dict) -> dict[str, dict]:
    out = {}
    for model in MODELS:
        path = Path(meta["base_dir"]) / model / "test_results.json"
        with open(path) as f:
            out[model] = json.load(f)
    return out


def safe_mean(values: list) -> float:
    valid = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------

all_data = {label: load_results(meta) for label, meta in CONFIGS.items()}

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

DARK_BG    = "#0f1117"
PANEL_BG   = "#1a1d27"
TEXT_COLOR  = "#e8eaf0"
GRID_COLOR  = "#2a2d3a"
NAN_COLOR   = "#2e2e3a"

cmap = LinearSegmentedColormap.from_list("dark_rg", ["#8b0000", "#cc4400", "#e8b84b", "#4878CF", "#1a6e2e"])

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(22, 22))
fig.patch.set_facecolor(DARK_BG)

gs = GridSpec(
    3, 2,
    figure=fig,
    hspace=0.45,
    wspace=0.35,
    top=0.92, bottom=0.05,
    left=0.07, right=0.97,
    height_ratios=[0.24, 0.38, 0.38],
)

config_items = list(CONFIGS.items())
n_configs    = len(config_items)
n_models     = len(MODELS)

# ---------------------------------------------------------------------------
# Panel 0 — Config summary table
# ---------------------------------------------------------------------------

ax_table = fig.add_subplot(gs[0, :])
ax_table.set_facecolor(PANEL_BG)
ax_table.set_xlim(0, 1)
ax_table.set_ylim(0, 1)
ax_table.axis("off")

col_headers = ["", "Loss Function", "Sampler", "Learning Rate", "Weight Decay", "Labels", "Notes"]
col_x       = [0.01, 0.19, 0.34, 0.46, 0.57, 0.67, 0.76]

# 4 rows evenly spaced between y=0.84 and y=0.06
row_y = [0.82, 0.60, 0.38, 0.16]

config_rows = [
    ["Ed's Config  —  BCE, No Sampler",          "BCE",               "✗  No",  "1e-4",  "1e-5", "11", "Best overall AUROC"],
    ["Nick Config 1  —  Focal γ=1.5 + Sampler",  "Focal Loss  γ=1.5", "✓  Yes", "5e-5",  "5e-5", "14", "Instability study"],
    ["Nick Config 2  —  BCE + Sampler",           "BCE",               "✓  Yes", "1e-4",  "1e-5", "14", "Sampler-only fix"],
    ["Nick Config 3  —  Focal γ=2.0, No Sampler", "Focal Loss  γ=2.0", "✗  No", "1e-4",  "1e-5", "14", "Focal-only fix"],
]

header_props = dict(fontsize=9, color="#aab0c4", fontweight="bold", ha="left", va="center",
                    transform=ax_table.transAxes)
for hdr, x in zip(col_headers, col_x):
    ax_table.text(x, 0.93, hdr, **header_props)

ax_table.axhline(0.87, color=GRID_COLOR, linewidth=0.8)

row_height = 0.20
for (cfg_label, meta), row_data, y in zip(config_items, config_rows, row_y):
    color = meta["color"]
    ax_table.add_patch(mpatches.FancyBboxPatch(
        (col_x[0] - 0.005, y - row_height * 0.42), 0.165, row_height * 0.84,
        boxstyle="round,pad=0.01", facecolor=color, alpha=0.18,
        transform=ax_table.transAxes, zorder=2,
    ))
    for val, x in zip(row_data, col_x):
        is_name = (x == col_x[0])
        ax_table.text(x, y, val,
                      fontsize=8.5 if not is_name else 9,
                      color=color if is_name else TEXT_COLOR,
                      fontweight="bold" if is_name else "normal",
                      ha="left", va="center",
                      transform=ax_table.transAxes)
    ax_table.axhline(y - row_height * 0.46, color=GRID_COLOR, linewidth=0.5)

ax_table.set_title(
    "Training Configuration Comparison  —  Ed & Nick Experiment Grid",
    color=TEXT_COLOR, fontsize=13, fontweight="bold", loc="left", pad=8,
)

# ---------------------------------------------------------------------------
# Panel 1 — Grouped bar chart: mean AUROC by model × config
# ---------------------------------------------------------------------------

ax_bar = fig.add_subplot(gs[1, :])
ax_bar.set_facecolor(PANEL_BG)
ax_bar.tick_params(colors=TEXT_COLOR)
for spine in ax_bar.spines.values():
    spine.set_edgecolor(GRID_COLOR)

bar_w     = 0.17
group_gap = 0.12
x_base    = np.arange(n_models) * (n_configs * bar_w + group_gap + 0.08)

for ci, (cfg_label, meta) in enumerate(config_items):
    means = [all_data[cfg_label][m]["auroc_mean"] for m in MODELS]
    x_pos = x_base + ci * bar_w
    bars  = ax_bar.bar(x_pos, means, width=bar_w, color=meta["color"],
                       alpha=0.88, label=cfg_label.replace("\n", "  "),
                       zorder=3, edgecolor="white", linewidth=0.3)
    for bar, val in zip(bars, means):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=6.8, color=TEXT_COLOR, fontweight="bold")

center_offset = (n_configs * bar_w) / 2 - bar_w / 2
ax_bar.set_xticks(x_base + center_offset)
ax_bar.set_xticklabels([MODEL_LABELS[m] for m in MODELS], color=TEXT_COLOR, fontsize=9.5)
ax_bar.set_ylabel("Mean AUROC (test set)", color=TEXT_COLOR, fontsize=10)
ax_bar.set_ylim(0.55, 0.93)
ax_bar.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
ax_bar.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
ax_bar.set_facecolor(PANEL_BG)
ax_bar.set_title("Mean AUROC by Model Variant and Training Config",
                 color=TEXT_COLOR, fontsize=12, fontweight="bold", loc="left")
ax_bar.legend(framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=8.5, loc="lower right")

# ---------------------------------------------------------------------------
# Panel 2 — Per-label heatmap: original model, all 4 configs
# ---------------------------------------------------------------------------

ax_heat = fig.add_subplot(gs[2, 0])
ax_heat.set_facecolor(PANEL_BG)

heat_data      = []
col_labels_heat = []
for cfg_label, meta in config_items:
    col_labels_heat.append(cfg_label.replace("\n", "\n"))
    per_label = all_data[cfg_label]["original"]["per_label"]
    row = [per_label.get(lbl, float("nan")) for lbl in LABEL_ORDER]
    heat_data.append(row)

heat_arr = np.array(heat_data, dtype=float).T  # (n_labels, n_configs)

# Build masked array so NaN cells render in a distinct neutral colour
masked = np.ma.masked_invalid(heat_arr)
cmap_nan = cmap.copy()
cmap_nan.set_bad(color=NAN_COLOR)

im = ax_heat.imshow(masked, aspect="auto", cmap=cmap_nan, vmin=0.0, vmax=1.0)

ax_heat.set_xticks(range(n_configs))
ax_heat.set_xticklabels(col_labels_heat, color=TEXT_COLOR, fontsize=7.5)
ax_heat.set_yticks(range(len(LABEL_ORDER)))
ax_heat.set_yticklabels(LABEL_ORDER, color=TEXT_COLOR, fontsize=8)
ax_heat.tick_params(colors=TEXT_COLOR)

for i in range(len(LABEL_ORDER)):
    for j in range(n_configs):
        val = heat_arr[i, j]
        if np.isnan(val):
            ax_heat.text(j, i, "N/A", ha="center", va="center",
                         fontsize=7, color="#666888", fontweight="bold")
        else:
            txt_color = "white" if val < 0.65 else "#111"
            ax_heat.text(j, i, f"{val:.3f}", ha="center", va="center",
                         fontsize=7, color=txt_color, fontweight="bold")

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
cbar.set_label("AUROC", color=TEXT_COLOR, fontsize=9)

ax_heat.set_title("Per-Label AUROC — Original Model (all configs)\n"
                  "N/A = label not in Ed's 11-label training set",
                  color=TEXT_COLOR, fontsize=10, fontweight="bold", loc="left")

# ---------------------------------------------------------------------------
# Panel 3 — Ed's config per-label across all 5 model variants
# ---------------------------------------------------------------------------

ax_heat2 = fig.add_subplot(gs[2, 1])
ax_heat2.set_facecolor(PANEL_BG)

ed_label  = list(CONFIGS.keys())[0]
ed_labels = [lbl for lbl in LABEL_ORDER
             if not np.isnan(all_data[ed_label]["original"]["per_label"].get(lbl, float("nan")))]

heat2_data = []
for model in MODELS:
    per_label = all_data[ed_label][model]["per_label"]
    row = [per_label.get(lbl, float("nan")) for lbl in ed_labels]
    heat2_data.append(row)

heat2_arr    = np.array(heat2_data, dtype=float).T
masked2      = np.ma.masked_invalid(heat2_arr)

im2 = ax_heat2.imshow(masked2, aspect="auto", cmap=cmap_nan, vmin=0.0, vmax=1.0)

ax_heat2.set_xticks(range(n_models))
ax_heat2.set_xticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODELS],
                          color=TEXT_COLOR, fontsize=8, rotation=15, ha="right")
ax_heat2.set_yticks(range(len(ed_labels)))
ax_heat2.set_yticklabels(ed_labels, color=TEXT_COLOR, fontsize=8)
ax_heat2.tick_params(colors=TEXT_COLOR)

for i in range(len(ed_labels)):
    for j in range(n_models):
        val = heat2_arr[i, j]
        if np.isnan(val):
            ax_heat2.text(j, i, "N/A", ha="center", va="center",
                          fontsize=7, color="#666888", fontweight="bold")
        else:
            txt_color = "white" if val < 0.65 else "#111"
            ax_heat2.text(j, i, f"{val:.3f}", ha="center", va="center",
                          fontsize=7, color=txt_color, fontweight="bold")

cbar2 = fig.colorbar(im2, ax=ax_heat2, fraction=0.046, pad=0.04)
cbar2.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
cbar2.set_label("AUROC", color=TEXT_COLOR, fontsize=9)

ax_heat2.set_title("Ed's Config — Per-Label AUROC Across All 5 Model Variants\n"
                   "(highest-performing config, 11 labels)",
                   color=TEXT_COLOR, fontsize=10, fontweight="bold", loc="left")

# ---------------------------------------------------------------------------
# Super title & save
# ---------------------------------------------------------------------------

fig.suptitle(
    "Shape vs. Texture Bias — DenseNet121 on CheXpert  |  Training Config Ablation Study  (Ed + Nick)",
    color=TEXT_COLOR, fontsize=14, fontweight="bold", y=0.975,
)

out_path = ARCHIVE_ROOT / "config_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"Saved: {out_path}")
plt.show()
