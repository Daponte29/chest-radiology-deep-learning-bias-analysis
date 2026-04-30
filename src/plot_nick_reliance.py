"""
plot_nick_reliance.py — Matching vs opposing reliance for Nick's 3 configs.

Reads bias_eval/reliance.json from ARCHIVE_NICK/{1,2,3}/ and produces a
2×2 panel figure (one panel per biased model) saved to results/ARCHIVE_NICK/.

Note: the reliance.json files were generated with an older bias-type definition
(gb/pr were swapped). This script re-derives the correct matching/opposing
groupings using the current definitions before plotting.

Usage:
    python -m src.plot_nick_reliance
"""

import json
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

ARCHIVE_ROOT = Path("results/ARCHIVE_NICK")

CONFIGS = {
    "Config 1\nFocal γ=1.5 + Sampler": {"folder": "1", "color": "#E07B39"},
    "Config 2\nBCE + Sampler":          {"folder": "2", "color": "#4878CF"},
    "Config 3\nFocal γ=2.0, No Sampler": {"folder": "3", "color": "#6ACC65"},
}

# Correct definitions (fixed from original bias_eval run)
_TEXTURE = {"gb", "ps"}
_SHAPE   = {"ce", "pr"}
_CORRECT_BIAS = {"gb": "texture", "ps": "texture", "ce": "shape", "pr": "shape"}

BIASED_MODELS = ["gb", "ps", "ce", "pr"]
MODEL_DISPLAY = {
    "gb": "Gaussian Blur\n(texture bias)",
    "ps": "Patch Shuffle\n(texture bias)",
    "ce": "Canny Edge\n(shape bias)",
    "pr": "Patch Rotation\n(shape bias)",
}

DARK_BG    = "#0f1117"
PANEL_BG   = "#1a1d27"
TEXT_COLOR = "#e8eaf0"
GRID_COLOR = "#2a2d3a"


def load_reliance(folder: str) -> dict[str, dict]:
    """Load reliance.json and re-derive correct matching/opposing per model."""
    path = ARCHIVE_ROOT / folder / "bias_eval" / "reliance.json"
    with open(path) as f:
        raw = json.load(f)

    result = {}
    for entry in raw:
        model = entry["model"]
        bias  = _CORRECT_BIAS[model]

        # Flatten all per-test-set ratios regardless of how they were grouped
        all_ratios: dict[str, float] = {}
        all_ratios.update(entry["matching"])
        all_ratios.update(entry["opposing"])

        matching_keys = sorted(_TEXTURE if bias == "texture" else _SHAPE)
        opposing_keys = sorted(_SHAPE   if bias == "texture" else _TEXTURE)

        matching = {k: all_ratios[k] for k in matching_keys if k in all_ratios}
        opposing = {k: all_ratios[k] for k in opposing_keys if k in all_ratios}

        result[model] = {
            "bias_type":        bias,
            "mean_matching":    np.mean(list(matching.values())) if matching else np.nan,
            "mean_opposing":    np.mean(list(opposing.values())) if opposing else np.nan,
            "matching":         matching,
            "opposing":         opposing,
            "auroc_original":   entry["auroc_original_test"],
        }
    return result


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

all_reliance = {}
for cfg_label, meta in CONFIGS.items():
    try:
        all_reliance[cfg_label] = load_reliance(meta["folder"])
    except FileNotFoundError as e:
        print(f"SKIP {cfg_label}: {e}")

if not all_reliance:
    raise SystemExit("No reliance.json files found. Run bias_eval.py first.")

# ---------------------------------------------------------------------------
# Build figure — 2×2 panels, one per biased model
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(DARK_BG)

config_items = list(CONFIGS.items())
n_configs    = len(config_items)
bar_w        = 0.28
x_base       = np.arange(n_configs)

for ax_idx, model in enumerate(BIASED_MODELS):
    ax  = axes[ax_idx // 2, ax_idx % 2]
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)

    match_vals   = []
    oppose_vals  = []
    bar_colors   = []

    for cfg_label, meta in config_items:
        data = all_reliance.get(cfg_label, {}).get(model, {})
        match_vals.append(data.get("mean_matching", np.nan))
        oppose_vals.append(data.get("mean_opposing", np.nan))
        bar_colors.append(meta["color"])

    x_match  = x_base - bar_w / 2
    x_oppose = x_base + bar_w / 2

    for i, (mv, ov, col) in enumerate(zip(match_vals, oppose_vals, bar_colors)):
        # Matching bar — solid
        b1 = ax.bar(x_match[i], mv, width=bar_w, color=col, alpha=0.90,
                    zorder=3, edgecolor="white", linewidth=0.5)
        # Opposing bar — same colour but cross-hatched and lighter
        b2 = ax.bar(x_oppose[i], ov, width=bar_w, color=col, alpha=0.40,
                    zorder=3, edgecolor=col, linewidth=1.0, hatch="///")

        # Value labels
        for bar, val in [(b1[0], mv), (b2[0], ov)]:
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.006,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=8, color=TEXT_COLOR, fontweight="bold")

    # Reference line at 1.0
    ax.axhline(1.0, color="#aab0c4", linewidth=1.0, linestyle="--", zorder=2, alpha=0.6)

    ax.set_xticks(x_base)
    ax.set_xticklabels(
        [cfg.replace("\n", "  ") for cfg, _ in config_items],
        color=TEXT_COLOR, fontsize=8.5,
    )
    ax.set_ylabel("Reliance ratio  (stylized AUC / original AUC)",
                  color=TEXT_COLOR, fontsize=8)
    ax.set_ylim(0.60, 1.08)
    ax.yaxis.set_tick_params(labelcolor=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_title(MODEL_DISPLAY[model], color=TEXT_COLOR,
                 fontsize=10.5, fontweight="bold", loc="left")

    # Per-panel legend (matching / opposing)
    match_patch  = mpatches.Patch(facecolor="#aaaaaa", alpha=0.9,
                                  label="Matching reliance\n(same bias test set)")
    oppose_patch = mpatches.Patch(facecolor="#aaaaaa", alpha=0.4, hatch="///",
                                  label="Opposing reliance\n(opposite bias test set)")
    ax.legend(handles=[match_patch, oppose_patch],
              framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, fontsize=7.5, loc="lower right")

# ---------------------------------------------------------------------------
# Config colour legend (shared, bottom of figure)
# ---------------------------------------------------------------------------

legend_handles = [
    mpatches.Patch(facecolor=meta["color"], alpha=0.85,
                   label=cfg.replace("\n", "  "))
    for cfg, meta in config_items
]
fig.legend(handles=legend_handles, loc="lower center", ncol=n_configs,
           framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
           labelcolor=TEXT_COLOR, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Bias-Eval Reliance Ratios — Nick's 3 Configs  |  Matching vs Opposing (corrected bias definitions)",
    color=TEXT_COLOR, fontsize=13, fontweight="bold", y=1.01,
)
fig.tight_layout(rect=[0, 0.05, 1, 1])

out_path = ARCHIVE_ROOT / "nick_3_configs.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
print(f"Saved: {out_path}")
plt.show()
