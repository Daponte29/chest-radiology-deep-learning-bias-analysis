TEXTURE_TESTS = {"ps", "pr"}
SHAPE_TESTS   = {"gb", "ce"}

BIAS_TYPE = {
    "gb": "shape",
    "ps": "texture",
    "ce": "shape",
    "pr": "texture",
}


def compute_reliance(model_name: str, aucs: dict[str, float]) -> dict:
    """Compute matching and opposing reliance ratios for one biased model.

    Args:
        model_name: One of gb, ps, ce, pr.
        aucs: Dict mapping test set name -> auroc_5 score.
              Must include 'original' and all 4 stylized keys.

    Returns:
        Dict with keys: model, bias_type, matching (2 entries), opposing (2 entries),
        mean_matching_reliance, mean_opposing_reliance.
    """
    baseline = aucs["original"]
    bias     = BIAS_TYPE[model_name]

    if bias == "texture":
        matching_keys  = sorted(TEXTURE_TESTS)
        opposing_keys  = sorted(SHAPE_TESTS)
    else:
        matching_keys  = sorted(SHAPE_TESTS)
        opposing_keys  = sorted(TEXTURE_TESTS)

    matching  = {k: round(aucs[k] / baseline, 6) for k in matching_keys}
    opposing  = {k: round(aucs[k] / baseline, 6) for k in opposing_keys}

    return {
        "model":                    model_name,
        "bias_type":                bias,
        "auroc_original_test":      round(baseline, 6),
        "matching":                 matching,
        "opposing":                 opposing,
        "mean_matching_reliance":   round(sum(matching.values()) / len(matching), 6),
        "mean_opposing_reliance":   round(sum(opposing.values()) / len(opposing), 6),
    }
