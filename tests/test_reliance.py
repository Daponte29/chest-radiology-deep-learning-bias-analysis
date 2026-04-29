"""
Tests for src/utils/reliance.py — compute_reliance()
"""
import math
import pytest
from utils.reliance import compute_reliance, BIAS_TYPE, TEXTURE_TESTS, SHAPE_TESTS


SAMPLE_AUCS = {
    "original": 0.800,
    "gb": 0.760,
    "ps": 0.720,
    "ce": 0.680,
    "pr": 0.740,
}


# ---------------------------------------------------------------------------
# Bias type constants
# ---------------------------------------------------------------------------

class TestBiasTypeConstants:
    def test_gb_is_texture(self):
        assert BIAS_TYPE["gb"] == "texture"

    def test_ps_is_texture(self):
        assert BIAS_TYPE["ps"] == "texture"

    def test_ce_is_shape(self):
        assert BIAS_TYPE["ce"] == "shape"

    def test_pr_is_shape(self):
        assert BIAS_TYPE["pr"] == "shape"

    def test_texture_tests_set(self):
        assert TEXTURE_TESTS == {"gb", "ps"}

    def test_shape_tests_set(self):
        assert SHAPE_TESTS == {"ce", "pr"}

    def test_no_overlap_between_bias_sets(self):
        assert TEXTURE_TESTS.isdisjoint(SHAPE_TESTS)

    def test_all_four_models_covered(self):
        assert TEXTURE_TESTS | SHAPE_TESTS == {"gb", "ps", "ce", "pr"}


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_dict(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert isinstance(result, dict)

    def test_has_model_key(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert result["model"] == "gb"

    def test_has_bias_type_key(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert "bias_type" in result

    def test_has_auroc_original_test(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert result["auroc_original_test"] == pytest.approx(0.800)

    def test_has_matching_dict(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert isinstance(result["matching"], dict)

    def test_has_opposing_dict(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert isinstance(result["opposing"], dict)

    def test_has_mean_matching_reliance(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert "mean_matching_reliance" in result

    def test_has_mean_opposing_reliance(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert "mean_opposing_reliance" in result

    def test_matching_has_two_entries(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert len(result["matching"]) == 2

    def test_opposing_has_two_entries(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert len(result["opposing"]) == 2


# ---------------------------------------------------------------------------
# Bias type assignment and key routing
# ---------------------------------------------------------------------------

class TestBiasRouting:
    def test_texture_model_gb_bias_type(self):
        assert compute_reliance("gb", SAMPLE_AUCS)["bias_type"] == "texture"

    def test_texture_model_ps_bias_type(self):
        assert compute_reliance("ps", SAMPLE_AUCS)["bias_type"] == "texture"

    def test_shape_model_ce_bias_type(self):
        assert compute_reliance("ce", SAMPLE_AUCS)["bias_type"] == "shape"

    def test_shape_model_pr_bias_type(self):
        assert compute_reliance("pr", SAMPLE_AUCS)["bias_type"] == "shape"

    def test_texture_model_matching_keys_are_texture(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert set(result["matching"].keys()) == {"gb", "ps"}

    def test_texture_model_opposing_keys_are_shape(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        assert set(result["opposing"].keys()) == {"ce", "pr"}

    def test_shape_model_matching_keys_are_shape(self):
        result = compute_reliance("ce", SAMPLE_AUCS)
        assert set(result["matching"].keys()) == {"ce", "pr"}

    def test_shape_model_opposing_keys_are_texture(self):
        result = compute_reliance("ce", SAMPLE_AUCS)
        assert set(result["opposing"].keys()) == {"gb", "ps"}


# ---------------------------------------------------------------------------
# Ratio computation correctness
# ---------------------------------------------------------------------------

class TestRatioValues:
    def test_matching_ratio_gb(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        expected = round(0.760 / 0.800, 6)
        assert result["matching"]["gb"] == pytest.approx(expected)

    def test_matching_ratio_ps(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        expected = round(0.720 / 0.800, 6)
        assert result["matching"]["ps"] == pytest.approx(expected)

    def test_opposing_ratio_ce(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        expected = round(0.680 / 0.800, 6)
        assert result["opposing"]["ce"] == pytest.approx(expected)

    def test_mean_matching_reliance(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        expected = round((0.760 / 0.800 + 0.720 / 0.800) / 2, 6)
        assert result["mean_matching_reliance"] == pytest.approx(expected, rel=1e-5)

    def test_mean_opposing_reliance(self):
        result = compute_reliance("gb", SAMPLE_AUCS)
        expected = round((0.680 / 0.800 + 0.740 / 0.800) / 2, 6)
        assert result["mean_opposing_reliance"] == pytest.approx(expected, rel=1e-5)

    def test_perfect_reliance_when_stylized_equals_original(self):
        aucs = {"original": 0.8, "gb": 0.8, "ps": 0.8, "ce": 0.8, "pr": 0.8}
        result = compute_reliance("gb", aucs)
        assert result["mean_matching_reliance"] == pytest.approx(1.0)
        assert result["mean_opposing_reliance"] == pytest.approx(1.0)

    def test_zero_stylized_gives_zero_reliance(self):
        aucs = {"original": 0.8, "gb": 0.0, "ps": 0.0, "ce": 0.0, "pr": 0.0}
        result = compute_reliance("gb", aucs)
        assert result["mean_matching_reliance"] == pytest.approx(0.0)

    def test_all_four_model_names_produce_results(self):
        for model in ["gb", "ps", "ce", "pr"]:
            result = compute_reliance(model, SAMPLE_AUCS)
            assert result["model"] == model
            assert not math.isnan(result["mean_matching_reliance"])
