"""
Tests for src/models/densenet.py — DenseNetClassifier
Uses pretrained=False everywhere to avoid downloading weights in CI.
"""
import pytest
import torch
import torch.nn as nn

from models.densenet import DenseNetClassifier, NUM_CHEXPERT_CLASSES


def _model(num_classes=NUM_CHEXPERT_CLASSES, variant="densenet121", dropout_p=0.0):
    return DenseNetClassifier(
        num_classes=num_classes,
        pretrained=False,
        variant=variant,
        dropout_p=dropout_p,
    )


def _batch(batch_size=2, channels=3, h=224, w=224):
    return torch.zeros(batch_size, channels, h, w)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_instantiates_default(self):
        m = _model()
        assert isinstance(m, DenseNetClassifier)

    def test_num_classes_stored(self):
        assert _model(num_classes=5).num_classes == 5

    def test_default_num_classes(self):
        assert _model().num_classes == NUM_CHEXPERT_CLASSES

    def test_invalid_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            DenseNetClassifier(variant="densenet999", pretrained=False)

    def test_densenet169_instantiates(self):
        m = _model(variant="densenet169")
        assert m.num_classes == NUM_CHEXPERT_CLASSES

    def test_densenet201_instantiates(self):
        m = _model(variant="densenet201")
        assert m.num_classes == NUM_CHEXPERT_CLASSES

    def test_dropout_layer_present_when_set(self):
        m = _model(dropout_p=0.5)
        head = m.backbone.classifier
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in head)
        assert has_dropout

    def test_no_dropout_layer_by_default(self):
        m = _model(dropout_p=0.0)
        head = m.backbone.classifier
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in head)
        assert not has_dropout

    def test_classifier_ends_with_linear(self):
        m = _model()
        last_layer = list(m.backbone.classifier.children())[-1]
        assert isinstance(last_layer, nn.Linear)

    def test_output_features_match_num_classes(self):
        m = _model(num_classes=7)
        linear = [l for l in m.backbone.classifier.children() if isinstance(l, nn.Linear)][-1]
        assert linear.out_features == 7


# ---------------------------------------------------------------------------
# forward()
# ---------------------------------------------------------------------------

class TestForward:
    def test_output_shape_default(self):
        m = _model()
        out = m(_batch())
        assert out.shape == (2, NUM_CHEXPERT_CLASSES)

    def test_output_shape_custom_classes(self):
        m = _model(num_classes=3)
        out = m(_batch())
        assert out.shape == (2, 3)

    def test_output_is_tensor(self):
        assert isinstance(_model()(_batch()), torch.Tensor)

    def test_output_dtype_float32(self):
        assert _model()(_batch()).dtype == torch.float32

    def test_output_contains_negatives(self):
        # raw logits — not bounded to [0,1], so negatives should appear
        m = _model()
        out = m(torch.randn(4, 3, 224, 224))
        assert (out < 0).any()

    def test_batch_size_one(self):
        out = _model()(_batch(batch_size=1))
        assert out.shape == (1, NUM_CHEXPERT_CLASSES)

    def test_gradient_flows(self):
        m = _model()
        x = torch.randn(2, 3, 224, 224, requires_grad=False)
        out = m(x)
        loss = out.sum()
        loss.backward()  # should not raise


# ---------------------------------------------------------------------------
# predict_proba()
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_output_shape(self):
        out = _model().predict_proba(_batch())
        assert out.shape == (2, NUM_CHEXPERT_CLASSES)

    def test_values_in_zero_one(self):
        out = _model().predict_proba(torch.randn(4, 3, 224, 224))
        assert (out >= 0).all() and (out <= 1).all()

    def test_output_is_not_logits(self):
        # After sigmoid the output must be strictly in (0, 1)
        out = _model().predict_proba(torch.randn(4, 3, 224, 224))
        assert (out > 0).all() and (out < 1).all()


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestPredict:
    def test_output_shape(self):
        out = _model().predict(_batch())
        assert out.shape == (2, NUM_CHEXPERT_CLASSES)

    def test_output_is_binary(self):
        out = _model().predict(torch.randn(4, 3, 224, 224))
        unique = out.unique().tolist()
        assert set(unique).issubset({0, 1})

    def test_output_dtype_long(self):
        assert _model().predict(_batch()).dtype == torch.int64

    def test_custom_threshold_all_ones(self):
        # threshold=0.0 → everything predicted positive
        out = _model().predict(torch.randn(4, 3, 224, 224), threshold=0.0)
        assert out.all()

    def test_custom_threshold_all_zeros(self):
        # threshold=1.0 → nothing predicted positive (sigmoid < 1 always)
        out = _model().predict(torch.randn(4, 3, 224, 224), threshold=1.0)
        assert not out.any()
