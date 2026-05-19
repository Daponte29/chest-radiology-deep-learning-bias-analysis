import pytest
import torch
from src.config import ModelConfig
from src.model import build_model


# TODO: update input shape and output shape once architecture is defined

@pytest.fixture
def model():
    cfg = ModelConfig()
    return build_model(cfg).eval()


def test_model_builds(model):
    assert model is not None


# def test_forward_shape(model):
#     x = torch.zeros(2, 3, 224, 224)   # TODO: set correct input shape
#     with torch.no_grad():
#         out = model(x)
#     assert out.shape == (2, NUM_CLASSES)  # TODO: set expected output shape
