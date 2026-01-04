import torch
from src.model import EnergyPredictor

def test_forward():
    model = EnergyPredictor()
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 1)
