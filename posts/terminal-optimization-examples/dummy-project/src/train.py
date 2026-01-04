"""Training script for energy model."""
from model import EnergyPredictor
import torch

def train(epochs=100, lr=0.001):
    model = EnergyPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # TODO: Add training loop
    return model

if __name__ == "__main__":
    train()
# TODO: Add validation
