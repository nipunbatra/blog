"""Neural network model for energy prediction."""
import torch
import torch.nn as nn

class EnergyPredictor(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
