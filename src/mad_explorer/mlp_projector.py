import torch
from torch import nn


class MLPProjector(nn.Module):
    """A simple MLP used to project feature vectors to low dimention
    representations"""

    def __init__(self, input_dim: int = 1024, output_dim: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)
