from typing import Optional

import torch


class MLPProjector(torch.nn.Module):
    """
    MLP used to project feature vectors to low-dimensional representations

    :param input_dim: dimensionality of input features
    :param output_dim: target output dimensionality
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = torch.nn.Linear(self.input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.output = torch.nn.Linear(128, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.output(x)


class TorchStandardScaler(torch.nn.Module):
    """The scaler to standatize features by removing the mean and scaling to
    unit variance"""

    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = 1e-7

    def fit(self, data: torch.Tensor):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return data

        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return data

        return data * (self.std + self.eps) + self.mean
