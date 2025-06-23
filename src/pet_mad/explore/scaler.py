from typing import Optional

import torch
from torch import nn


class TorchStandardScaler(nn.Module):
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
