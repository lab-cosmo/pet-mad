from typing import Optional, Union

import numpy as np
import torch
from torch import nn


class TorchStandardScaler(nn.Module):
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.eps = 1e-7

    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)

    def transform(self, data):
        if self.mean is None or self.std is None:
            return data

        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data):
        if self.mean is None or self.std is None:
            return data

        return data * (self.std + self.eps) + self.mean
