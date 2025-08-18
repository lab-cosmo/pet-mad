import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=32, stride = 1, dilation = 1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=16, dilation = 1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8, dilation = 1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(64 * 72, 64 * 72)  # Adjust size after pooling
        self.fc2 = nn.Linear(64 * 72, 1)
        self.silu = nn.SiLU()

    def forward(self, x, last_layer = False):
        x = self.silu(self.conv1(x))  # Conv1 + ReLU
        x = self.pool(x)             # MaxPooling1D
        x = self.silu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool(x)             # MaxPooling1D
        x = self.silu(self.conv3(x))  # Conv2 + ReLU
        x = self.pool(x)             # MaxPooling1D
        x = x.view(x.size(0), -1)    # Flatten
        x = self.silu(self.fc1(x))  # Fully Connected Layer 1
        output = self.fc2(x)             # Fully Connected Layer 2 (Output)
        if last_layer:
            return x
        else:
            return output