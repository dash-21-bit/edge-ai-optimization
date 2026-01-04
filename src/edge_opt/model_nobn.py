from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNNNoBN(nn.Module):
    """
    Small CNN without BatchNorm.
    This exports and quantizes more reliably on some ONNX toolchains.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Convolutions (same shapes as baseline)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # FC layers (same as baseline)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)       # 28 -> 14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)       # 14 -> 7
        x = torch.flatten(x, 1)      # (N, 32*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
