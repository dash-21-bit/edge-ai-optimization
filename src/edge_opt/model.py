from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    A small CNN suitable for edge experiments (Fashion-MNIST 28x28 grayscale).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        # Conv block 1: input=1 channel (grayscale), output=16 feature maps
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Conv block 2: output=32 feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Fully connected layers
        # After two maxpools, 28x28 -> 14x14 -> 7x7, channels=32 => 32*7*7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)         # convolution
        x = self.bn1(x)           # stabilizes training
        x = F.relu(x)             # non-linearity
        x = F.max_pool2d(x, 2)    # downsample 28->14

        # Block 2
        x = self.conv2(x)         # convolution
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)    # downsample 14->7

        # Flatten for dense layers
        x = torch.flatten(x, 1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
