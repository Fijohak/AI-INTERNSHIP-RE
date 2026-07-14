"""
LeNet-5 model definition - Tanh activation version (original paper setting)

Reference: Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998.

This version uses Tanh activation as in the original paper.
All other settings remain the same as model.py.
"""

import torch
import torch.nn as nn


class LeNet5Tanh(nn.Module):
    """LeNet-5 with Tanh activation (original paper setting)"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # C1: Conv layer 1 - 6@28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
        # S2: Max pooling 1 - 6@14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # C3: Conv layer 2 - 16@10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.tanh2 = nn.Tanh()
        # S4: Max pooling 2 - 16@5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # C5: Conv layer 3 - 120@1x1
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.tanh3 = nn.Tanh()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.tanh1(self.conv1(x)))  # (batch, 6, 14, 14)
        x = self.pool2(self.tanh2(self.conv2(x)))  # (batch, 16, 5, 5)
        x = self.tanh3(self.conv3(x))              # (batch, 120, 1, 1)
        x = x.view(x.size(0), -1)                  # flatten: (batch, 120)
        x = self.tanh4(self.fc1(x))                 # (batch, 84)
        x = self.fc2(x)                             # (batch, 10)
        return x


def lenet5_tanh(num_classes: int = 10) -> LeNet5Tanh:
    """Factory function for LeNet-5 (Tanh version)"""
    return LeNet5Tanh(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = lenet5_tanh()
    dummy = torch.randn(1, 1, 28, 28)
    output = model(dummy)
    print(f"Model: {model}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {count_parameters(model):,}")
