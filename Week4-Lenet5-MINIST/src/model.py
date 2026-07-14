"""
LeNet-5 model definition (PyTorch)

Reference: Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition", 1998.

Adjustments:
- Input size adapted from 32x32 to 28x28 (MNIST default)
- Activation: Tanh -> ReLU (faster convergence)
- Pooling: AvgPool -> MaxPool (modern practice)
- Output: RBF -> Linear + CrossEntropyLoss
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5 (adapted for MNIST 28x28 input)"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # C1: Conv layer 1 - 6@28x28
        # padding=2 to maintain 28x28 (original LeNet-5 used 32x32 input)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        # S2: Pooling layer 1 - 6@14x14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # C3: Conv layer 2 - 16@10x10
        # Simplified: full connection map instead of sparse table
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        # S4: Pooling layer 2 - 16@5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # C5: Conv layer 3 - 120@1x1 (kernel 5x5, acts as FC)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.relu3 = nn.ReLU()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.pool1(self.relu1(self.conv1(x)))  # (batch, 6, 14, 14)
        x = self.pool2(self.relu2(self.conv2(x)))  # (batch, 16, 5, 5)
        x = self.relu3(self.conv3(x))              # (batch, 120, 1, 1)

        x = x.view(x.size(0), -1)                  # flatten: (batch, 120)
        x = self.relu4(self.fc1(x))                 # (batch, 84)
        x = self.fc2(x)                             # (batch, 10)
        return x


def lenet5(num_classes: int = 10) -> LeNet5:
    """Factory function for LeNet-5"""
    return LeNet5(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = lenet5()
    dummy = torch.randn(1, 1, 28, 28)
    output = model(dummy)
    print(f"Model: {model}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {count_parameters(model):,}")
