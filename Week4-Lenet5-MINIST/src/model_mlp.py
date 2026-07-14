"""
Simple MLP baseline for MNIST — to contrast with LeNet-5 CNN.

~59K parameters, comparable to LeNet-5's 61,706.
"""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """3-layer MLP with ~59K parameters (comparable to LeNet-5's 61,706)"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 72)   # 784*72+72 = 56,520
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(72, 32)         # 72*32+32 = 2,336
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes) # 32*10+10 = 330
        # Total: 56,520 + 2,336 + 330 = 59,186

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)            # (batch, 784)
        x = self.relu1(self.fc1(x))   # (batch, 72)
        x = self.relu2(self.fc2(x))   # (batch, 32)
        x = self.fc3(x)               # (batch, 10)
        return x


def mlp_baseline(num_classes: int = 10) -> MLPBaseline:
    return MLPBaseline(num_classes=num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = mlp_baseline()
    dummy = torch.randn(1, 1, 28, 28)
    output = model(dummy)
    print(f"Model: {model}")
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {count_parameters(model):,}")
