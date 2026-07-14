"""
LeNet-5 with Tanh + SGD backbone, but only 1 FC layer (120→10 directly).
Contrasts with standard 3-layer FC head (120→84→10).
"""
import torch
import torch.nn as nn


class LeNet5Tanh1FC(nn.Module):
    """LeNet-5 (Tanh): only 1 FC layer after convs"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Conv layers (same as original LeNet-5)
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 120, 5)
        self.tanh3 = nn.Tanh()

        # Only 1 FC: 120 → num_classes (skip the 84-unit middle FC)
        self.fc = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool1(self.tanh1(self.conv1(x)))
        x = self.pool2(self.tanh2(self.conv2(x)))
        x = self.tanh3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LeNet5Tanh1FC()
    print(f"1-FC Tanh model: {count_parameters(model):,} params")
    # Standard LeNet-5: 61,706
    # 1-FC version: subtract (120*84+84=10,164) and (84*10+10=850), add (120*10+10=1,210)
    # = 61,706 - 11,014 + 1,210 = 51,902
