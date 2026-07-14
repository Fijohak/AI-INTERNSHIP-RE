"""
LeNet-5 with AvgPool + Tanh (original paper's exact combo).
"""
import torch
import torch.nn as nn


class LeNet5AvgTanh(nn.Module):
    """LeNet-5: AvgPool + Tanh activation"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 120, 5)
        self.tanh3 = nn.Tanh()

        self.fc1 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.tanh1(self.conv1(x)))
        x = self.pool2(self.tanh2(self.conv2(x)))
        x = self.tanh3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.tanh4(self.fc1(x))
        x = self.fc2(x)
        return x


def lenet5_avg_tanh(num_classes=10):
    return LeNet5AvgTanh(num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
