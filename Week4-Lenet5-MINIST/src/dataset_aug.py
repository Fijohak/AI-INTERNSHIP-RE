"""
Data augmentation transforms for MNIST.
"""
from torchvision import transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_aug_transforms():
    """Train transforms with augmentation + test transform (no augmentation)"""
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    return train_transform, test_transform
