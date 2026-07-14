"""
Visualization utilities

Functions:
  1. Plot training loss/accuracy curves
  2. Show random test predictions
  3. Show misclassified samples
  4. Plot confusion matrix heatmap
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import lenet5
from dataset import get_dataloaders
from evaluate import evaluate_test, per_class_accuracy, confusion_matrix


plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_training_curves(history: dict, save_path: str = None):
    """
    Plot training loss and accuracy curves

    Args:
        history: {"train_losses", "train_accs", "test_losses", "test_accs"}
        save_path: Save path (None to display)
    """
    epochs = range(1, len(history["train_losses"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    ax1.plot(epochs, history["train_losses"], "o-", label="Train Loss", color="#2196F3")
    ax1.plot(epochs, history["test_losses"], "s--", label="Test Loss", color="#FF5722")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Test Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history["train_accs"], "o-", label="Train Acc", color="#2196F3")
    ax2.plot(epochs, history["test_accs"], "s--", label="Test Acc", color="#FF5722")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Test Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(90, 100)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved: {save_path}")
    plt.show()


def show_random_predictions(
    model, test_loader, device, num_samples: int = 10, save_path: str = None,
):
    """
    Show random test images with predictions

    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Device
        num_samples: Number of samples to show
        save_path: Save path
    """
    model.eval()

    # Collect test data
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        if sum(len(a) for a in all_images) >= num_samples * 2:
            break
    all_images = torch.cat(all_images)[:num_samples * 2]
    all_labels = torch.cat(all_labels)[:num_samples * 2]

    # Randomly select num_samples
    indices = list(range(len(all_images)))
    selected = random.sample(indices, num_samples)
    sample_images = all_images[selected]
    sample_labels = all_labels[selected]

    # Predict
    with torch.no_grad():
        outputs = model(sample_images.to(device))
        _, predicted = outputs.max(1)

    # Denormalize: mean=0.1307, std=0.3081
    mean, std = 0.1307, 0.3081

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(num_samples):
        img = sample_images[i].cpu().numpy().squeeze()
        img = img * std + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")

        pred = predicted[i].item()
        true = sample_labels[i].item()
        color = "green" if pred == true else "red"
        title = f"Pred: {pred}\nTrue: {true}"
        axes[i].set_title(title, color=color, fontsize=11)

    plt.suptitle("MNIST Test Predictions (green=correct, red=wrong)", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Predictions saved: {save_path}")
    plt.show()


def show_misclassified(
    misclassified: list, num_samples: int = 10, save_path: str = None,
):
    """
    Show misclassified samples

    Args:
        misclassified: [(image_tensor, pred, true), ...]
        num_samples: Number to display
        save_path: Save path
    """
    if not misclassified:
        print("No misclassified samples!")
        return

    samples = random.sample(misclassified, min(num_samples, len(misclassified)))
    mean, std = 0.1307, 0.3081

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i, (img_tensor, pred, true) in enumerate(samples):
        img = img_tensor.numpy().squeeze()
        img = img * std + mean
        img = np.clip(img, 0, 1)

        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Pred: {pred} / True: {true}", color="red", fontsize=11)

    plt.suptitle("Misclassified Samples Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Misclassified samples saved: {save_path}")
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.colorbar(im)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 Visualization")
    parser.add_argument("--model-path", type=str, default="../experiments/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="../experiments/figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = lenet5(num_classes=10).to(device)
    model_path = Path(args.model_path)
    if model_path.exists():
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Warning: {model_path} not found. Using untrained model.")

    loaders = get_dataloaders(batch_size=args.batch_size, data_root=args.data_root)
    test_loader = loaders["test_loader"]

    results = evaluate_test(model, test_loader, device)
    print(f"Test Accuracy: {results['accuracy']:.2f}%")

    save_dir = args.save_dir

    show_random_predictions(
        model, test_loader, device, num_samples=10,
        save_path=f"{save_dir}/predictions.png",
    )

    if results["misclassified"]:
        show_misclassified(
            results["misclassified"], num_samples=10,
            save_path=f"{save_dir}/misclassified.png",
        )

    cm = confusion_matrix(results["predictions"], results["targets"])
    plot_confusion_matrix(cm, save_path=f"{save_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
