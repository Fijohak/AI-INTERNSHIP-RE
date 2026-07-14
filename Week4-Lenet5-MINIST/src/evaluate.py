"""
Test set evaluation and error analysis

Usage: python evaluate.py [--model-path ../experiments/best_model.pth]
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from model import lenet5
from dataset import get_dataloaders


def evaluate_test(model, test_loader, device):
    """Full evaluation on test set"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    misclassified = []  # [(image, pred, true), ...]

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            for i in range(inputs.size(0)):
                if predicted[i] != targets[i]:
                    misclassified.append((
                        inputs[i].cpu(), predicted[i].item(), targets[i].item(),
                    ))

            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": all_preds,
        "targets": all_targets,
        "misclassified": misclassified,
    }


def per_class_accuracy(predictions, targets, num_classes=10):
    """Per-class accuracy"""
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for pred, true in zip(predictions, targets):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    per_class = {}
    for i in range(num_classes):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        per_class[i] = {
            "accuracy": acc,
            "correct": class_correct[i],
            "total": class_total[i],
        }
    return per_class


def confusion_matrix(predictions, targets, num_classes=10):
    """Build confusion matrix"""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(predictions, targets):
        matrix[true, pred] += 1
    return matrix


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 Evaluation")
    parser.add_argument("--model-path", type=str, default="../experiments/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = lenet5(num_classes=10).to(device)
    model_path = Path(args.model_path)
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from: {model_path}")
    else:
        print(f"Warning: {model_path} not found, using random weights.")

    loaders = get_dataloaders(batch_size=args.batch_size, data_root=args.data_root)
    test_loader = loaders["test_loader"]

    results = evaluate_test(model, test_loader, device)
    print(f"\nTest Accuracy: {results['accuracy']:.2f}% ({results['correct']}/{results['total']})")

    per_class = per_class_accuracy(results["predictions"], results["targets"])
    print("\nPer-class accuracy:")
    for digit in range(10):
        info = per_class[digit]
        print(f"  Digit {digit}: {info['accuracy']:.2f}% ({info['correct']}/{info['total']})")

    cm = confusion_matrix(results["predictions"], results["targets"])
    print(f"\nConfusion matrix (row=true, col=pred):\n{cm}")

    n_errors = len(results["misclassified"])
    print(f"\nTotal misclassified: {n_errors}")

    return results


if __name__ == "__main__":
    main()
