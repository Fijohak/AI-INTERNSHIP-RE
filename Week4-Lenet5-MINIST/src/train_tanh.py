"""
LeNet-5 training script (Tanh activation version)

Usage: python train_tanh.py [--epochs 10] [--batch-size 64] [--lr 0.001]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model_tanh import lenet5_tanh, count_parameters
from dataset import get_dataloaders


def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, log_interval=100,
):
    """Train for one epoch, return average loss and accuracy"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % log_interval == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx:>4d}/{len(loader):<4d} | "
                f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%"
            )

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate on val/test set, return average loss and accuracy"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="LeNet-5 Tanh MNIST Training")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="../experiments_tanh")
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Arguments: {vars(args)}")

    torch.manual_seed(args.seed)

    loaders = get_dataloaders(
        batch_size=args.batch_size,
        data_root=args.data_root,
        val_split=0.0,
    )
    train_loader = loaders["train_loader"]
    test_loader = loaders["test_loader"]
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    model = lenet5_tanh(num_classes=10).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_test_acc = 0.0

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Starting training (Tanh activation)...")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, args.log_interval,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"\n--- Epoch {epoch} Summary ---\n"
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%\n"
            f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%\n"
            f"  Time: {epoch_time:.1f}s\n"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  -> New best model saved! (Acc: {best_test_acc:.2f}%)\n")

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training complete! Total time: {total_time:.1f}s")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print("=" * 60)

    history = {
        "train_losses": train_losses,
        "train_accs": train_accs,
        "test_losses": test_losses,
        "test_accs": test_accs,
        "best_test_acc": best_test_acc,
    }
    return history, model, device


if __name__ == "__main__":
    main()
