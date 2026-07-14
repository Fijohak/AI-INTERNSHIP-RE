"""Train LeNet-5 with data augmentation."""
import argparse, time, json
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import lenet5, count_parameters

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def get_aug_loaders(batch_size, data_root):
    train_tf = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=test_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=100):
    model.train(); total_loss = 0.0; correct = 0; total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(); loss = criterion(model(inputs), targets)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        _, predicted = model(inputs).max(1)
        total += targets.size(0); correct += predicted.eq(targets).sum().item()
        if batch_idx % log_interval == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx:>4d}/{len(loader):<4d} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval(); total_loss = 0.0; correct = 0; total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs); loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0); correct += predicted.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="../experiments_aug")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader = get_aug_loaders(args.batch_size, args.data_root)
    print(f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    model = lenet5(10).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_acc = 0.0

    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        vl, va = evaluate(model, test_loader, criterion, device)
        train_losses.append(tl); train_accs.append(ta)
        test_losses.append(vl); test_accs.append(va)
        print(f"\n--- Epoch {epoch} | Train Loss: {tl:.4f} Acc: {ta:.2f}% | Test Loss: {vl:.4f} Acc: {va:.2f}% ---\n")
        if va > best_test_acc:
            best_test_acc = va
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  -> Best: {best_test_acc:.2f}%\n")

    total_time = time.time() - start_time
    print(f"Done! Time: {total_time:.1f}s | Best: {best_test_acc:.2f}%")

    os.makedirs(save_dir / "logs", exist_ok=True)
    with open(save_dir / "logs" / "training_history.json", "w") as f:
        json.dump({"train_losses": train_losses, "train_accs": train_accs,
                    "test_losses": test_losses, "test_accs": test_accs,
                    "best_test_acc": best_test_acc}, f, indent=2)


if __name__ == "__main__":
    import os
    main()
