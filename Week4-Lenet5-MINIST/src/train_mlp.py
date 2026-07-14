"""Training script for MLP Baseline."""
import argparse, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from model_mlp import mlp_baseline, count_parameters
from dataset import get_dataloaders


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
    parser.add_argument("--save-dir", type=str, default="../experiments_mlp")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    loaders = get_dataloaders(batch_size=args.batch_size, data_root=args.data_root)
    model = mlp_baseline(10).to(device)
    print(f"MLP Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_test_acc = 0.0
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders['train_loader'], criterion, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, loaders['test_loader'], criterion, device)
        print(f"\n--- Epoch {epoch} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% ---\n")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  -> Best: {best_test_acc:.2f}%\n")

    print(f"Done! Time: {time.time()-start_time:.1f}s | Best: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()
