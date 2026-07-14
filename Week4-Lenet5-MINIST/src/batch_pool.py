"""Fill the Pool dimension: 3 missing combos × 3 seeds = 9 runs."""
import json, os, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from dataset import get_dataloaders

# Import all model variants
from model import lenet5          # MaxPool + ReLU
from model_avg import lenet5_avg  # AvgPool + ReLU
from model_tanh import lenet5_tanh  # MaxPool + Tanh
from model_avg_tanh import lenet5_avg_tanh  # AvgPool + Tanh
from model_tanh_1fc import LeNet5Tanh1FC   # 1-FC Tanh
from model_tanh_3fc import LeNet5Tanh3FC   # 3-FC Tanh


def _tanh1fc(num_classes=10):
    return LeNet5Tanh1FC(num_classes=num_classes)


def _tanh3fc(num_classes=10):
    return LeNet5Tanh3FC(num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = model(inputs).max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return total_loss / len(loader), 100.0 * correct / total


def run_combo(label, model_fn, opt_name, lr, seed, save_dir, data_root):
    """Run one training and return best accuracy."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if (Path(save_dir) / "best_model.pth").exists():
        print(f"  [SKIP] {save_dir} exists")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    loaders = get_dataloaders(batch_size=64, data_root=data_root)
    model = model_fn(10).to(device)
    criterion = nn.CrossEntropyLoss()

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best = 0.0
    train_losses, train_accs, test_losses, test_accs = [], [], [], []
    t0 = time.time()

    for epoch in range(1, 11):
        tl, ta = train_one_epoch(model, loaders['train_loader'], criterion, optimizer, device)
        vl, va = evaluate(model, loaders['test_loader'], criterion, device)
        train_losses.append(tl); train_accs.append(ta)
        test_losses.append(vl); test_accs.append(va)
        if va > best:
            best = va
            torch.save(model.state_dict(), Path(save_dir) / "best_model.pth")

    dt = time.time() - t0
    print(f"  {label} seed={seed}: {best:.2f}% ({dt:.0f}s)")

    os.makedirs(Path(save_dir) / "logs", exist_ok=True)
    with open(Path(save_dir) / "logs" / "training_history.json", "w") as f:
        json.dump({"train_losses": train_losses, "train_accs": train_accs,
                    "test_losses": test_losses, "test_accs": test_accs,
                    "best_test_acc": best, "label": label, "seed": seed}, f, indent=2)
    return best


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent

    COMBOS = [
        # (label, model_fn, opt, lr, prefix, extra_note)
        ("Avg+SGD+Tanh (原论文)", lenet5_avg_tanh, "sgd", 0.01, "experiments_avg_sgd_tanh"),
        ("Avg+SGD+ReLU (池化对照)", lenet5_avg, "sgd", 0.01, "experiments_avg_sgd_relu"),
        ("Avg+Adam+Tanh (池化对照)", lenet5_avg_tanh, "adam", 0.001, "experiments_avg_adam_tanh"),
        # FC depth comparison (all AvgPool+SGD+Tanh backbone)
        ("1FC+SGD+Tanh (FC深度)", _tanh1fc, "sgd", 0.01, "experiments_1fc_sgd_tanh"),
        ("3FC+SGD+Tanh (FC深度)", _tanh3fc, "sgd", 0.01, "experiments_3fc_sgd_tanh"),
    ]

    SEEDS = [42, 123, 456]
    results = {}

    for label, model_fn, opt, lr, prefix in COMBOS:
        print(f"\n{'='*50}")
        print(f"  {label}  |  {opt} lr={lr}")
        print(f"{'='*50}")
        combo_results = []
        for seed in SEEDS:
            save_dir = f"{prefix}_seed{seed}"
            acc = run_combo(label, model_fn, opt, lr, seed,
                            str(ROOT / save_dir), str(ROOT / "data"))
            if acc:
                combo_results.append(acc)
        if combo_results:
            avg = sum(combo_results) / len(combo_results)
            results[label] = {"seeds": combo_results, "mean": avg}
            print(f"  => Mean: {avg:.2f}%")

    print(f"\n{'='*60}")
    print("ALL DONE — Summary:")
    for label, r in results.items():
        print(f"  {label}: {r['mean']:.2f}%  (seeds: {r['seeds']})")
