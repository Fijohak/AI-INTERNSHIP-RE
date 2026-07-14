"""
compare_models.py —— 对比 ResNet18 和 ResNet50 在验证集上的表现

评估指标：
  1. Val Log Loss（CrossEntropyLoss，等价于 Kaggle 的 Multi-class Log Loss）
  2. Val Accuracy（Top-1）
  3. 模型参数量
  4. 模型文件大小
  5. 推理速度（单张图片平均耗时）
"""

import os
import time
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ============================================================
# 配置
# ============================================================

DATA_DIR = "dog-breed-identification"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

BATCH_SIZE = 32
NUM_WORKERS = 0
SEED = 42
VAL_SPLIT = 0.2
IMG_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class DogBreedDataset(Dataset):
    def __init__(self, samples, img_dir, transform=None):
        self.samples = samples
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class SubsetDataset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_id, label = self.base.samples[self.indices[idx]]
        img_path = os.path.join(self.base.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def load_model(model_name, weight_path, num_classes, device):
    """加载模型和权重"""
    model = getattr(models, model_name)(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def benchmark_inference_speed(model, loader, device, num_batches=50):
    """测试推理速度（单张图片平均耗时，ms）"""
    model.eval()
    times = []
    count = 0

    with torch.no_grad():
        for images, _ in loader:
            if count >= num_batches:
                break
            images = images.to(device)

            # Warmup (first batch)
            if count == 0:
                _ = model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            times.append(elapsed / images.size(0) * 1000)  # ms per image
            count += 1

    return sum(times) / len(times)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 准备验证集 ----
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    breed_list = list(sample_df.columns[1:])
    num_classes = len(breed_list)
    breed_to_idx = {breed: i for i, breed in enumerate(breed_list)}

    labels_df = pd.read_csv(LABELS_CSV)
    labels_df["label"] = labels_df["breed"].map(breed_to_idx)
    labels_df = labels_df.dropna(subset=["label"])
    labels_df["label"] = labels_df["label"].astype(int)
    samples = list(zip(labels_df["id"], labels_df["label"]))

    all_labels = [label for _, label in samples]
    _, val_indices = train_test_split(
        range(len(samples)),
        test_size=VAL_SPLIT,
        stratify=all_labels,
        random_state=SEED,
    )

    full_dataset = DogBreedDataset(samples, TRAIN_IMG_DIR, transform=None)
    val_dataset = SubsetDataset(full_dataset, val_indices, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    print(f"验证集: {len(val_dataset)} 张图片\n")

    criterion = nn.CrossEntropyLoss()

    # ---- 对比的模型列表 ----
    models_to_eval = [
        {
            "name": "ResNet18",
            "model_name": "resnet18",
            "weight_path": "model_resnet18.pth",
        },
        {
            "name": "ResNet50",
            "model_name": "resnet50",
            "weight_path": "model_resnet50.pth",
        },
    ]

    results = []

    for cfg in models_to_eval:
        weight_path = cfg["weight_path"]
        if not os.path.exists(weight_path):
            print(f"[跳过] {cfg['name']}: 权重文件 {weight_path} 不存在")
            continue

        print(f"{'='*60}")
        print(f"评估 {cfg['name']}")
        print(f"{'='*60}")

        # 加载模型
        model = load_model(cfg["model_name"], weight_path, num_classes, device)

        # 参数量
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {num_params/1e6:.2f}M")
        print(f"  可训练参数: {trainable_params/1e6:.2f}M")

        # 模型文件大小
        file_size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        print(f"  权重文件大小: {file_size_mb:.1f} MB")

        # 评估 log_loss 和 accuracy
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"  Val Log Loss: {val_loss:.6f}")
        print(f"  Val Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

        # 推理速度
        avg_ms = benchmark_inference_speed(model, val_loader, device)
        print(f"  推理速度: {avg_ms:.2f} ms/张")

        results.append({
            "模型": cfg["name"],
            "参数量": f"{num_params/1e6:.2f}M",
            "文件大小": f"{file_size_mb:.1f} MB",
            "Val Log Loss": f"{val_loss:.6f}",
            "Val Accuracy": f"{val_acc*100:.2f}%",
            "推理速度": f"{avg_ms:.2f} ms/张",
        })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- 输出对比表 ----
    if len(results) >= 2:
        print("\n" + "=" * 80)
        print("对比结果汇总")
        print("=" * 80)
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        # Winner analysis
        r18 = {r["模型"]: r for r in results}["ResNet18"]
        r50 = {r["模型"]: r for r in results}["ResNet50"]

        loss_18 = float(r18["Val Log Loss"])
        loss_50 = float(r50["Val Log Loss"])
        acc_18 = float(r18["Val Accuracy"].rstrip('%'))
        acc_50 = float(r50["Val Accuracy"].rstrip('%'))
        speed_18 = float(r18["推理速度"].rstrip(' ms/张'))
        speed_50 = float(r50["推理速度"].rstrip(' ms/张'))

        print(f"\n=== 关键对比 ===")
        print(f"  Log Loss:   ResNet18={loss_18:.6f}  vs  ResNet50={loss_50:.6f}  "
              f"({'ResNet50 更优' if loss_50 < loss_18 else 'ResNet18 更优'})")
        print(f"  Accuracy:   ResNet18={acc_18:.2f}%  vs  ResNet50={acc_50:.2f}%  "
              f"({'ResNet50 更优' if acc_50 > acc_18 else 'ResNet18 更优'})")
        print(f"  推理速度:   ResNet18={speed_18:.2f}ms  vs  ResNet50={speed_50:.2f}ms  "
              f"({'ResNet18 更快' if speed_18 < speed_50 else 'ResNet50 更快'})")
        print(f"  模型大小:   ResNet18={r18['文件大小']}  vs  ResNet50={r50['文件大小']}")

    print("\n完成!")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
