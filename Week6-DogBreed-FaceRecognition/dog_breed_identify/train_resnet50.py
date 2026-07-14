"""
train.py —— 狗品种识别训练脚本

完整流程（对应你列的每一步）：
  ① 读取 labels.csv → 建立 "品种名 → 整数标签" 映射
  ② 拼出 train/{id}.jpg 图片路径
  ③ Stratified split：保证每个品种在训练/验证集的比例都是 80/20
  ④ 阶段一：冻结 backbone，只训练最后分类头
  ⑤ 阶段二：解冻 backbone，用小学习率微调全部参数
  ⑥ 验证集计算 log_loss（CrossEntropyLoss 在数学上等价于 log loss）
  ⑦ 保存模型和品种列表，供 predict.py 使用
"""

import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # 用于分层抽样

# ============================================================
# 0. 配置参数 —— 需要改路径就改这里
# ============================================================

DATA_DIR = "dog-breed-identification"

TRAIN_IMG_DIR        = os.path.join(DATA_DIR, "train")
LABELS_CSV           = os.path.join(DATA_DIR, "labels.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

MODEL_NAME    = "resnet50"    # 可选: resnet18, resnet34, resnet50, resnet101
# 超参数
BATCH_SIZE    = 8         # ResNet50 显存需求更大，从 16 降为 8
NUM_WORKERS   = 0         # Windows 必须为 0，否则多进程报错
SEED          = 42

# 阶段一：冻结 backbone，只训练分类头
STAGE1_EPOCHS = 2         # 训练轮数
STAGE1_LR     = 1e-3      # 学习率可以大一点，因为只训练全新初始化的 fc 层

# 阶段二：解冻 backbone，小学习率微调
STAGE2_EPOCHS = 3         # 训练轮数
STAGE2_LR     = 1e-5      # backbone 的学习率（很小，防止破坏预训练特征）
STAGE2_FC_LR  = 1e-4      # 分类头的学习率（比 backbone 大 10 倍）

VAL_SPLIT = 0.2           # 验证集比例

IMG_SIZE = 224            # ResNet 需要的输入尺寸

# 输出文件
MODEL_SAVE_PATH = "model.pth"

# ============================================================
# 自定义 Dataset 类（模块级别，供 main() 使用）
# ============================================================

class DogBreedDataset(Dataset):
    """
    给定 (id, label) 列表和图片目录，返回 (图片张量, 标签)
    ② 拼图片路径在这里：train/{id}.jpg
    """

    def __init__(self, samples, img_dir, transform=None):
        self.samples   = samples   # [(img_id, label), ...]
        self.img_dir   = img_dir
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
    """从完整 Dataset 中按索引取子集，并覆盖 transform"""

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


def main():
    # ============================================================
    # 1. 设定随机种子
    # ============================================================

    random.seed(SEED)
    torch.manual_seed(SEED)

    # ============================================================
    # 2. 自动选择设备（GPU / CPU）
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 使用设备: {device}")
    if device.type == "cuda":
        print(f"[信息] GPU 型号: {torch.cuda.get_device_name(0)}")

    # ============================================================
    # ① 读取 labels.csv
    #    - 每一行: {图片id},{品种名}
    #    - 收集所有品种名 → 排序 → 映射成 0~119 的整数标签
    # ============================================================

    print("[信息] ① 正在读取 labels.csv 和 sample_submission.csv ...")

    # 读取 sample_submission.csv 获取品种列顺序（这是唯一的品种顺序来源）
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    breed_list = list(sample_df.columns[1:])  # 去掉第一列 "id"
    NUM_CLASSES = len(breed_list)
    print(f"     sample_submission 品种数: {NUM_CLASSES}")

    # 建立映射：品种名 → 整数标签（按 sample_submission 的列顺序）
    breed_to_idx = {breed: i for i, breed in enumerate(breed_list)}
    idx_to_breed = {i: breed for i, breed in enumerate(breed_list)}

    # 读取 labels.csv，建立 (图片id, label) 列表
    labels_df = pd.read_csv(LABELS_CSV)  # 列: id, breed
    labels_df["label"] = labels_df["breed"].map(breed_to_idx)

    # 检查是否有 breed 不在 sample_submission 中
    missing = labels_df["label"].isna().sum()
    if missing > 0:
        print(f"     [警告] {missing} 个品种在 sample_submission.csv 中未找到，将被跳过")
        labels_df = labels_df.dropna(subset=["label"])
    labels_df["label"] = labels_df["label"].astype(int)

    print(f"     图片数: {len(labels_df)}，品种数: {NUM_CLASSES}")

    # 转成 samples 列表
    samples = list(zip(labels_df["id"], labels_df["label"]))

    # ============================================================
    # ③ Stratified split（分层抽样）
    #    关键区别：
    #    - 随机 split：每种狗随机分，运气差的话某品种在验证集里可能一张都没有
    #    - Stratified split：按品种比例分，训练/验证集里每个品种都是 80/20
    #    使用 sklearn 的 train_test_split，传入 stratify=labels
    # ============================================================

    print("[信息] ③ 正在进行 Stratified split ...")

    all_labels = [label for _, label in samples]  # 每个样本的标签

    # stratify=all_labels 是核心：保证每个类别的比例在 train/val 中一致
    train_indices, val_indices = train_test_split(
        range(len(samples)),
        test_size=VAL_SPLIT,
        stratify=all_labels,
        random_state=SEED,
    )

    # 验证一下分层效果
    train_labels = [all_labels[i] for i in train_indices]
    val_labels   = [all_labels[i] for i in val_indices]
    print(f"     训练集: {len(train_indices)} 张, 验证集: {len(val_indices)} 张")
    # 对比随机 split，每种狗在验证集中的比例都接近 20%
    for breed_idx in range(min(5, NUM_CLASSES)):  # 只展示前 5 个品种
        train_count = train_labels.count(breed_idx)
        val_count   = val_labels.count(breed_idx)
        total = train_count + val_count
        print(f"     {idx_to_breed[breed_idx]:30s}: "
              f"train={train_count}/{total} ({train_count/total:.0%}), "
              f"val={val_count}/{total} ({val_count/total:.0%})")

    # ============================================================
    # 数据增强 / 预处理
    # ============================================================

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    # 训练集：随机增强 → 让模型看到更多样化的图片
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 验证集：只做固定预处理 → 保证评估公平
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    full_dataset = DogBreedDataset(samples, TRAIN_IMG_DIR, transform=None)
    train_dataset = SubsetDataset(full_dataset, train_indices, train_transform)
    val_dataset   = SubsetDataset(full_dataset, val_indices,   val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    # ============================================================
    # 辅助函数：训练一个 epoch
    # ============================================================

    def train_one_epoch(model, loader, criterion, optimizer, epoch, stage_name):
        """训练一个 epoch，返回平均 loss 和 accuracy"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"[{stage_name}] Epoch {epoch} Train")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "acc": f"{correct/total:.4f}"})

        return total_loss / total, correct / total

    # ============================================================
    # 辅助函数：验证
    # ============================================================

    @torch.no_grad()
    def validate(model, loader, criterion, epoch, stage_name):
        """
        ⑥ 验证集计算 log_loss

        CrossEntropyLoss 的数学公式:
          loss = -log(softmax(output)[true_class])

        这和 Kaggle 用的 Multi-class Log Loss 完全等价：
          log_loss = -1/N * Σ Σ y_ij * log(p_ij)

        所以这里直接用 CrossEntropyLoss 就可以了。
        """
        model.eval()
        total_loss = 0.0   # 这就是 log_loss（所有样本累加）
        correct = 0
        total = 0

        pbar = tqdm(loader, desc=f"[{stage_name}] Epoch {epoch} Val  ")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)   # ← 这就是 log_loss

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"log_loss": f"{loss.item():.4f}",
                              "acc": f"{correct/total:.4f}"})

        return total_loss / total, correct / total

    # ============================================================
    # 构建模型
    # ============================================================

    print(f"[信息] 加载预训练模型 {MODEL_NAME} ...")
    # 动态选择模型：支持 resnet18/34/50/101
    model = getattr(models, MODEL_NAME)(weights="IMAGENET1K_V1")

    # 替换最后一层分类头：1000（ImageNet 类别）→ 120（犬种）
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)

    print(f"     参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 损失函数（等价于 log loss）
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # ④ 阶段一：冻结 backbone，只训练分类头
    #
    #    为什么要这样做？
    #    - 预训练 backbone 已经在 ImageNet（1400 万张图，1000 类）
    #      上学会了识别边缘、纹理、形状等通用视觉特征
    #    - 新换上的 fc 层参数是随机初始化的，什么都不懂
    #    - 如果一开始就训练所有参数，随机梯度会破坏 backbone 的预训练知识
    #    - 正确做法：先"教会"分类头怎么从已有特征分辨 120 种狗，
    #      等分类头学得差不多了，再微调 backbone
    # ============================================================

    print("\n" + "=" * 60)
    print("④ 阶段一：冻结 backbone，只训练分类头")
    print("=" * 60)

    # 冻结 backbone：把所有参数的 requires_grad 设为 False
    for param in model.parameters():
        param.requires_grad = False
    # 解冻 fc 层（分类头）：只让 fc 层的参数可以训练
    for param in model.fc.parameters():
        param.requires_grad = True

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"     可训练参数: {trainable_params:,} / {total_params:,}"
          f" ({trainable_params/total_params*100:.1f}%)")

    optimizer_stage1 = optim.AdamW(model.fc.parameters(), lr=STAGE1_LR)

    for epoch in range(1, STAGE1_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer_stage1, epoch, "Stage1")
        val_loss, val_acc = validate(model, val_loader, criterion,
                                      epoch, "Stage1")
        print(f"  → Stage1 Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val log_loss={val_loss:.4f} Acc={val_acc:.4f}")

    # ============================================================
    # ⑤ 阶段二：解冻 backbone，小学习率微调
    #
    #    为什么学习率要调小？
    #    - backbone 已经有很好的特征提取能力，只需要微调
    #    - 如果学习率太大，一个大步就毁掉了 backbone 的知识
    #    - 分类头还是用稍微大一点的学习率，因为它是从阶段一继续学
    # ============================================================

    print("\n" + "=" * 60)
    print("⑤ 阶段二：解冻 backbone，用小学习率微调全部参数")
    print("=" * 60)

    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    # 使用分组学习率：
    #   - backbone（参数名中不含 'fc' 的层）：学习率 1e-5（很小）
    #   - 分类头（fc 层）：学习率 1e-4（是 backbone 的 10 倍）
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    fc_params = list(model.fc.parameters())

    optimizer_stage2 = optim.AdamW([
        {"params": backbone_params, "lr": STAGE2_LR},
        {"params": fc_params,       "lr": STAGE2_FC_LR},
    ])

    print(f"     backbone 学习率: {STAGE2_LR}")
    print(f"     分类头 学习率: {STAGE2_FC_LR}")

    best_val_loss = float("inf")

    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer_stage2, epoch, "Stage2")
        val_loss, val_acc = validate(model, val_loader, criterion,
                                      epoch, "Stage2")
        print(f"  → Stage2 Epoch {epoch}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val log_loss={val_loss:.4f} Acc={val_acc:.4f}")

        # 保存验证集 log_loss 最低的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"     OK 模型已保存（Val log_loss={val_loss:.4f}）")

    print(f"\n[信息] 训练完成！最佳验证 log_loss: {best_val_loss:.4f}")
    print(f"[信息] 模型: {MODEL_SAVE_PATH}")
    print(f"[信息] 品种顺序来源: {SAMPLE_SUBMISSION_CSV}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
