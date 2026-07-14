"""
predict.py —— 狗品种识别预测脚本

功能：
  1. 加载训练好的模型权重
  2. 读取 test 文件夹中的所有图片
  3. 对每张图片预测 120 个品种的概率
  4. 按照 sample_submission.csv 的格式生成 submission.csv

注意：
  - 模型的类别顺序和 breed_list.txt 一致
  - sample_submission.csv 的列顺序和 breed_list.txt 一致（都是字母排序）
  - 如果不一致也不会出错，因为我们按 breed_list.txt 的列名去生成结果

使用方法：
  python predict.py
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ============================================================
# 0. 配置 —— 和 train.py 一样的路径设置
# ============================================================

DATA_DIR = "dog-breed-identification"

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test")

LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

MODEL_NAME    = "resnet18"    # 可选: resnet18, resnet34, resnet50, resnet101
BATCH_SIZE   = 32
NUM_WORKERS  = 0

MODEL_LOAD_PATH  = "model.pth"
SUBMISSION_PATH  = "submission.csv"

IMG_SIZE = 224

# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ============================================================
# 测试集 Dataset 和预处理（模块级别）
# ============================================================

class TestDataset(Dataset):
    """测试集 Dataset：返回 (图片张量, 图片id)"""

    def __init__(self, img_dir, img_ids, transform=None):
        self.img_dir   = img_dir
        self.img_ids   = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_id


# 测试集预处理（和验证集一样，无随机增强）
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def main():
    # ============================================================
    # 1. 自动选择设备
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 使用设备: {device}")

    # ============================================================
    # 2. 从 sample_submission.csv 读取品种列表（和训练时相同的来源）
    # ============================================================

    print("[信息] 正在从 sample_submission.csv 读取品种列表 ...")
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    breed_list = list(sample_df.columns[1:])  # 去掉第一列 "id"
    NUM_CLASSES = len(breed_list)
    print(f"[信息] 品种数: {NUM_CLASSES}")

    # ============================================================
    # 3. 加载模型
    #    - 用相同的网络结构
    #    - 加载训练好的权重
    #    - 切换到 eval 模式
    # ============================================================

    print("[信息] 正在加载模型 ...")

    # 创建和训练时相同的网络结构
    model = getattr(models, MODEL_NAME)(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    # 加载训练好的权重
    state_dict = torch.load(MODEL_LOAD_PATH, map_location=device)
    model.load_state_dict(state_dict)

    # 模型搬到设备，并切换到评估模式
    model = model.to(device)
    model.eval()

    print("[信息] 模型加载完成")

    # ============================================================
    # 4. 定义测试集 DataLoader
    # ============================================================

    # 获取所有 test 图片的 id（去掉 .jpg 后缀）
    print("[信息] 正在扫描测试图片 ...")
    test_ids = []
    for fname in os.listdir(TEST_IMG_DIR):
        if fname.endswith(".jpg"):
            test_ids.append(fname[:-4])

    test_ids = sorted(test_ids)
    print(f"[信息] 共 {len(test_ids)} 张测试图片")

    test_dataset = TestDataset(TEST_IMG_DIR, test_ids, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # ============================================================
    # 5. 对测试集做预测
    # ============================================================

    print("[信息] 正在进行预测 ...")

    softmax = nn.Softmax(dim=1)

    all_probs = []
    all_ids   = []

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader, desc="Predict"):
            images = images.to(device)

            outputs = model(images)
            probs = softmax(outputs)

            all_probs.append(probs.cpu())
            all_ids.extend(img_ids)

    all_probs = torch.cat(all_probs, dim=0)
    print(f"[信息] 预测完成，共 {all_probs.shape[0]} 张图片，{all_probs.shape[1]} 个类别")

    # ============================================================
    # 6. 生成 submission.csv
    # ============================================================

    print("[信息] 正在生成 submission.csv ...")

    sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)

    breed_to_col = {}
    for col_idx, breed in enumerate(sample_df.columns[1:], start=1):
        breed_to_col[breed] = col_idx

    print(f"[信息] sample_submission 有 {len(sample_df.columns) - 1} 个品种列")

    submission_rows = []

    for i, img_id in enumerate(all_ids):
        row = [img_id] + [0.0] * NUM_CLASSES
        for class_idx, breed_name in enumerate(breed_list):
            col = breed_to_col.get(breed_name)
            if col is not None:
                row[col] = float(all_probs[i, class_idx])
        submission_rows.append(row)

    submission_df = pd.DataFrame(submission_rows, columns=sample_df.columns.tolist())
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"[信息] submission.csv 已保存到: {SUBMISSION_PATH}")
    print(f"[信息] 行数: {len(submission_df)}, 列数: {len(submission_df.columns)}")

    print(f"[信息] 前 3 行预览:")
    print(submission_df.head(3).to_string())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
