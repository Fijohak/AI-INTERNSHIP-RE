# MNIST 数据集来源说明

> 项目：LeNet-5 MNIST 手写数字识别 — 论文复现

## 数据集概述

| 属性 | 说明 |
|---|---|
| 名称 | MNIST (Modified National Institute of Standards and Technology) |
| 创建者 | Yann LeCun, Corinna Cortes, Christopher J.C. Burges |
| 主页 | http://yann.lecun.com/exdb/mnist/ |
| 任务 | 手写数字 0–9 分类（10 类） |
| 样本数 | 70,000 张（60,000 训练 / 10,000 测试） |
| 图像尺寸 | 28×28 像素，灰度图 |
| 像素范围 | 0–255（uint8），归一化后 ~ [-1, 1] |
| 数据格式 | IDX 二进制格式 |

## 获取方式

通过 `torchvision.datasets.MNIST` 自动下载：

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
```

- `root` 参数指定下载路径，首次运行自动从 http://yann.lecun.com/exdb/mnist/ 下载
- 下载后解压为 4 个 IDX 文件：训练图像、训练标签、测试图像、测试标签
- 后续运行直接从本地加载，无需重新下载

## 数据分布

| 类别 | 训练集 | 测试集 |
|---|---|---|
| 0 | 5,923 | 980 |
| 1 | 6,742 | 1,135 |
| 2 | 5,958 | 1,032 |
| 3 | 6,131 | 1,010 |
| 4 | 5,842 | 982 |
| 5 | 5,421 | 892 |
| 6 | 5,918 | 958 |
| 7 | 6,265 | 1,028 |
| 8 | 5,851 | 974 |
| 9 | 5,949 | 1,009 |
| **总计** | **60,000** | **10,000** |

各类别数据量基本均衡，无需额外的类别平衡处理。

## 归一化参数

MNIST 预计算的全局统计量（用于 Normalize）：

| 参数 | 值 |
|---|---|
| 均值 (μ) | 0.1307 |
| 标准差 (σ) | 0.3081 |

Normalize 的作用：将像素值从 [0, 1] 映射到近似标准正态分布 N(0, 1)，加速模型收敛。

## 许可证

MNIST 数据集由 NIST（美国国家标准与技术研究院）提供，可自由用于学术和研究目的。原始 NIST 数据集 SD-3 和 SD-1 经过混合和归一化处理形成 MNIST。

## 在项目中的使用

本项目中 MNIST 仅用于训练集和测试集划分（无独立验证集），DataLoader 默认配置：

```python
batch_size = 64
shuffle = True  # 训练集
shuffle = False # 测试集
num_workers = 2
```

> 注：由于项目 .gitignore 已排除 `data/` 目录，运行训练脚本前会自动下载 MNIST 数据集。
