# Dog Breed Identification —— Kaggle 图像分类 Baseline

基于 PyTorch + ResNet18 迁移学习的狗品种识别项目，用于 Kaggle 竞赛 [Dog Breed Identification](https://www.kaggle.com/c/dog-breed-identification)。

---

## 项目概览

| 项目 | 详情 |
|------|------|
| 任务类型 | 多类别图像分类（120 种狗品种） |
| 框架 | PyTorch 2.x + torchvision |
| 骨干网络 | ResNet18（ImageNet-1K 预训练） |
| 训练策略 | 两阶段迁移学习（冻结 backbone → 全参数微调） |
| 数据划分 | Stratified 分层抽样（80/20） |
| 评估指标 | Multi-class Log Loss（= CrossEntropyLoss） |
| 输出格式 | Kaggle submission.csv（10357 行 × 121 列） |

---

## 数据集

| 数据集 | 数量 | 说明 |
|--------|------|------|
| 训练集 | 10,222 张 | 含标签，来自 `labels.csv` |
| 测试集 | 10,357 张 | 无标签，用于生成提交文件 |
| 类别数 | 120 种 | 按字母排序（与 sample_submission.csv 列顺序一致） |

数据来源：需从 Kaggle 下载，解压后放入 `dog-breed-identification/` 目录。

### 标签格式

`labels.csv`（训练集标签）：
```
id,breed
000bec180eb18c7604dcecc8fe0dba07,boston_bull
001513dfcb2ffafc82cccf4d8bbaba97,dingo
...
```

`sample_submission.csv`（提交模板，每列一个品种的概率）：
```
id,affenpinscher,afghan_hound,...,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.00833,0.00833,...,0.00833
...
```

---

## 项目结构

```
dog_breed_identify/
├── dog-breed-identification/      # Kaggle 数据集（需自行下载）
│   ├── train/                     # 10,222 张训练图片（.jpg）
│   ├── test/                      # 10,357 张测试图片（.jpg）
│   ├── labels.csv                 # 训练集标签（id, breed）
│   └── sample_submission.csv      # Kaggle 提交模板（120 个品种列）
├── train.py                       # 训练脚本（两阶段迁移学习）
├── predict.py                     # 预测脚本（生成 submission.csv）
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

训练后生成：
- `model.pth` —— 模型权重（保存在验证集上 log_loss 最低的那个 epoch）
- `submission.csv` —— Kaggle 提交文件

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包：

| 包 | 版本 | 用途 |
|----|------|------|
| torch | ≥2.0.0 | 深度学习框架 |
| torchvision | ≥0.15.0 | 预训练模型 + 图像变换 |
| pandas | ≥1.5.0 | CSV 读写 |
| scikit-learn | ≥1.0.0 | 分层抽样（train_test_split） |
| Pillow | ≥9.0.0 | 图像读取 |
| tqdm | ≥4.64.0 | 进度条 |

### 2. 训练模型

```bash
python train.py
```

训练分为两个阶段（详见下文"训练策略"），每个阶段都会显示：
- 实时 batch 级别的 loss 和 accuracy
- 每个 epoch 结束后的训练集和验证集指标（log_loss + accuracy）

训练完成后生成 `model.pth`（最佳验证集 log_loss 对应的权重）。

### 3. 生成预测结果

```bash
python predict.py
```

生成 `submission.csv`，格式与 `sample_submission.csv` 完全一致，可直接提交到 Kaggle。

---

## 训练策略（核心设计）

### 两阶段迁移学习

这是本项目最核心的设计思路，分为两个阶段：

#### 阶段一：冻结 Backbone，只训练分类头

| 参数 | 值 |
|------|-----|
| Epochs | 2 |
| 学习率 | 1e-3（AdamW） |
| 可训练参数 | 仅 `fc` 层（约 61K / 11.18M ≈ 0.5%） |

**设计原因**：
- ResNet18 的 backbone 在 ImageNet（1400 万张图，1000 类）上预训练，已学会识别边缘、纹理、形状等通用视觉特征
- 新换上的 `fc` 层参数是随机初始化的，什么都不懂
- 如果一开始就训练所有参数，随机梯度会破坏 backbone 的预训练知识
- 正确做法：先让分类头学会从已有的预训练特征中分辨 120 种狗

```python
# 冻结 backbone
for param in model.parameters():
    param.requires_grad = False
# 只解冻分类头
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = AdamW(model.fc.parameters(), lr=1e-3)
```

#### 阶段二：解冻 Backbone，全参数微调

| 参数 | 值 |
|------|-----|
| Epochs | 3 |
| Backbone 学习率 | 1e-5（极小） |
| FC 层学习率 | 1e-4（backbone 的 10 倍） |
| 可训练参数 | 全部 11.18M |

**设计原因**：
- backbone 已经有很好的特征提取能力，只需要用较小学习率微调
- 如果学习率太大，一个大步就会破坏 backbone 的知识
- 分类头使用比 backbone 大 10 倍的学习率，因为它需要更快地从阶段一的基础上继续学

```python
# 分组学习率
optimizer = AdamW([
    {"params": backbone_params, "lr": 1e-5},   # backbone 微调
    {"params": fc_params,       "lr": 1e-4},   # 分类头
])
```

**总计：2 + 3 = 5 个 epoch**

### 模型保存策略

不是保存最后一个 epoch，而是保存**验证集 log_loss 最低**的 epoch：
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), "model.pth")
```

---

## 数据预处理

### 训练集数据增强

```python
transforms.Compose([
    RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪 + 缩放
    RandomHorizontalFlip(p=0.5),                 # 随机水平翻转
    ToTensor(),                                  # 转为张量
    Normalize(mean=[0.485, 0.456, 0.406],       # ImageNet 标准化
              std=[0.229, 0.224, 0.225]),
])
```

### 验证集 / 测试集预处理（无随机增强）

```python
transforms.Compose([
    Resize(256),           # 先缩放到 256×256
    CenterCrop(224),       # 中心裁剪到 224×224
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])
```

---

## 数据集划分：Stratified Split

使用 `sklearn.model_selection.train_test_split` 的 `stratify` 参数进行**分层抽样**：

```python
train_indices, val_indices = train_test_split(
    range(len(samples)),
    test_size=0.2,
    stratify=all_labels,    # 关键：按标签比例分层
    random_state=42,
)
```

**为什么不用随机 split？**
- 随机 split 运气差的话，某个品种在验证集中可能一张都没有
- Stratified split 保证：每个品种的训练/验证集比例严格为 80/20
- 代码会打印前 5 个品种的分布来验证分层效果

---

## 标签映射

品种顺序来源于 `sample_submission.csv` 的列顺序（按字母排序），而不是 `labels.csv`：

```python
sample_df = pd.read_csv("sample_submission.csv")
breed_list = list(sample_df.columns[1:])  # 去掉 "id" 列
breed_to_idx = {breed: i for i, breed in enumerate(breed_list)}  # 120 个品种 → 0~119
```

这样确保了训练时的类别索引和提交文件中的列顺序完全一致。

---

## 损失函数：CrossEntropyLoss = Multi-class Log Loss

Kaggle 使用 Multi-class Log Loss 作为评估指标：

$$\text{log loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} y_{ij} \cdot \log(p_{ij})$$

PyTorch 的 `CrossEntropyLoss` 公式：

$$\text{CE} = -\log\left(\frac{e^{z_{y}}}{\sum_{j} e^{z_j}}\right) = -\log(\text{softmax}(z)_y)$$

两者在数学上完全等价（标签为 one-hot 编码时），所以直接用 `nn.CrossEntropyLoss()` 即可。

---

## 预测流程（predict.py）

1. **品种列表读取** —— 从 `sample_submission.csv` 读取 120 个品种的列名（和训练时相同的来源）
2. **模型加载** —— 创建相同结构的 ResNet18 + 加载 `model.pth` 权重 + `model.eval()`
3. **测试集扫描** —— 遍历 `test/` 目录，收集所有 `.jpg` 文件名（去掉后缀作为 id）
4. **批量推理** —— 使用 `DataLoader` 按 batch 做前向传播，通过 `Softmax` 得到概率分布
5. **生成 submission.csv** —— 每行格式为 `[id, prob_breed1, prob_breed2, ..., prob_breed120]`，列名和列顺序与 `sample_submission.csv` 严格一致

```python
# 核心推理逻辑
with torch.no_grad():
    for images, img_ids in test_loader:
        outputs = model(images.to(device))
        probs = softmax(outputs)        # 120 个类别的概率分布
        all_probs.append(probs.cpu())
        all_ids.extend(img_ids)
```

---

## 模型架构细节

| 组件 | 详情 |
|------|------|
| 骨干网络 | ResNet18（`torchvision.models.resnet18`） |
| 预训练权重 | IMAGENET1K_V1（Top-1 Acc: 69.76%） |
| 输入尺寸 | 224 × 224 × 3 |
| 原始输出 | 1000 类（ImageNet） |
| 修改后输出 | 120 类（替换 `fc` 层） |
| 总参数量 | ~11.18M |
| FC 层参数量 | 61,440（512 × 120 + 120） |

```python
model = models.resnet18(weights="IMAGENET1K_V1")
in_features = model.fc.in_features       # 512
model.fc = nn.Linear(in_features, 120)   # 替换全连接层
```

---

## 完整超参数汇总

| 超参数 | 值 | 说明 |
|--------|-----|------|
| BATCH_SIZE（训练） | 16 | 笔记本 GPU 保守设置 |
| BATCH_SIZE（预测） | 32 | 推理可适当增大 |
| NUM_WORKERS | 0 | Windows 必须为 0 |
| SEED | 42 | 随机种子 |
| IMG_SIZE | 224 | ResNet 标准输入尺寸 |
| VAL_SPLIT | 0.2 | 验证集比例 |
| STAGE1_EPOCHS | 2 | 阶段一训练轮数 |
| STAGE1_LR | 1e-3 | 阶段一学习率 |
| STAGE2_EPOCHS | 3 | 阶段二训练轮数 |
| STAGE2_LR (backbone) | 1e-5 | 骨干网络微调学习率 |
| STAGE2_FC_LR | 1e-4 | 分类头学习率 |
| 优化器 | AdamW | 带权重衰减的 Adam |
| 损失函数 | CrossEntropyLoss | 等价于 Multi-class Log Loss |
| 数据增强（训练） | RandomResizedCrop + RandomHorizontalFlip | — |
| 预处理（验证/测试） | Resize(256) → CenterCrop(224) | 无随机性，保证评估公平 |

---

## 预期结果

| 指标 | 预期值 |
|------|--------|
| 验证集准确率 | 40%~60%（仅 5 个 epoch，未调参） |
| 验证集 log_loss | 随训练下降 |

可进一步提升的方向：
- 增加 epoch 数（如 Stage1=5, Stage2=10）
- 使用更大的模型（ResNet50、EfficientNet、ViT）
- 添加更多数据增强（ColorJitter、Rotation、MixUp/CutMix）
- 学习率调度器（CosineAnnealingLR、ReduceLROnPlateau）
- 交叉验证
- 模型集成

---

## 技术细节备注

1. **Windows 兼容性**：`NUM_WORKERS=0` 避免多进程报错；`multiprocessing.freeze_support()` 防止 PyInstaller 打包时的问题
2. **CUDA 优化**：`pin_memory=True`（仅 CUDA 可用时），加速 CPU→GPU 数据传输
3. **GPU/CPU 自适应**：自动检测 CUDA 可用性，CPU 上也能运行（只是慢）
4. **训练时实时监控**：通过 tqdm 进度条实时显示 loss 和 accuracy
5. **自定义 Dataset 类**：
   - `DogBreedDataset`：完整数据集（所有图片 + 标签）
   - `SubsetDataset`：从完整数据集的指定索引取子集，并覆盖 transform
   - `TestDataset`：测试集 Dataset，返回 (图片张量, 图片id)

---

## 如果遇到问题

### 路径不对
修改 `train.py` 和 `predict.py` 顶部的 `DATA_DIR` 变量。

### 显存不够
把 `BATCH_SIZE` 改小（比如 8 或 4）。

### Windows 上报错
确保 `NUM_WORKERS` 设置为 0。

### 下载预训练权重太慢
首次运行会自动下载 ResNet18 预训练权重（约 45MB），需要联网。
如果下载失败，可手动下载后放到 `C:\Users\<用户名>\.cache\torch\hub\checkpoints\`。
