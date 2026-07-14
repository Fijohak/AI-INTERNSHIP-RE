# 今日修改摘要

> 供导师汇报使用 — 从 P0 到 P4 的全部改动

---

## 一、实验报告重构（P0）

**修改文件**：`report/experiment_report.md`

原报告 ~135 行，仅含基础结果 + 1 组对照实验。重构后约 500 行，新增章节：

| 新增 | 内容 |
|---|---|
| §0 摘要 | 全文一句话概述 |
| §1 引言 | 扩充为 1.1 原文概述 + 1.2 复现思路（含现代化调整对照表）+ 1.3 改进方法 |
| §2 相关工作 | CNN 发展脉络 + MNIST 基准简介 |
| §4 实验设计 | 评价指标 + 8 组对照实验矩阵 + 实验环境 |
| §5.1 基础结果 | 增加 epoch-by-epoch 训练表 + 训练曲线分析 |
| §5.2 池化对照 | 增加 epoch 逐行对比表 |
| §6 讨论 | 扩充，新增局限性(6.4) + 与原论文对比表(6.5) |
| §7 结论与展望 | 全新章节，四点结论 + 五点未来方向 |
| §8 问题记录 | 新增 WinError 1455 的解决方案 |
| 混淆矩阵 | 完整 10×10 数值表 + Top-5 易混淆对 + 错误样本分析 |

---

## 二、新增 6 组对照实验（P1–P3）

原项目仅 1 组对照实验（MaxPool vs AvgPool），本次增至 **7 组**，累计 **23 次训练**：

### P1 — 优化器 & 激活函数对比（+4 次训练）

| 新增文件 | 用途 |
|---|---|
| `src/model_tanh.py` | Tanh 激活函数版 LeNet-5 |
| `src/train_tanh.py` | Tanh 版训练脚本 |

| 实验 | 对照组 | 实验组 | 关键发现 |
|---|---|---|---|
| Exp-2 优化器 | Adam 99.04% | SGD 99.11% | SGD 略优，但需 lr=0.01 |
| Exp-3 激活函数 | ReLU 99.04% | Tanh 98.76% | ReLU 泛化更好，Tanh 易过拟合 |

### P2 — 多种子稳定性 & 学习率对比（+11 次训练）

| 新增目录 | 说明 |
|---|---|
| `experiments_seed42_r2/`, `r3/` | seed=42 第 2、3 轮 |
| `experiments_seed123_r1/`, `r2/`, `r3/` | seed=123 三轮 |
| `experiments_seed456_r1/`, `r2/`, `r3/` | seed=456 三轮 |
| `experiments_lr_high/`, `experiments_lr_low/` | lr=0.01, 0.0001 |

| 实验 | 关键发现 |
|---|---|
| Exp-4 多种子稳定性 | 9 次训练均值 99.04% ± 0.07%，结果高度可信 |
| Exp-5 学习率对比 | lr=0.001 是甜点；0.01 震荡（98.24%），0.0001 太慢（98.44%） |

### P3 — 深度分析实验（+4 次训练）

| 新增文件 | 用途 |
|---|---|
| `src/model_mlp.py` + `src/train_mlp.py` | MLP Baseline 对比 |
| `src/dataset_aug.py` + `src/train_aug.py` | 数据增强训练 |
| `src/visualize_features.py` | 卷积层特征图可视化 |

| 实验 | 对照组 | 实验组 | 关键发现 |
|---|---|---|---|
| Exp-6 CNN vs MLP | LeNet-5 99.04% | MLP 97.38% | **同参数量下 CNN 错误率仅为 MLP 的 37%** |
| Exp-7 数据增强 | 无增强 99.04% | 增强 **99.21%** | 本项目最高分，错误率下降 17.7% |
| Exp-8 Batch Size | bs=64 99.04% | bs=16/256 各 98.93% | 64 是最优平衡点 |
| 特征图可视化 | — | — | C1→边缘，C3→形状，C5→全局，验证层次化特征提取 |

---

## 三、PDF 生成（P4）

- 安装 MiKTeX + 配置中文支持（xelatex + ctex）
- 修复 Unicode 字符（β₁ 等希腊字母 → LaTeX 数学模式）
- 添加 YAML 封面元数据（标题、副标题、作者、日期、摘要）
- 产出 `report/experiment_report2.0.pdf`（680KB，含目录、图片、公式）

---

## 四、新增项目文件一览

```
新增文件：
├── src/model_tanh.py          ← Tanh 激活函数模型
├── src/train_tanh.py          ← Tanh 训练脚本
├── src/model_mlp.py           ← MLP Baseline 模型
├── src/train_mlp.py           ← MLP 训练脚本
├── src/dataset_aug.py         ← 数据增强变换
├── src/train_aug.py           ← 数据增强训练脚本
├── src/visualize_features.py  ← 特征图可视化
├── notes/teaching_guide.md    ← 教学指南
│
新增实验目录（15 个）：
├── experiments_sgd/           ← SGD 优化器结果
├── experiments_tanh/          ← Tanh 激活函数结果
├── experiments_seed42_r2,r3/  ← 稳定性分析
├── experiments_seed123_r1,r2,r3/
├── experiments_seed456_r1,r2,r3/
├── experiments_lr_high/       ← 学习率 0.01
├── experiments_lr_low/        ← 学习率 0.0001
├── experiments_mlp/           ← MLP Baseline
├── experiments_aug/           ← 数据增强
├── experiments_bs16,bs256/    ← Batch Size

修改文件：
└── report/experiment_report.md ← 核心改动（135 行 → ~500 行）
```

---

## 五、汇报用一句话总结

> 在复现 LeNet-5 的基础上，新增了 6 组对照实验（23 次训练），覆盖优化器、激活函数、学习率、Batch Size、MLP Baseline 对比和数据增强。最佳模型（数据增强）准确率 99.21%，并通过 9 次多种子重复实验验证了结果稳定性（标准差 0.07%）。其中 CNN vs MLP 的对比最具说服力——同等参数量下，卷积网络的错误率仅为全连接网络的 37%。
