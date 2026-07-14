# LeNet-5 MNIST 复现 — 实验报告

> 完整报告见 [report/experiment_report.md](report/experiment_report.md)
> PDF 版本: [report/experiment_report2.0.pdf](report/experiment_report2.0.pdf)

## 摘要

本项目使用 PyTorch 复现了 LeCun 等人 1998 年提出的 LeNet-5 卷积神经网络，在 MNIST 手写数字数据集上完成 0–9 分类任务。通过 9 组对照实验共 23 次训练，系统评估了池化方法、优化器、激活函数、FC 深度、学习率、Batch Size、数据增强和多种子稳定性的影响。

## 核心结果

| 实验 | 配置 | 最佳测试准确率 |
|---|---|---|
| 基线 | MaxPool + Adam + ReLU | 99.04% |
| 最佳 | MaxPool + SGD + ReLU | **99.12%** |
| 数据增强 | 基线 + RandomRotation/RandomAffine | **99.21%** |
| AvgPool | 原论文池化设定 | 99.00% |
| MLP Baseline | 同参数量全连接 | 97.38% |

## 关键发现

1. **激活函数是压倒性的第一因子**：ReLU 比 Tanh 稳定提升 0.16–0.25%，是唯一值得做的架构改动
2. **池化方法和优化器的选择影响极小**（< 0.08%）：AvgPool 和 MaxPool 几乎等价，Adam 和 SGD 在最优 lr 下无显著差异
3. **CNN 在同等参数量下远优于 MLP**：错误率仅为 MLP 的 37%，完美复现了 LeCun 1998 的核心洞见
4. **数据增强是有效的正则化**：准确率 99.21%，错误样本从 96 降至 79
5. **9 次多种子实验标准差仅 0.07%**：结论高度可靠

## 项目结构

```
Week4-Lenet5-MINIST/
├── README.md              # 项目总览
├── data_source.md          # 数据集来源说明
├── error_analysis.md       # 错误分析报告
├── report.md               # 本文档（报告入口）
├── run_log.txt             # 运行日志
├── notes/
│   ├── paper_notes.md      # 论文阅读笔记
│   └── teaching_guide.md   # CNN 教学指南
├── src/                    # 源代码
├── experiments/            # 基线实验输出
└── report/                 # 完整实验报告
    ├── experiment_report.md
    ├── experiment_report2.0.pdf
    └── experiment_report.html
```
