# LeNet-5 MNIST 手写数字识别 — 论文复现项目

> **论文**: Yann LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, Proceedings of the IEEE, 1998.
>
> **数据集**: [MNIST](http://yann.lecun.com/exdb/mnist/) — 手写数字 0–9，共 70,000 张 28×28 灰度图（60,000 训练 / 10,000 测试）
>
> **框架**: PyTorch 2.6 + torchvision

---

## 项目目标

完整复现 LeNet-5 在 MNIST 上的分类实验，跑通「读论文 → 搭模型 → 训练 → 评估 → 分析 → 写报告」全流程，目标测试准确率 **≥ 98%**。

---

## 项目结构

```
NumberRec/
├── README.md                ← 项目总览（本文档）
├── requirements.txt         ← 依赖清单
├── .gitignore
├── notes/
│   └── paper_notes.md       ← 论文阅读笔记（核心思想、结构解析）
├── src/
│   ├── model.py             ← LeNet-5 模型定义
│   ├── dataset.py           ← MNIST 数据加载 & 预处理
│   ├── train.py             ← 训练脚本
│   ├── evaluate.py          ← 测试评估 & 错误分析
│   └── visualize.py         ← 绘图工具（loss/accuracy 曲线、预测展示）
├── experiments/
│   ├── logs/                ← 训练日志（.txt / .csv）
│   └── figures/             ← 输出图片（曲线图、预测样例图）
└── report/                  ← 实验报告（最终产物）
    └── experiment_report.md
```

---

## 论文核心思想（LeCun, 1998）

| 要点 | 说明 |
|---|---|
| **卷积 + 下采样交替** | 利用局部连接 + 权值共享提取平移不变特征，降采样降低分辨率 |
| **层次化特征提取** | 底层检测边缘/角点，高层组合成全局模式 |
| **全连接分类头** | 经过展平后接两层全连接，用 RBF（原始）或 Softmax（现代变体）输出 |
| **端到端训练** | 整个网络用 BP + 梯度下降联合优化 |
| **MNIST 上效果** | 测试错误率约 0.95%（原始 LeNet-5） |

> 详见 [notes/paper_notes.md](notes/paper_notes.md)。

---

## 技术路线

| 项目 | 选择 |
|---|---|
| 编程语言 | Python |
| 深度学习框架 | PyTorch |
| 数据集 | MNIST（torchvision 内置） |
| 模型 | LeNet-5（适配 28×28 输入的简化版） |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam（收敛更快）或 SGD + Momentum |
| 训练轮数 | 5–10 epoch |
| 目标测试准确率 | ≥ 98% |
| 硬件 | CPU / CUDA (GPU) 自动检测 |

---

## 任务分解 & 执行计划

### Phase 1：读论文 & 整理笔记
- [x] 创建项目骨架
- [ ] 阅读 LeCun 1998 论文，提取 LeNet-5 核心结构
- [ ] 撰写 1–2 页笔记（模型结构、关键设计、训练细节）

### Phase 2：搭建代码
- [ ] `src/model.py` — 定义 LeNet-5（Conv → Pool → Conv → Pool → Conv → FC → FC → FC）
- [ ] `src/dataset.py` — MNIST 下载、归一化、DataLoader
- [ ] `src/train.py` — 训练循环（loss 记录、checkpoint 保存）
- [ ] `src/evaluate.py` — 测试集评估、错误样例收集
- [ ] `src/visualize.py` — loss 曲线、accuracy 曲线、预测结果可视化

### Phase 3：训练 & 调优
- [ ] 运行训练（5–10 epoch），记录 loss / accuracy
- [ ] 监控收敛情况，必要时调整学习率或 batch size
- [ ] 保存最佳模型权重到 `experiments/`

### Phase 4：评估 & 可视化
- [ ] 测试集最终准确率（目标 ≥ 98%）
- [ ] 绘制训练 loss 曲线 & 测试 accuracy 曲线
- [ ] 随机展示 10 张测试图片（预测 vs 真实标签）
- [ ] 分析分类错误的样例（可选）

### Phase 5：撰写实验报告
- [ ] 汇总模型结构、超参数、实验结果
- [ ] 分析遇到的问题与解决方式
- [ ] 对论文方法的个人理解

---

## 交付物清单

| 交付物 | 说明 |
|---|---|
| `src/` 代码 | 完整可运行的 PyTorch 训练/测试代码 |
| `notes/paper_notes.md` | 论文阅读笔记 |
| `experiments/figures/` | 训练曲线图 / 预测样例图 / 错误分析图 |
| `experiments/logs/` | 训练日志（含最终准确率） |
| `report/experiment_report.md` | 实验报告 |

---

## 参考资源

- [LeCun 1998 论文 PDF](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- [MNIST 数据集主页](http://yann.lecun.com/exdb/mnist/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/)
- [torchvision.datasets.MNIST](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.MNIST)
