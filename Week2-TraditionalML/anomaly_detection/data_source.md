# 异常行为识别 — 数据集来源

## 概述

| 属性 | 说明 |
|---|---|
| 来源 | sklearn.datasets.make_blobs() 合成数据 |
| 数据生成 | 3 个高斯簇 (正常) + 随机散点 (异常) |
| 正常样本 | 1,000 个 |
| 异常样本 | 50 个 (5%) |
| 特征 | 2 维 (便于可视化) |
| 任务 | 异常检测 |

## 设计理由

异常检测的核心挑战是：异常样本极少且样本间差异大。使用合成数据可以精确控制异常比例和分布，便于直观对比 Isolation Forest 和 One-Class SVM 的决策边界。

## 生成方式

```python
from sklearn.datasets import make_blobs
X_normal, _ = make_blobs(n_samples=1000, centers=3, cluster_std=0.8)
X_outliers = rng.uniform(low=-6, high=6, size=(50, 2))
```
