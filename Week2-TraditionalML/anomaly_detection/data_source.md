# 异常行为识别 — 数据集来源

## 概述

| 属性 | 说明 |
|---|---|
| 来源 | sklearn.datasets.load_wine() |
| 原始数据 | UCI Wine 数据集（意大利三个品种葡萄酒化学分析） |
| 正常样本 | 130 条（class_0 + class_1） |
| 异常样本 | 48 条（class_2） |
| 异常比例 | 26.9% |
| 特征 | 13 个连续化学特征 |
| 任务 | 半监督异常检测 |

## 异常检测场景设计

采用**半监督异常检测范式**：
- **训练集**：仅使用 class_0 和 class_1（视为"正常"），模型从未见过异常样本
- **测试集**：class_0/1 的正常样本 + class_2 的"异常"样本混合

这是一个标准的异常检测评估设定——训练时模型只学习"正常是什么"，测试时检验能否识别不同于正常的样本。

## 加载方式

```python
from sklearn.datasets import load_wine
wine = load_wine()
X, y = wine.data, wine.target
# class_0, class_1 → 正常; class_2 → 异常
```
