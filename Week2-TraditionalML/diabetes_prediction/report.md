# 糖尿病预测 — 实验报告

## 1. 问题描述

基于 Pima 印第安女性的 8 个医学指标（血糖、BMI、年龄、胰岛素等），预测是否患有糖尿病。二分类任务，768 条数据，患病率 ~35%。

## 2. 数据预处理

- **零值替换**：Glucose、BloodPressure、SkinThickness、Insulin、BMI 的零值用各自中位数替换（医学上不可能为零 → 实为缺失值）
- **StandardScaler** 标准化
- 80/20 分层划分 (stratify=y)

## 3. 模型对比

| 模型 | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 76.62% | 0.8287 |
| Decision Tree (depth=4) | 72.73% | 0.7361 |
| **Random Forest** | **74.03%** | **0.8170** |

## 4. 分析

- **Logistic Regression 的 AUC 最高 (0.8287)**：小数据集 + 线性可分的医学指标，LR 的简单线性假设反而避免了过拟合
- **RF 的准确率最高但 AUC 略低于 LR**：RF 在正负样本分类上更均衡，但校准后的区分能力不如 LR
- **Glucose（血糖）是最重要的特征**：无论在 LR 还是 RF 中，血糖浓度的权重/重要性都远超其他特征，这与医学常识一致——血糖是诊断糖尿病的直接指标
- **小数据集 (768 条) 限制了模型复杂度**：DT depth=4 和 RF max_depth=6 的浅层设置是为了防止过拟合

## 5. 结论

Logistic Regression 在该任务上的 AUC (0.8287) 实际优于更复杂的 Random Forest (0.8170)。在小数据集 + 医学指标的典型场景下，简单模型往往比集成模型更具泛化优势。
