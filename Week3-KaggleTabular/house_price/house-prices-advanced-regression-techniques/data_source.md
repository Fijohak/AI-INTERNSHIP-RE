# House Prices 房价预测 — 数据集来源

> 竞赛：[Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## 数据概述

| 属性 | 说明 |
|---|---|
| 来源 | Kaggle House Prices 竞赛 |
| 训练样本 | 1,460 条 |
| 测试样本 | 1,459 条 |
| 原始特征数 | 79 个 (含 Id) |
| 预处理后特征数 | 250+ 维 (One-Hot 编码后) |
| 目标 | SalePrice（房价，美元） |
| 任务类型 | 回归 |

## 特征分类

| 类别 | 数量 | 示例 |
|---|---|---|
| 数值连续 | ~36 | LotArea, GrLivArea, YearBuilt |
| 有序类别 | ~20 | OverallQual (1-10), ExterQual (Ex/Gd/TA/Fa/Po) |
| 名义类别 | ~23 | MSZoning, Neighborhood, SaleType |

## 目标变量转换

原始 SalePrice 呈右偏分布，取对数变换 `log1p` 后近似正态分布：

```
y = log(1 + SalePrice)
```

预测后通过 `expm1` 反变换回美元。

## 数据文件

| 文件 | 大小 | 说明 |
|---|---|---|
| train.csv | 450KB | 训练集 (1,460 条 + SalePrice) |
| test.csv | 440KB | 测试集 (1,459 条, 无 label) |
| sample_submission.csv | 31KB | Kaggle 提交模板 |
| data_description.txt | 13KB | 79 个特征的完整说明 |

## 预处理产物 (processed/)

| 文件 | 说明 |
|---|---|
| X_train.npy | 训练集特征矩阵 |
| y_train.npy | 训练集目标 (log1p 空间) |
| X_test.npy | 测试集特征矩阵 |
| feature_names.txt | 250+ 维特征名列表 |
| scaler.pkl | StandardScaler 模型 |
| test_ids.npy | 测试集 ID |
