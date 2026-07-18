# House Prices 房价预测 — 实验报告

> 竞赛：Kaggle House Prices - Advanced Regression Techniques
> 日期：2026-06-17 ~ 2026-06-18

## 1. 问题描述

基于 Ames 房屋数据集的 79 个特征（面积、质量等级、建造年份、社区等），预测房屋销售价格。这是一个回归任务，评估指标为 RMSE（均方根误差，对数空间）。

## 2. 数据预处理

> 详细预处理方案见 [house_price_preprocessing_plan.md](house_price_preprocessing_plan.md)

### 2.1 目标变换

`SalePrice` 呈右偏分布 → `y = log1p(SalePrice)` 变换为正态分布，预测后 `expm1` 反变换。

### 2.2 缺失值填充

| 类型 | 策略 | 示例 |
|---|---|---|
| 设施缺失（无地下室/无车库） | 填 "None" | BsmtQual, GarageType |
| 面积/尺寸缺失 | 填 0 | BsmtFinSF1, TotalBsmtSF |
| 临街面长度缺失 | 按 Neighborhood 分组填中位数 | LotFrontage |
| 年份特征 | 按社区填中位数 | GarageYrBlt |

### 2.3 特征工程

| 特征 | 计算方式 |
|---|---|
| TotalSF | TotalBsmtSF + 1stFlrSF + 2ndFlrSF |
| TotalBath | FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath |
| TotalPorchSF | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF |
| HasGarage | GarageArea > 0 |
| HasBsmt | TotalBsmtSF > 0 |
| HouseAge | YrSold - YearBuilt |
| RemodelAge | YrSold - YearRemodAdd |

### 2.4 编码

| 类型 | 编码方式 |
|---|---|
| 有序类别 (ExterQual 等) | 序数编码 (Ex=5, Gd=4, TA=3, Fa=2, Po=1) |
| 名义类别 (MSZoning 等) | One-Hot 编码 |

### 2.5 标准化

`StandardScaler` 对全部特征做标准化（均值 0，标准差 1）。

## 3. 模型

### 3.1 模型列表

| 模型 | 参数 | 特点 |
|---|---|---|
| **Random Forest** | n_estimators=500, max_depth=20, min_samples_leaf=2 | 集成 500 棵树，限制深度防过拟合 |
| **Gradient Boosting** | n_estimators=500, max_depth=5, lr=0.05, subsample=0.8 | 逐步优化残差，学习率保守 |

### 3.2 训练策略

- 训练/验证：80/20 划分（random_state=42）
- 验证集评估后，全量数据重新训练用于预测测试集

## 4. 实验结果

| 指标 | Random Forest | Gradient Boosting |
|---|---|---|
| Val RMSE (log) | 0.15658 | **0.13150** |
| Val RMSE ($) | $34,689 | **$25,938** |
| Val MAE ($) | $17,846 | **$15,078** |
| Val R² (log) | 0.8686 | **0.9073** |
| 提交文件 | submission_rf.csv | submission_gbr.csv |

## 5. 特征重要性 (Random Forest Top-10)

根据 RF 的特征重要性排序（仅列出代表性特征）：

| 排名 | 特征 | 重要性 | 含义 |
|---|---|---|---|
| 1 | OverallQual | 最高 | 整体材料和装修质量 |
| 2 | GrLivArea | 高 | 地上居住面积 |
| 3 | TotalSF | 高 | 总面积（工程特征） |
| 4 | Neighborhood | 高 | 社区位置 |
| 5 | GarageCars | 中高 | 车库容量 |

## 6. 分析与改进方向

1. **当前预处理比较完善**：缺失值处理、特征工程、编码方案都有系统设计
2. **未做超参数搜索**：GridSearchCV 可进一步优化 max_depth、n_estimators
3. **可引入正则化线性模型**：Ridge/Lasso/ElasticNet 作为基线对比
4. **Stacking 集成**：将 RF 和 GBR 的预测结果作为特征，训练一个元模型
5. **Kaggle Score 对比**：两个模型分别提交可直观比较泛化能力
