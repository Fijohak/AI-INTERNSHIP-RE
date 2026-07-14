# Week 3 — Kaggle 表格竞赛专题

> 日期：2026-06-16 ~ 2026-06-18
> 课程：吴恩达 Generative AI for Everyone

## 项目列表

| 项目 | 任务类型 | 模型 | 最佳结果 |
|---|---|---|---|
| [Titanic 生存预测](titanic_survive/) | 二分类 | Random Forest | Val Acc: 78.21% |
| [House Prices 房价预测](house_price/house-prices-advanced-regression-techniques/) | 回归 | Random Forest / Gradient Boosting | Val RMSE (log): 0.15658 |

## 学习要点

1. **结构化表格数据处理**：缺失值填充、类别编码、特征工程
2. **模型选择**：随机森林适用于非线性特征交互，梯度提升通过逐步残差优化达到更好的泛化
3. **Kaggle 竞赛流程**：数据探索 → 预处理 → 模型训练 → 提交预测

## 技术栈

- Python 3.x
- pandas（数据处理）
- scikit-learn（RandomForest、GradientBoosting、预处理）
- numpy（数值计算）
