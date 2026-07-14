# Week 1 — AI 认知基础

> 日期：2026-07-15
> 课程：吴恩达 AI for Everyone, AI Prompting for Everyone

## 项目列表

| 项目 | 任务类型 | 模型 | 最佳结果 |
|---|---|---|---|
| [Iris 分类](iris_classification/) | 多分类 | KNN / Logistic Regression / SVM | SVM: 96.67% |
| [California Housing 线性回归](linear_regression/) | 回归 | Linear Regression | R2=0.5758, RMSE=$74,560 |

## 学习要点

1. **Iris 分类**：经典 ML 入门任务，对比了 KNN、Logistic Regression、SVM 三种分类器的表现差异
2. **线性回归**：使用 California Housing 数据集，完整走通数据探索 → 标准化 → 训练 → 残差分析流程

## 技术栈

- Python 3.13
- scikit-learn（模型、预处理、评估）
- matplotlib（可视化）
- numpy / pandas
