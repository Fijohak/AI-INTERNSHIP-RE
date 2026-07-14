# Week 2 — 传统机器学习实战专题

> 日期：2026-07-15

## 项目列表

| 类型 | 项目 | 任务 | 模型 | 最佳结果 |
|---|---|---|---|---|
| 必做 | [信用卡违约预测](credit_card_default/) | 二分类 | LR / DT / RF | RF: Acc 81.67%, AUC 0.7727 |
| 必做 | [糖尿病预测](diabetes_prediction/) | 二分类 | LR / DT / RF | RF: Acc 74.03%, AUC 0.8170 |
| 选做 | [异常行为识别](anomaly_detection/) | 异常检测 | Isolation Forest / One-Class SVM | IF: Acc 95.56% |
| 选做 | [聚类分析](clustering/) | 聚类 | KMeans / Agglomerative / PCA | KMeans ARI: 0.8975 |

## 学习要点

1. **分类任务全流程**：数据探索 → 缺失值处理 → 标准化 → 多模型训练 → ROC 评估 → 特征重要性分析
2. **异常检测**：Isolation Forest 和 One-Class SVM 的决策边界差异，Precision/Recall 权衡
3. **聚类分析**：Elbow Method 选 K → KMeans/Agglomerative → PCA 降维可视化 → ARI/Silhouette 评估

## 技术栈

- scikit-learn (LR, DT, RF, Isolation Forest, One-Class SVM, KMeans, PCA)
- matplotlib（可视化）
- numpy / pandas
