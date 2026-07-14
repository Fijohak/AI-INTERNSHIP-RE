# 聚类分析 — 实验报告

## 1. 问题描述

在无标签条件下，对 Wine 数据集的 178 个样本进行聚类，评估无监督算法能否恢复真实的三个葡萄品种分类。使用 KMeans 和 Agglomerative Clustering（层次聚类）两种方法，结合 PCA 降维可视化。

## 2. 数据预处理

- `StandardScaler` 标准化（聚类算法对尺度敏感）
- PCA 降维到 2 维用于可视化（累计解释方差 > 55%）

## 3. 最佳 K 值选择

- **Elbow Method**：inertia 在 K=3 时出现明显拐点
- **Silhouette Score**：K=3 时得分最高
- 两个方法一致指向 K=3，与真实类别数吻合

## 4. 模型对比

| 指标 | KMeans | Agglomerative |
|---|---|---|
| Adjusted Rand Index (ARI) | **0.8975** | 0.7899 |
| Silhouette Score | **0.2849** | 0.2774 |
| Homogeneity | 0.8788 | — |

> Agglomerative 使用 Ward 链接方法

## 5. 分析

- **KMeans 显著优于 Agglomerative (ARI 高 0.11)**：Wine 数据的三个品种在特征空间中形成较紧凑的球形簇，KMeans 的 Voronoi 划分更适合这种分布
- **PAC 降维后 2D 可视化清晰展示三个簇**：第一主成分（PC1）主要承载酒精和酚类特征，第二主成分（PC2）承载颜色和化学属性
- **Agglomerative 倾向于产生大小不均的簇**：在 Ward 链接下偶尔会将边界样本划入相邻簇，导致部分样本分类错误
- **聚类与真实标签高度一致**：ARI=0.8975 说明无监督聚类几乎完美地恢复了真实的品种分类

## 6. 结论

Wine 数据集的 13 个化学特征具有极强的聚类结构，KMeans (K=3) 以 ARI=0.8975 的质量几乎完美地恢复了三个真实品种。这验证了无监督学习在特征质量高的领域中即使没有标签也能发现有意义的群体结构。
