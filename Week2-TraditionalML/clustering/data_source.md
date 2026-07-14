# 聚类分析 — 数据集来源

## 概述

| 属性 | 说明 |
|---|---|
| 来源 | sklearn.datasets.load_wine() |
| 原始数据 | UCI Wine 数据集 |
| 样本数 | 178 条（意大利三个品种的葡萄酒化学分析） |
| 特征数 | 13 个连续特征 |
| 类别数 | 3 类（真实标签仅用于评估，聚类过程不使用） |
| 任务 | 无监督聚类 |

## 特征

| 特征 | 描述 |
|---|---|
| alcohol | 酒精含量 |
| malic_acid | 苹果酸 |
| ash | 灰分 |
| alcalinity_of_ash | 灰分碱度 |
| magnesium | 镁含量 |
| total_phenols | 总酚 |
| flavanoids | 类黄酮 |
| nonflavanoid_phenols | 非类黄酮酚 |
| proanthocyanins | 原花青素 |
| color_intensity | 颜色强度 |
| hue | 色调 |
| od280/od315_of_diluted_wines | 稀释葡萄酒 OD280/OD315 |
| proline | 脯氨酸 |

## 加载方式

```python
from sklearn.datasets import load_wine
wine = load_wine()
X, y = wine.data, wine.target
```

sklearn 内置数据集，无需下载。注意：`y`（真实类别）仅用于外部评估（ARI, Homogeneity），聚类过程完全无监督。
