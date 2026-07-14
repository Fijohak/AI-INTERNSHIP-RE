# Iris 数据集来源

## 概述

| 属性 | 说明 |
|---|---|
| 来源 | sklearn.datasets.load_iris() |
| 原始数据 | R.A. Fisher, 1936, "The use of multiple measurements in taxonomic problems" |
| 样本数 | 150 条 (每类 50 条) |
| 特征数 | 4 个连续特征 |
| 类别数 | 3 类 (setosa, versicolor, virginica) |
| 任务 | 多分类 |

## 特征

| 特征 | 描述 | 单位 |
|---|---|---|
| sepal length | 花萼长度 | cm |
| sepal width | 花萼宽度 | cm |
| petal length | 花瓣长度 | cm |
| petal width | 花瓣宽度 | cm |

## 加载方式

```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
```

sklearn 内置数据集，无需下载，直接 import 即可使用。
