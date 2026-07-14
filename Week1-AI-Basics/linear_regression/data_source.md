# California Housing 数据集来源

## 概述

| 属性 | 说明 |
|---|---|
| 来源 | sklearn.datasets.fetch_california_housing() |
| 原始数据 | 1990 年美国人口普查数据 |
| 样本数 | 20,640 条 |
| 特征数 | 8 个连续特征 |
| 目标 | 房价中位数 (单位: $100,000) |
| 任务 | 回归 |

## 特征

| 特征 | 描述 |
|---|---|
| MedInc | 街区收入中位数 |
| HouseAge | 房屋年龄中位数 |
| AveRooms | 每户平均房间数 |
| AveBedrms | 每户平均卧室数 |
| Population | 街区人口数 |
| AveOccup | 每户平均居住人数 |
| Latitude | 纬度 |
| Longitude | 经度 |

## 加载方式

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target
```

sklearn 内置数据集，首次调用自动下载，后续从缓存加载。
