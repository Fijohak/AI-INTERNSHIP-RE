# 糖尿病预测 — 数据集来源

## 概述

| 属性 | 说明 |
|---|---|
| 来源 | Pima Indians Diabetes Database |
| 原始来源 | National Institute of Diabetes and Digestive and Kidney Diseases |
| 样本数 | 768 条 (Pima 印第安女性, ≥21 岁) |
| 特征数 | 8 个 (含零值缺失) |
| 目标 | Outcome (0=健康, 1=糖尿病) |
| 患病率 | ~34.9% |
| 任务 | 二分类 |

## 特征

| 特征 | 描述 | 单位 |
|---|---|---|
| Pregnancies | 怀孕次数 | 次 |
| Glucose | 口服葡萄糖耐量试验 2 小时血糖浓度 | mg/dL |
| BloodPressure | 舒张压 | mm Hg |
| SkinThickness | 三头肌皮褶厚度 | mm |
| Insulin | 2 小时血清胰岛素 | mu U/ml |
| BMI | 身体质量指数 | kg/m² |
| DiabetesPedigreeFunction | 糖尿病家族遗传函数 | — |
| Age | 年龄 | 岁 |

## 数据质量问题

Glucose、BloodPressure、SkinThickness、Insulin、BMI 五个特征存在零值，在医学上不可能为 0，实际含义是"缺失值"。预处理时用中位数填充。

## 加载方式

```python
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
df = pd.read_csv(url, names=feature_names + ["Outcome"])
```
