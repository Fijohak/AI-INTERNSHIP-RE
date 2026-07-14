# 信用卡违约预测 — 数据集来源

## 概述

| 属性 | 说明 |
|---|---|
| 来源 | UCI Machine Learning Repository |
| 数据集 | Default of Credit Card Clients |
| 链接 | https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| 样本数 | 30,000 条 |
| 特征数 | 23 个 (含 ID) |
| 目标 | default payment next month (0=未违约, 1=违约) |
| 违约率 | ~22% |
| 任务 | 二分类 |

## 关键特征

| 特征 | 描述 |
|---|---|
| LIMIT_BAL | 信用额度 |
| SEX | 性别 |
| EDUCATION | 教育程度 |
| MARRIAGE | 婚姻状态 |
| AGE | 年龄 |
| PAY_0 ~ PAY_6 | 近 6 个月还款状态 (-2 ~ 8) |
| BILL_AMT1 ~ BILL_AMT6 | 近 6 个月账单金额 |
| PAY_AMT1 ~ PAY_AMT6 | 近 6 个月还款金额 |

## 加载方式

```python
import pandas as pd
df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", header=1)
```
