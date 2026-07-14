# Titanic 生存预测 — 数据集来源

> 竞赛：[Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

## 数据概述

| 属性 | 说明 |
|---|---|
| 来源 | Kaggle Titanic 竞赛 |
| 训练样本 | 891 条 |
| 测试样本 | 418 条 |
| 特征数 | 7 (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked) |
| 目标 | Survived (0=死亡, 1=生存) |
| 任务类型 | 二分类 |

## 特征说明

| 特征 | 类型 | 描述 |
|---|---|---|
| PassengerId | int | 乘客编号 |
| Pclass | int | 舱位等级 (1=一等, 2=二等, 3=三等) |
| Sex | string | 性别 (male/female) |
| Age | float | 年龄（含缺失值约 20%） |
| SibSp | int | 船上兄弟姐妹/配偶数 |
| Parch | int | 船上父母/子女数 |
| Fare | float | 票价（含少量缺失值） |
| Embarked | string | 登船港口 (C=瑟堡, Q=皇后镇, S=南安普顿) |

## 数据文件

| 文件 | 大小 | 说明 |
|---|---|---|
| train.csv | 60KB | 训练集 (891 条, 含 label) |
| test.csv | 28KB | 测试集 (418 条, 无 label) |
| gender_submission.csv | 3KB | Kaggle 提供的基准提交样例 |
