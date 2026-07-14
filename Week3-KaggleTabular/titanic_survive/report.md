# Titanic 生存预测 — 实验报告

> 竞赛：Kaggle Titanic - Machine Learning from Disaster
> 日期：2026-06-16

## 1. 问题描述

基于泰坦尼克号乘客的个人信息（舱位、性别、年龄、家属数量、票价等），预测其是否在沉船事故中幸存。这是一个典型的二分类问题。

## 2. 数据预处理

### 2.1 特征选择

选用 6 个特征：`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`

### 2.2 缺失值处理

| 特征 | 缺失处理方式 |
|---|---|
| Age | 中位数填充（训练集和测试集分别使用各自的中位数） |
| Fare | 中位数填充 |

### 2.3 编码

| 特征 | 编码方式 |
|---|---|
| Sex | male → 0, female → 1 |

## 3. 模型

**Random Forest Classifier**

| 超参数 | 值 |
|---|---|
| n_estimators | 100 |
| max_depth | 5 |
| random_state | 42 |

**选型理由**：结构化表格数据、非线性特征交互、小数据集（891 条），随机森林通过集成多个决策树能较好地捕获复杂的非线性关系，同时 max_depth=5 的限制可避免过拟合。

## 4. 实验结果

| 指标 | 值 |
|---|---|
| 验证集准确率 | **78.21%** |
| 训练/验证划分 | 80/20 (stratified) |

## 5. 分析与改进方向

当前模型仅使用了 6 个基础特征，未做深入特征工程。可改进方向：

1. **特征工程**：从 Name 提取 Title (Mr/Mrs/Miss/Dr 等)，从 Cabin 提取甲板层，构建 FamilySize = SibSp + Parch + 1
2. **缺失值处理升级**：Age 按 Pclass 分组填充中位数，而非全局中位数
3. **Embarked 特征**：加入登船港口信息
4. **模型调参**：GridSearchCV 搜索最优 n_estimators 和 max_depth
5. **模型对比**：对比 Logistic Regression、SVM、XGBoost
