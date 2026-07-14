# Ames 房价预测数据预处理整理文档

> 适用任务：使用随机森林 / 梯度提升树 / XGBoost / LightGBM 等模型预测 `SalePrice`  
> 数据来源：Ames Housing 房价数据字段说明  
> 说明：本文是基于字段含义整理的预处理方案。若后续拿到 `train.csv`，还应进一步统计每列缺失比例、异常值和类别分布。

---

## 1. 总体处理原则

这个数据集的数据类型比较复杂，主要包含四类字段：

1. **数值型字段**
   - 例如面积、年份、房间数量、车库面积等。
   - 通常可以直接用于随机森林。
   - 缺失值一般用中位数、0 或特殊规则填充。

2. **名义类别型字段**
   - 类别之间没有明显大小顺序。
   - 例如 `Neighborhood`、`MSZoning`、`BldgType`。
   - 推荐使用 One-Hot Encoding。

3. **有序类别型字段**
   - 类别之间有明显等级高低。
   - 例如 `Ex > Gd > TA > Fa > Po`。
   - 推荐使用 Ordinal Encoding。

4. **ID 或弱信息字段**
   - 例如 `Id`。
   - 一般不参与训练。

---

## 2. 建议不参与训练的字段

| 字段 | 原因 | 建议 |
|---|---|---|
| `Id` | 只是样本编号，不代表房屋特征 | 删除，不参与训练 |
| `SalePrice` | 目标变量 | 作为 y，不放入 X |
| `Utilities` | 通常几乎全是 `AllPub`，信息量很低 | 可以先删除；如果想保守，也可以保留编码 |

> 注意：`Utilities` 是否删除最好结合实际数据分布判断。如果几乎所有样本都是同一类别，删除更合适。

---

## 3. 数值型字段处理

这些字段本身就是数值，可以直接作为模型输入。

### 3.1 连续数值型字段

| 字段 | 含义 | 缺失值建议 | 是否参与训练 |
|---|---|---|---|
| `LotFrontage` | 街道连接长度 | 用中位数填充；更好做法是按 `Neighborhood` 分组中位数填充 | 是 |
| `LotArea` | 地块面积 | 一般无缺失；有缺失用中位数 | 是 |
| `MasVnrArea` | 砌体贴面面积 | 无砌体时填 0 | 是 |
| `BsmtFinSF1` | 地下室一类完成面积 | 无地下室填 0 | 是 |
| `BsmtFinSF2` | 地下室二类完成面积 | 无地下室填 0 | 是 |
| `BsmtUnfSF` | 地下室未完成面积 | 无地下室填 0 | 是 |
| `TotalBsmtSF` | 地下室总面积 | 无地下室填 0 | 是 |
| `1stFlrSF` | 一楼面积 | 一般无缺失 | 是 |
| `2ndFlrSF` | 二楼面积 | 无二楼通常为 0 | 是 |
| `LowQualFinSF` | 低质量完成面积 | 无则为 0 | 是 |
| `GrLivArea` | 地上居住面积 | 一般无缺失 | 是 |
| `GarageArea` | 车库面积 | 无车库填 0 | 是 |
| `WoodDeckSF` | 木露台面积 | 无则为 0 | 是 |
| `OpenPorchSF` | 开放门廊面积 | 无则为 0 | 是 |
| `EnclosedPorch` | 封闭门廊面积 | 无则为 0 | 是 |
| `3SsnPorch` | 三季门廊面积 | 无则为 0 | 是 |
| `ScreenPorch` | 纱窗门廊面积 | 无则为 0 | 是 |
| `PoolArea` | 游泳池面积 | 无泳池填 0 | 是 |
| `MiscVal` | 其他设施价值 | 无则为 0 | 是 |

### 3.2 离散数值型字段

这些字段虽然是数字，但含义是数量或评分。

| 字段 | 含义 | 缺失值建议 | 是否参与训练 |
|---|---|---|---|
| `OverallQual` | 整体材料和完成质量评分，1-10 | 一般无缺失 | 是 |
| `OverallCond` | 整体状况评分，1-10 | 一般无缺失 | 是 |
| `BsmtFullBath` | 地下室全卫数量 | 无地下室填 0 | 是 |
| `BsmtHalfBath` | 地下室半卫数量 | 无地下室填 0 | 是 |
| `FullBath` | 地上全卫数量 | 一般无缺失 | 是 |
| `HalfBath` | 地上半卫数量 | 一般无缺失 | 是 |
| `BedroomAbvGr` / `Bedroom` | 地上卧室数量 | 一般无缺失 | 是 |
| `KitchenAbvGr` / `Kitchen` | 地上厨房数量 | 一般无缺失 | 是 |
| `TotRmsAbvGrd` | 地上总房间数 | 一般无缺失 | 是 |
| `Fireplaces` | 壁炉数量 | 无则为 0 | 是 |
| `GarageCars` | 车库可停车辆数 | 无车库填 0 | 是 |
| `MoSold` | 出售月份 | 一般无缺失；可当类别，也可当数值 | 是 |
| `YrSold` | 出售年份 | 一般无缺失 | 是 |

---

## 4. 年份字段处理

年份字段可以直接作为数值，也可以进一步构造“房龄”类特征。

| 字段 | 含义 | 缺失值建议 | 推荐处理 |
|---|---|---|---|
| `YearBuilt` | 建造年份 | 一般无缺失 | 保留；可构造 `HouseAge = YrSold - YearBuilt` |
| `YearRemodAdd` | 翻修年份 | 一般无缺失 | 保留；可构造 `RemodAge = YrSold - YearRemodAdd` |
| `GarageYrBlt` | 车库建造年份 | 无车库时缺失 | 无车库可填 0，或填 `YearBuilt`，并增加 `HasGarage` 特征 |

推荐新特征：

| 新字段 | 构造方式 | 含义 |
|---|---|---|
| `HouseAge` | `YrSold - YearBuilt` | 房屋出售时房龄 |
| `RemodAge` | `YrSold - YearRemodAdd` | 距离上次翻修的时间 |
| `GarageAge` | `YrSold - GarageYrBlt` | 车库年龄 |
| `IsRemodeled` | `YearRemodAdd != YearBuilt` | 是否翻修过 |

---

## 5. 需要 One-Hot Encoding 的字段

以下字段是**名义类别型变量**，类别之间没有严格高低顺序，推荐做 One-Hot Encoding。

| 字段 | 含义 | 缺失值建议 | 编码方式 |
|---|---|---|---|
| `MSSubClass` | 房屋类型编号 | 一般无缺失 | 转为字符串后 One-Hot |
| `MSZoning` | 土地分区 | 众数填充 | One-Hot |
| `Street` | 道路类型 | 众数填充 | One-Hot |
| `Alley` | 巷道类型 | 缺失填 `None`，表示无巷道 | One-Hot |
| `LandContour` | 地形平坦程度 | 众数填充 | One-Hot |
| `LotConfig` | 地块配置 | 众数填充 | One-Hot |
| `Neighborhood` | 所在社区 | 众数填充 | One-Hot |
| `Condition1` | 主要周边条件 | 众数填充 | One-Hot |
| `Condition2` | 第二周边条件 | 众数填充 | One-Hot |
| `BldgType` | 建筑类型 | 众数填充 | One-Hot |
| `HouseStyle` | 房屋风格 | 众数填充 | One-Hot |
| `RoofStyle` | 屋顶类型 | 众数填充 | One-Hot |
| `RoofMatl` | 屋顶材料 | 众数填充 | One-Hot |
| `Exterior1st` | 外墙主材料 | 众数填充 | One-Hot |
| `Exterior2nd` | 外墙第二材料 | 众数填充 | One-Hot |
| `MasVnrType` | 砌体贴面类型 | 缺失填 `None` | One-Hot |
| `Foundation` | 地基类型 | 众数填充 | One-Hot |
| `Heating` | 供暖类型 | 众数填充 | One-Hot |
| `CentralAir` | 是否中央空调 | 众数填充 | One-Hot 或二值编码 |
| `Electrical` | 电气系统 | 众数填充 | One-Hot |
| `GarageType` | 车库类型 | 缺失填 `None` | One-Hot |
| `MiscFeature` | 其他设施 | 缺失填 `None` | One-Hot |
| `SaleType` | 销售类型 | 众数填充 | One-Hot |
| `SaleCondition` | 销售条件 | 众数填充 | One-Hot |

### 特别提醒

`MSSubClass` 虽然看起来是数字，但它其实是房屋类型编号，例如 20、30、60 等。  
它不是连续数值，因此建议先转成字符串，再做 One-Hot Encoding。

---

## 6. 需要 Ordinal Encoding 的字段

以下字段是**有序类别型变量**，类别之间存在明显等级关系，推荐映射成数字。

### 6.1 通用质量等级字段

适用于：

- `ExterQual`
- `ExterCond`
- `HeatingQC`
- `KitchenQual`
- `FireplaceQu`
- `GarageQual`
- `GarageCond`
- `PoolQC`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Ex` | Excellent | 5 |
| `Gd` | Good | 4 |
| `TA` | Typical/Average | 3 |
| `Fa` | Fair | 2 |
| `Po` | Poor | 1 |
| `NA` / `None` | 不存在该设施 | 0 |

### 6.2 地下室质量相关字段

适用于：

- `BsmtQual`
- `BsmtCond`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Ex` | Excellent | 5 |
| `Gd` | Good | 4 |
| `TA` | Typical | 3 |
| `Fa` | Fair | 2 |
| `Po` | Poor | 1 |
| `NA` / `None` | No Basement | 0 |

### 6.3 地下室采光 / 暴露程度

适用于：

- `BsmtExposure`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Gd` | Good Exposure | 4 |
| `Av` | Average Exposure | 3 |
| `Mn` | Minimum Exposure | 2 |
| `No` | No Exposure | 1 |
| `NA` / `None` | No Basement | 0 |

### 6.4 地下室完成类型

适用于：

- `BsmtFinType1`
- `BsmtFinType2`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `GLQ` | Good Living Quarters | 6 |
| `ALQ` | Average Living Quarters | 5 |
| `BLQ` | Below Average Living Quarters | 4 |
| `Rec` | Average Rec Room | 3 |
| `LwQ` | Low Quality | 2 |
| `Unf` | Unfinished | 1 |
| `NA` / `None` | No Basement | 0 |

### 6.5 地块形状

适用于：

- `LotShape`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Reg` | Regular | 4 |
| `IR1` | Slightly irregular | 3 |
| `IR2` | Moderately irregular | 2 |
| `IR3` | Irregular | 1 |

### 6.6 土地坡度

适用于：

- `LandSlope`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Gtl` | Gentle | 3 |
| `Mod` | Moderate | 2 |
| `Sev` | Severe | 1 |

### 6.7 功能性字段

适用于：

- `Functional`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Typ` | Typical | 7 |
| `Min1` | Minor Deductions 1 | 6 |
| `Min2` | Minor Deductions 2 | 5 |
| `Mod` | Moderate Deductions | 4 |
| `Maj1` | Major Deductions 1 | 3 |
| `Maj2` | Major Deductions 2 | 2 |
| `Sev` | Severely Damaged | 1 |
| `Sal` | Salvage only | 0 |

### 6.8 车库完成情况

适用于：

- `GarageFinish`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Fin` | Finished | 3 |
| `RFn` | Rough Finished | 2 |
| `Unf` | Unfinished | 1 |
| `NA` / `None` | No Garage | 0 |

### 6.9 铺装车道

适用于：

- `PavedDrive`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `Y` | Paved | 2 |
| `P` | Partial Pavement | 1 |
| `N` | Dirt/Gravel | 0 |

### 6.10 围栏质量

适用于：

- `Fence`

推荐映射：

| 原始值 | 含义 | 编码 |
|---|---|---|
| `GdPrv` | Good Privacy | 4 |
| `MnPrv` | Minimum Privacy | 3 |
| `GdWo` | Good Wood | 2 |
| `MnWw` | Minimum Wood/Wire | 1 |
| `NA` / `None` | No Fence | 0 |

---

## 7. 特殊缺失值处理

这个数据集里很多 `NA` 并不是“数据缺失”，而是表示“没有该设施”。这类字段不能简单删除。

### 7.1 缺失表示“没有该设施”的字段

| 字段 | `NA` 的真实含义 | 推荐填充值 |
|---|---|---|
| `Alley` | 无巷道 | `None` |
| `BsmtQual` | 无地下室 | `None` |
| `BsmtCond` | 无地下室 | `None` |
| `BsmtExposure` | 无地下室 | `None` |
| `BsmtFinType1` | 无地下室 | `None` |
| `BsmtFinType2` | 无地下室 | `None` |
| `FireplaceQu` | 无壁炉 | `None` |
| `GarageType` | 无车库 | `None` |
| `GarageFinish` | 无车库 | `None` |
| `GarageQual` | 无车库 | `None` |
| `GarageCond` | 无车库 | `None` |
| `PoolQC` | 无泳池 | `None` |
| `Fence` | 无围栏 | `None` |
| `MiscFeature` | 无其他设施 | `None` |
| `MasVnrType` | 无砌体贴面 | `None` |

### 7.2 对应数值字段填 0

如果某类设施不存在，对应的面积、数量、价值字段一般应填 0。

| 类别 | 字段 | 推荐填充值 |
|---|---|---|
| 地下室 | `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath` | 0 |
| 车库 | `GarageCars`, `GarageArea` | 0 |
| 砌体贴面 | `MasVnrArea` | 0 |
| 泳池 | `PoolArea` | 0 |
| 其他设施 | `MiscVal` | 0 |
| 壁炉 | `Fireplaces` | 0 |

---

## 8. 可以新增的特征

随机森林不强制要求特征工程，但适当新增特征通常有帮助。

| 新字段 | 构造方法 | 说明 |
|---|---|---|
| `TotalSF` | `TotalBsmtSF + 1stFlrSF + 2ndFlrSF` | 总面积 |
| `TotalBath` | `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath` | 总浴室数量 |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` | 门廊总面积 |
| `HasGarage` | `GarageArea > 0` | 是否有车库 |
| `HasBsmt` | `TotalBsmtSF > 0` | 是否有地下室 |
| `HasPool` | `PoolArea > 0` | 是否有泳池 |
| `HasFireplace` | `Fireplaces > 0` | 是否有壁炉 |
| `HasMasVnr` | `MasVnrArea > 0` | 是否有砌体贴面 |
| `HouseAge` | `YrSold - YearBuilt` | 房龄 |
| `RemodAge` | `YrSold - YearRemodAdd` | 翻修距今时间 |
| `IsRemodeled` | `YearRemodAdd != YearBuilt` | 是否翻修过 |

---

## 9. 推荐最终处理清单

### 9.1 删除字段

```text
Id
```

可选删除：

```text
Utilities
```

---

### 9.2 转换为字符串再编码的字段

```text
MSSubClass
```

原因：它虽然是数字，但本质是房屋类型编号。

---

### 9.3 One-Hot Encoding 字段

```text
MSSubClass
MSZoning
Street
Alley
LandContour
LotConfig
Neighborhood
Condition1
Condition2
BldgType
HouseStyle
RoofStyle
RoofMatl
Exterior1st
Exterior2nd
MasVnrType
Foundation
Heating
CentralAir
Electrical
GarageType
MiscFeature
SaleType
SaleCondition
```

---

### 9.4 Ordinal Encoding 字段

```text
LotShape
LandSlope
ExterQual
ExterCond
BsmtQual
BsmtCond
BsmtExposure
BsmtFinType1
BsmtFinType2
HeatingQC
KitchenQual
Functional
FireplaceQu
GarageFinish
GarageQual
GarageCond
PavedDrive
PoolQC
Fence
```

---

### 9.5 数值字段

```text
LotFrontage
LotArea
OverallQual
OverallCond
YearBuilt
YearRemodAdd
MasVnrArea
BsmtFinSF1
BsmtFinSF2
BsmtUnfSF
TotalBsmtSF
1stFlrSF
2ndFlrSF
LowQualFinSF
GrLivArea
BsmtFullBath
BsmtHalfBath
FullBath
HalfBath
BedroomAbvGr / Bedroom
KitchenAbvGr / Kitchen
TotRmsAbvGrd
Fireplaces
GarageYrBlt
GarageCars
GarageArea
WoodDeckSF
OpenPorchSF
EnclosedPorch
3SsnPorch
ScreenPorch
PoolArea
MiscVal
MoSold
YrSold
```

---

## 10. 推荐代码框架

下面是适合随机森林的预处理思路。

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("train.csv")

# 目标变量
y = np.log1p(df["SalePrice"])

# 删除不参与训练字段
X = df.drop(columns=["SalePrice", "Id"])

# MSSubClass 是类别编号，不是连续数值
X["MSSubClass"] = X["MSSubClass"].astype(str)

# 可选：删除信息量很低的字段
if "Utilities" in X.columns:
    X = X.drop(columns=["Utilities"])

# 缺失代表“没有该设施”的类别字段
none_cols = [
    "Alley",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
    "MasVnrType"
]

for col in none_cols:
    if col in X.columns:
        X[col] = X[col].fillna("None")

# 对应数值字段填 0
zero_cols = [
    "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
    "PoolArea", "MiscVal"
]

for col in zero_cols:
    if col in X.columns:
        X[col] = X[col].fillna(0)

# GarageYrBlt 特殊处理
if "GarageYrBlt" in X.columns:
    X["GarageYrBlt"] = X["GarageYrBlt"].fillna(0)

# 新增特征
if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(X.columns):
    X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]

if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(X.columns):
    X["TotalBath"] = (
        X["FullBath"] + 
        0.5 * X["HalfBath"] + 
        X["BsmtFullBath"] + 
        0.5 * X["BsmtHalfBath"]
    )

if {"OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"}.issubset(X.columns):
    X["TotalPorchSF"] = (
        X["OpenPorchSF"] + 
        X["EnclosedPorch"] + 
        X["3SsnPorch"] + 
        X["ScreenPorch"]
    )

if {"YrSold", "YearBuilt"}.issubset(X.columns):
    X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

if {"YrSold", "YearRemodAdd"}.issubset(X.columns):
    X["RemodAge"] = X["YrSold"] - X["YearRemodAdd"]

if {"YearBuilt", "YearRemodAdd"}.issubset(X.columns):
    X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)

# Ordinal Encoding 字段
ordinal_cols = [
    "LotShape",
    "LandSlope",
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence"
]

ordinal_cols = [col for col in ordinal_cols if col in X.columns]

# One-Hot 字段
onehot_cols = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LandContour",
    "LotConfig",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "Electrical",
    "GarageType",
    "MiscFeature",
    "SaleType",
    "SaleCondition"
]

onehot_cols = [col for col in onehot_cols if col in X.columns]

# 数值字段自动识别
numeric_cols = [
    col for col in X.select_dtypes(include=["int64", "float64"]).columns
    if col not in ordinal_cols
]

# 不在上面列表里的 object 字段，保守起见也加入 one-hot
extra_object_cols = [
    col for col in X.select_dtypes(include=["object"]).columns
    if col not in ordinal_cols and col not in onehot_cols
]
onehot_cols += extra_object_cols

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("ord", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), ordinal_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), onehot_cols)
    ]
)

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
    max_features="sqrt"
)

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)

pred = pipe.predict(X_valid)
rmse = mean_squared_error(y_valid, pred, squared=False)

print("Validation RMSE:", rmse)
```

---

## 11. 更稳妥的建议

如果你只是先跑通随机森林，最简单稳妥的方式是：

1. 删除 `Id`
2. `SalePrice` 作为目标变量
3. `MSSubClass` 转字符串
4. 特殊 `NA` 字段填 `None`
5. 数值缺失填中位数或 0
6. 类别字段 One-Hot
7. 有序等级字段 Ordinal Encoding
8. 随机森林训练
9. 用 log 后的 `SalePrice` 评估 RMSE

---

## 12. 新手容易踩的坑

### 坑 1：把 `MSSubClass` 当普通数字

错误理解：

```python
MSSubClass = 20, 30, 40, 60
```

这不是“数值越大房子越高级”，它只是类别编号。

正确做法：

```python
X["MSSubClass"] = X["MSSubClass"].astype(str)
```

---

### 坑 2：直接删除所有缺失行

这个数据集里很多缺失表示“没有该设施”，例如无车库、无地下室、无泳池。

错误做法：

```python
df.dropna()
```

正确做法：

```python
X["GarageType"] = X["GarageType"].fillna("None")
X["GarageArea"] = X["GarageArea"].fillna(0)
```

---

### 坑 3：把所有类别都 One-Hot

虽然可以，但有序类别会丢失等级信息。

例如：

```text
Ex > Gd > TA > Fa > Po
```

这类字段更适合 Ordinal Encoding。

---

### 坑 4：忘记对测试集做同样处理

推荐使用 `Pipeline`，这样训练集、验证集、测试集会自动走同一套预处理流程。

---

# 总结

对于这个房价数据集，推荐的核心策略是：

```text
删除无意义字段：Id
目标变量：SalePrice
数值字段：填 0 或中位数
无设施类 NA：填 None
名义类别字段：One-Hot Encoding
有序等级字段：Ordinal Encoding
特殊编号字段：MSSubClass 转字符串
可选特征工程：房龄、总面积、总浴室数、是否有车库等
```

这样处理后，随机森林就可以比较稳定地训练了。

---

# 随机森林训练记录

## 分析流程

1. **目的**：预测 Ames 地区房屋的 SalePrice（销售价格）
2. **数据预处理**：按上文预处理方案，完成缺失值填充、Ordinal Encoding、One-Hot Encoding、特征工程、StandardScaler 标准化，最终得到 250+ 维特征矩阵
3. **模型选择**：该问题为结构化表格数据的回归任务。选用 RandomForestRegressor 是因为：
   - 房价受多个特征（面积、年份、社区、质量等级）共同影响，这些特征之间存在复杂的非线性关系，随机森林能自然地捕获
   - 数据中包含大量类别特征（One-Hot 后维度高），随机森林的特征随机采样（`max_features="sqrt"`）能有效应对高维稀疏特征
   - 集成学习降低了对单一异常样本的依赖，减少了过拟合风险
4. **训练配置**：
   - `n_estimators=500`，`max_depth=20`，`min_samples_leaf=2`，`max_features="sqrt"`
   - 目标变量做 `log1p` 变换，使分布接近正态
   - 训练/验证集按 8:2 划分（`random_state=42`）
5. **验证集 RMSE (log)**：0.15658
6. **验证集 RMSE ($)**：$34,689.01
