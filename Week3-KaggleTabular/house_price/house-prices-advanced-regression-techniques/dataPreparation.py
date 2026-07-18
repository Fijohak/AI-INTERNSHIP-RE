"""
Ames House Price Prediction — 数据清洗 & 预处理
=================================================
基于 house_price_preprocessing_plan.md 的规范，
完成缺失值填充、特征工程、编码、标准化，
输出清洗后的 train/test 数组，供后续建模直接使用。

输出:
  X_train.npy         标准化后的训练集特征 (float32)
  X_test.npy          标准化后的测试集特征 (float32)
  y_train.npy         对数变换后的目标值 (float32)
  feature_names.txt   最终特征名列表
  scaler.pkl          StandardScaler 对象
"""

import numpy as np
import pandas as pd
import pickle
import os

# ====== 路径配置 =====================================================
BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ====== 1. 数据加载 ==================================================
train = pd.read_csv(f"{BASE}/train.csv")
test  = pd.read_csv(f"{BASE}/test.csv")

print(f"train  shape: {train.shape}")
print(f"test   shape: {test.shape}")

# ====== 2. 目标变量 (对数变换) =======================================
y = np.log1p(train["SalePrice"]).values.astype(np.float32)
print(f"SalePrice 对数变换后: mean={y.mean():.3f}, std={y.std():.3f}")

# ====== 3. 合并 train/test 统一处理 ==================================
all_data = pd.concat(
    [train.drop(columns=["SalePrice"]), test], axis=0, ignore_index=True
)
n_train = train.shape[0]
print(f"all_data shape: {all_data.shape}")

# ====== 4. 删除不参与训练的字段 ======================================
drop_cols = ["Id"]
# Utilities 几乎全是 AllPub，信息量极低
if "Utilities" in all_data.columns:
    unique_vals = all_data["Utilities"].nunique()
    print(f"  Utilities 唯一值数: {unique_vals}  →  删除")
    drop_cols.append("Utilities")

all_data.drop(columns=drop_cols, inplace=True, errors="ignore")
print(f"删除列: {drop_cols}")

# ====== 5. MSSubClass 类别编号 → 字符串 ==============================
all_data["MSSubClass"] = all_data["MSSubClass"].astype(str)

# ====== 6. 缺失值处理 =================================================
print("\n--- 缺失值处理 ---")

# 记录处理前的缺失情况
missing_before = all_data.isnull().sum()
missing_before = missing_before[missing_before > 0]
print(f"处理前有缺失的列数: {len(missing_before)}")

# 6a. "NA" 表示"没有该设施" → 填 "None"
none_fill_cols = [
    "Alley",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
    "MasVnrType",
]
for col in none_fill_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna("None")

# 6b. 对应数值字段 → 填 0 (没有该设施 = 面积/数量为 0)
zero_fill_cols = [
    "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
    "PoolArea", "MiscVal",
    "Fireplaces",
]
for col in zero_fill_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

# 6c. LotFrontage 按 Neighborhood 分组中位数填充
if "LotFrontage" in all_data.columns and "Neighborhood" in all_data.columns:
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )
    # 兜底：如果某个社区全部缺失，用全局中位数
    all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].median())

# 6d. GarageYrBlt 无车库时填 0
if "GarageYrBlt" in all_data.columns:
    all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(0)

# 6e. 剩余数值缺失 → 中位数
numeric_cols = all_data.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna(all_data[col].median())

# 6f. 剩余类别缺失 → 众数
obj_cols = all_data.select_dtypes(include=["object"]).columns
for col in obj_cols:
    if all_data[col].isnull().any():
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# 最终检查
remaining_missing = all_data.isnull().sum().sum()
assert remaining_missing == 0, f"仍有 {remaining_missing} 个缺失值！"
print(f"处理后缺失值: 0  [OK]")

# ====== 7. 特征工程 ===================================================
print("\n--- 特征工程 ---")
new_feats = []

if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(all_data.columns):
    all_data["TotalSF"] = all_data["TotalBsmtSF"] + all_data["1stFlrSF"] + all_data["2ndFlrSF"]
    new_feats.append("TotalSF")

if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(all_data.columns):
    all_data["TotalBath"] = (
        all_data["FullBath"] + 0.5 * all_data["HalfBath"]
        + all_data["BsmtFullBath"] + 0.5 * all_data["BsmtHalfBath"]
    )
    new_feats.append("TotalBath")

if {"OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"}.issubset(all_data.columns):
    all_data["TotalPorchSF"] = (
        all_data["OpenPorchSF"] + all_data["EnclosedPorch"]
        + all_data["3SsnPorch"] + all_data["ScreenPorch"]
    )
    new_feats.append("TotalPorchSF")

if {"YrSold", "YearBuilt"}.issubset(all_data.columns):
    all_data["HouseAge"] = all_data["YrSold"] - all_data["YearBuilt"]
    new_feats.append("HouseAge")

if {"YrSold", "YearRemodAdd"}.issubset(all_data.columns):
    all_data["RemodAge"] = all_data["YrSold"] - all_data["YearRemodAdd"]
    new_feats.append("RemodAge")

if {"YearBuilt", "YearRemodAdd"}.issubset(all_data.columns):
    all_data["IsRemodeled"] = (all_data["YearRemodAdd"] != all_data["YearBuilt"]).astype(int)
    new_feats.append("IsRemodeled")

# 是否有某设施
for src_col, new_col in [
    ("GarageArea", "HasGarage"),
    ("TotalBsmtSF", "HasBsmt"),
    ("PoolArea", "HasPool"),
    ("Fireplaces", "HasFireplace"),
    ("MasVnrArea", "HasMasVnr"),
    ("WoodDeckSF", "HasWoodDeck"),
]:
    if src_col in all_data.columns:
        all_data[new_col] = (all_data[src_col] > 0).astype(int)
        new_feats.append(new_col)

if {"YrSold", "GarageYrBlt"}.issubset(all_data.columns):
    all_data["GarageAge"] = all_data["YrSold"] - all_data["GarageYrBlt"]
    all_data.loc[all_data["GarageYrBlt"] == 0, "GarageAge"] = 0
    new_feats.append("GarageAge")

print(f"新增特征 ({len(new_feats)}): {new_feats}")

# ====== 8. 编码 =======================================================
print("\n--- 编码 ---")

# 8a. Ordinal Encoding
ORDINAL_MAPPINGS = {
    "ExterQual":    {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "ExterCond":    {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "HeatingQC":    {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "KitchenQual":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "FireplaceQu":  {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "GarageQual":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "GarageCond":   {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "PoolQC":       {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "BsmtQual":     {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "BsmtCond":     {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
    "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
    "BsmtFinType1": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "BsmtFinType2": {"GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0},
    "LotShape":     {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
    "LandSlope":    {"Gtl": 3, "Mod": 2, "Sev": 1},
    "Functional":   {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
    "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
    "PavedDrive":   {"Y": 2, "P": 1, "N": 0},
    "Fence":        {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0},
}

ordinal_done = []
for col, mapping in ORDINAL_MAPPINGS.items():
    if col in all_data.columns:
        all_data[col] = all_data[col].map(mapping).fillna(0).astype(int)
        ordinal_done.append(col)
print(f"Ordinal Encoding: {len(ordinal_done)} 列")

# 8b. One-Hot Encoding
onehot_cols = [
    "MSSubClass", "MSZoning", "Street", "Alley",
    "LandContour", "LotConfig", "Neighborhood",
    "Condition1", "Condition2", "BldgType", "HouseStyle",
    "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
    "MasVnrType", "Foundation", "Heating", "CentralAir",
    "Electrical", "GarageType", "MiscFeature",
    "SaleType", "SaleCondition",
]
onehot_cols = [c for c in onehot_cols if c in all_data.columns]

# 补充遗漏的 object 列 (防止有漏网之鱼)
remaining_obj = [
    c for c in all_data.select_dtypes(include=["object"]).columns
    if c not in onehot_cols and c not in ordinal_done
]
if remaining_obj:
    print(f"  补充 One-Hot 列: {remaining_obj}")
    onehot_cols += remaining_obj

all_data = pd.get_dummies(all_data, columns=onehot_cols, drop_first=False)
print(f"One-Hot Encoding: {len(onehot_cols)} 列 → {all_data.shape[1]} 总特征")

# ====== 9. 拆分回 train / test ========================================
print("\n--- 拆分 & 标准化 ---")

feature_names = all_data.columns.tolist()
X = all_data.iloc[:n_train, :].values.astype(np.float32)
X_test = all_data.iloc[n_train:, :].values.astype(np.float32)

print(f"X       shape: {X.shape}")
print(f"X_test  shape: {X_test.shape}")

# ====== 10. 标准化 =====================================================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

print(f"标准化后: X mean={X.mean():.4f}, std={X.std():.4f}")

# ====== 11. 保存 =======================================================
print("\n--- 保存处理后的数据 ---")

np.save(f"{PROCESSED_DIR}/X_train.npy", X)
np.save(f"{PROCESSED_DIR}/y_train.npy", y)
np.save(f"{PROCESSED_DIR}/X_test.npy", X_test)

with open(f"{PROCESSED_DIR}/feature_names.txt", "w", encoding="utf-8") as f:
    for name in feature_names:
        f.write(name + "\n")

with open(f"{PROCESSED_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 同时保存测试集 ID（提交时用）
test_ids = test["Id"].values
np.save(f"{PROCESSED_DIR}/test_ids.npy", test_ids)

print(f"已保存至 {PROCESSED_DIR}/")
print(f"  X_train.npy        {X.shape}")
print(f"  y_train.npy        {y.shape}")
print(f"  X_test.npy         {X_test.shape}")
print(f"  test_ids.npy       {test_ids.shape}")
print(f"  feature_names.txt  {len(feature_names)} 个特征名")
print(f"  scaler.pkl         StandardScaler")

# ====== 12. 数据概览 ===================================================
print("\n" + "=" * 55)
print("  数据清洗完成!")
print("=" * 55)
print(f"  训练样本:       {n_train}")
print(f"  测试样本:       {X_test.shape[0]}")
print(f"  最终特征数:     {X.shape[1]}")
print(f"  目标 y (log):   mean={y.mean():.3f}, std={y.std():.3f}")
print(f"  特征 X:         min={X.min():.2f}, max={X.max():.2f}")
print(f"  缺失值:         0")
print("=" * 55)
