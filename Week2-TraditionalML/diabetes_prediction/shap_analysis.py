"""
SHAP 特征归因分析 — 糖尿病预测

在已训练的模型上运行 SHAP 解释
"""

import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(__file__)
FIGS = os.path.join(BASE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
print("加载 Pima Indians Diabetes 数据集...")
url = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.csv"
)
df = pd.read_csv(url, names=FEATURE_NAMES + ["Outcome"])
X = df[FEATURE_NAMES].copy()
y = df["Outcome"]

# 缺失值处理
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    X[col] = X[col].replace(0, np.nan)
    X[col] = X[col].fillna(X[col].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y,
)

# ============================================================
# 2. 训练 Random Forest
# ============================================================
print("训练 Random Forest (SHAP 解释目标)...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42,
)
rf.fit(X_train, y_train)

# ============================================================
# 3. SHAP 分析（768 条全量，数据量小无需采样）
# ============================================================
print("计算 SHAP 值...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # 正类 (Diabetes)
else:
    shap_vals = shap_values

# ============================================================
# 4. SHAP 可视化
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
shap.summary_plot(shap_vals, X_test, feature_names=FEATURE_NAMES, show=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_summary.png")

fig, ax = plt.subplots(figsize=(7, 5))
shap.summary_plot(shap_vals, X_test, feature_names=FEATURE_NAMES,
                  plot_type="bar", show=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_bar.png")

# Waterfall: 第一个糖尿病预测样本
diabetes_idx = np.where(y_test == 1)[0]
if len(diabetes_idx) > 0:
    fig, ax = plt.subplots(figsize=(9, 5))
    # 取第 i 个样本在第 1 类（正类）上的 SHAP 值
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[diabetes_idx[0], :, 1] if shap_vals.ndim == 3 else shap_vals[diabetes_idx[0]],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and not np.isscalar(explainer.expected_value) else (explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value),
            data=X_test.iloc[diabetes_idx[0]].values,
            feature_names=FEATURE_NAMES,
        ),
        show=False,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_waterfall.png")

# Dependence: Glucose 和 BMI (fix for multi-class SHAP)
shap_vals_2d = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
shap.dependence_plot("Glucose", shap_vals_2d, X_test,
                     feature_names=FEATURE_NAMES, show=False, ax=axes[0])
shap.dependence_plot("BMI", shap_vals_2d, X_test,
                     feature_names=FEATURE_NAMES, show=False, ax=axes[1])
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "shap_dependence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_dependence.png")

print("\nSHAP 分析完成！")
