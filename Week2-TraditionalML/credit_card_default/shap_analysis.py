"""
SHAP 特征归因分析 — 信用卡违约预测

在已训练的模型上运行 SHAP 解释（需要在 model.py 之后运行）
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(__file__)
FIGS = os.path.join(BASE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 1. 加载数据（与 model.py 相同流程）
# ============================================================
print("加载数据...")
df = pd.read_excel(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
    header=1,
)
df.rename(columns={"default payment next month": "default"}, inplace=True)
X = df.drop(columns=["ID", "default"])
y = df["default"]
feature_names = list(X.columns)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y,
)

# ============================================================
# 2. 训练最优模型 (RF, 作为 demo)
# ============================================================
print("训练 Random Forest (SHAP 解释目标)...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
)
rf.fit(X_train, y_train)

# ============================================================
# 3. SHAP 分析（对测试集采样 500 条，加速计算）
# ============================================================
print("计算 SHAP 值（采样 500 条测试样本）...")
X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

# 使用 TreeExplainer（专门优化树模型的 SHAP 计算）
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# shap_values 对于二分类返回 list of 2 arrays，取正类 (index=1)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# ============================================================
# 4. SHAP 可视化
# ============================================================

# 4a. Summary Plot — 最重要的 SHAP 图
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_vals, X_sample, feature_names=feature_names, show=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_summary.png")

# 4b. Bar Plot — 特征重要性（按平均 |SHAP| 排序）
fig, ax = plt.subplots(figsize=(8, 6))
shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                  plot_type="bar", show=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_bar.png")

# 4c. Waterfall — 单样本解释
fig, ax = plt.subplots(figsize=(10, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0],
        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) and not np.isscalar(explainer.expected_value) else (explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value),
        data=X_sample.iloc[0].values,
        feature_names=feature_names,
    ),
    show=False,
)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "shap_waterfall.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_waterfall.png")

# 4d. Dependence Plot
shap_vals_2d = shap_vals[:, :, 1] if shap_vals.ndim == 3 else shap_vals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
shap.dependence_plot("PAY_0", shap_vals_2d, X_sample,
                     feature_names=feature_names, show=False, ax=axes[0])
shap.dependence_plot("LIMIT_BAL", shap_vals_2d, X_sample,
                     feature_names=feature_names, show=False, ax=axes[1])
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "shap_dependence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_dependence.png")

print("\nSHAP 分析完成！")
print("  - shap_summary.png: 所有特征对所有样本的 SHAP 值分布（最重要的图）")
print("  - shap_bar.png: 特征重要性排序（按平均 SHAP 值）")
print("  - shap_waterfall.png: 单个样本的预测解释")
print("  - shap_dependence.png: 特征交互效应")
