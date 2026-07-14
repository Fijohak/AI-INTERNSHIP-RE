"""
California Housing 线性回归

数据集：sklearn 内置 California Housing（20,640 条，8 特征）
"""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(__file__)
FIGS = os.path.join(BASE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
data = fetch_california_housing()
X = data.data
y = data.target
feature_names = data.feature_names

print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"目标: 房价中位数 ($100,000)")
print(f"特征: {list(feature_names)}")

# ============================================================
# 2. 数据探索
# ============================================================
# 目标变量分布
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(y, bins=50, color="#2196F3", alpha=0.8, edgecolor="white")
axes[0].set_title("房价中位数分布（原始）")
axes[0].set_xlabel("Price ($100k)")
axes[0].set_ylabel("Count")

axes[1].hist(np.log1p(y), bins=50, color="#4CAF50", alpha=0.8, edgecolor="white")
axes[1].set_title("房价中位数分布（log1p 变换后）")
axes[1].set_xlabel("log(1 + Price)")
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "target_distribution.png"), dpi=120)
plt.close()
print("Saved: target_distribution.png")

# 特征相关性热力图
corr = np.corrcoef(X.T)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
names_short = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
               "Population", "AveOccup", "Latitude", "Longitude"]
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_xticklabels(names_short, rotation=45, ha="right")
ax.set_yticklabels(names_short)
for i in range(8):
    for j in range(8):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(corr[i, j]) > 0.5 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_title("特征相关性矩阵", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "correlation_matrix.png"), dpi=120)
plt.close()
print("Saved: correlation_matrix.png")

# ============================================================
# 3. 数据预处理
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
)
print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. 模型训练 & 评估
# ============================================================
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n{'=' * 50}")
print(f"  Linear Regression — 测试集评估")
print(f"{'=' * 50}")
print(f"  R2:   {r2:.4f}")
print(f"  MSE:  {mse:.4f}")
print(f"  RMSE: {rmse:.4f}  ($100k)")
print(f"  MAE:  {mae:.4f}  ($100k)")

pd.DataFrame([{
    "Model": "Linear Regression",
    "R2": f"{r2:.4f}",
    "MSE": f"{mse:.4f}",
    "RMSE": f"{rmse:.4f}",
    "MAE": f"{mae:.4f}",
}]).to_csv(os.path.join(BASE, "results", "metrics.csv"), index=False)
print("Saved: results/metrics.csv")

# ============================================================
# 5. 可视化
# ============================================================
# 预测 vs 真实散点图
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, s=3, alpha=0.3, color="#2196F3")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="y = x")
ax.set_xlabel("真实值 ($100k)")
ax.set_ylabel("预测值 ($100k)")
ax.set_title(f"预测 vs 真实值 (R2 = {r2:.4f})", fontsize=13)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "pred_vs_true.png"), dpi=120)
plt.close()
print("Saved: pred_vs_true.png")

# 残差分析
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].scatter(y_pred, residuals, s=3, alpha=0.3, color="#FF9800")
axes[0].axhline(y=0, color="r", linestyle="--", lw=1.5)
axes[0].set_xlabel("预测值")
axes[0].set_ylabel("残差")
axes[0].set_title("残差 vs 预测值")

axes[1].hist(residuals, bins=50, color="#4CAF50", alpha=0.8, edgecolor="white")
axes[1].set_xlabel("残差")
axes[1].set_title("残差分布直方图")

axes[2].scatter(range(len(residuals)), np.sort(residuals), s=3, alpha=0.5, color="#2196F3")
axes[2].axhline(y=0, color="r", linestyle="--", lw=1)
axes[2].set_xlabel("样本（排序后）")
axes[2].set_ylabel("残差")
axes[2].set_title("残差 Q-Q 风格图")

fig.suptitle("残差分析", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "residual_analysis.png"), dpi=120)
plt.close()
print("Saved: residual_analysis.png")

# 特征权重
fig, ax = plt.subplots(figsize=(8, 5))
coefs = model.coef_
sorted_idx = np.argsort(np.abs(coefs))[::-1]
bar_colors = ["#4CAF50" if c > 0 else "#F44336" for c in coefs[sorted_idx]]
ax.barh(range(len(coefs)), coefs[sorted_idx], color=[bar_colors[i] for i in range(len(coefs))])
ax.set_yticks(range(len(coefs)))
ax.set_yticklabels(np.array(names_short)[sorted_idx])
ax.axvline(x=0, color="black", lw=0.8)
ax.set_xlabel("Coefficient")
ax.set_title("特征权重（正值=正相关，负值=负相关）", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "feature_coefficients.png"), dpi=120)
plt.close()
print("Saved: feature_coefficients.png")

print("\nDone!")
