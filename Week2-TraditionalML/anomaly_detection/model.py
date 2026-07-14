"""
异常行为识别 — Isolation Forest / One-Class SVM

数据集：Credit Card Fraud Detection 子集 + 合成异常
对比 Isolation Forest 和 One-Class SVM 在异常检测上的表现
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
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

BASE = os.path.dirname(__file__)
FIGS = os.path.join(BASE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

print("生成合成数据集用于异常检测演示...")

# ============================================================
# 1. 生成数据：正常点 (簇) + 异常点 (outliers)
# ============================================================
rng = np.random.RandomState(42)
n_normal = 1000
n_outliers = 50

# 正常数据：3 个高斯簇
X_normal, _ = make_blobs(
    n_samples=n_normal, centers=3, cluster_std=0.8, random_state=42,
)

# 异常数据：随机散布
X_outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 2))

X = np.vstack([X_normal, X_outliers])
y = np.hstack([np.ones(n_normal), -np.ones(n_outliers)])  # 1=正常, -1=异常

print(f"正常样本: {n_normal}, 异常样本: {n_outliers} ({n_outliers / len(y):.1%})")

# ============================================================
# 2. 数据可视化（真实标签）
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X_normal[:, 0], X_normal[:, 1], s=8, alpha=0.5,
           c="#4CAF50", label="Normal")
ax.scatter(X_outliers[:, 0], X_outliers[:, 1], s=30, alpha=0.9,
           c="#F44336", marker="x", label="Anomaly")
ax.set_title("数据集 — 正常点 vs 异常点", fontsize=13)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "ground_truth.png"), dpi=120)
plt.close()
print("Saved: ground_truth.png")

# ============================================================
# 3. 预处理
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=(y == -1),
)

# 训练集只用正常样本（One-Class SVM 和 Isolation Forest 的典型用法）
X_train_normal = X_train[y_train == 1]

# ============================================================
# 4. 模型训练 & 评估
# ============================================================
models = {
    "Isolation Forest": IsolationForest(
        n_estimators=100, contamination=0.05, random_state=42,
    ),
    "One-Class SVM": OneClassSVM(
        kernel="rbf", nu=0.05, gamma="scale",
    ),
}

results = []
all_y_pred = {}

for name, model in models.items():
    model.fit(X_train_normal)
    y_pred = model.predict(X_test)  # 1=正常, -1=异常

    # 转为 0/1 标签以便评估 (1=异常)
    y_test_binary = (y_test == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)

    all_y_pred[name] = y_pred

    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test_binary, y_pred_binary)

    # 异常检测专用指标
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    results.append({
        "Model": name,
        "Accuracy": f"{acc:.4f}",
        "AUC": f"{auc:.4f}",
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
    })
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"\n  Confusion Matrix (TN/FP/FN/TP):")
    print(f"  {cm}")

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# 真实标签
for label, c, marker in [(1, "#4CAF50", "o"), (-1, "#F44336", "x")]:
    mask = y_test == label
    axes[0].scatter(X_test[mask, 0], X_test[mask, 1], s=12, alpha=0.5,
                    c=c, marker=marker)
axes[0].set_title("Ground Truth")

# 模型预测
for ax, (name, y_pred) in zip(axes[1:], all_y_pred.items()):
    for label, c, marker in [(1, "#4CAF50", "o"), (-1, "#F44336", "x")]:
        mask = y_pred == label
        ax.scatter(X_test[mask, 0], X_test[mask, 1], s=12, alpha=0.5,
                   c=c, marker=marker)
    ax.set_title(f"{name} Predictions")

for ax in axes:
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
fig.suptitle("异常检测 — Ground Truth vs 模型预测", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "anomaly_detection_results.png"), dpi=120)
plt.close()
print("Saved: anomaly_detection_results.png")

# 决策边界对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5, 200),
    np.linspace(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5, 200),
)
for ax, (name, model) in zip(axes, models.items()):
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.5)
    ax.scatter(X_test[:, 0], X_test[:, 1], s=5, c=(y_test == -1).astype(int),
               cmap="RdYlGn_r", alpha=0.6)
    ax.set_title(f"{name} — Decision Boundary")
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "decision_boundary.png"), dpi=120)
plt.close()
print("Saved: decision_boundary.png")

pd.DataFrame(results).to_csv(os.path.join(BASE, "results", "metrics.csv"), index=False)
print("\nSaved: results/metrics.csv")
print("Done!")
