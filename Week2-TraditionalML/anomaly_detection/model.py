"""
异常行为识别 — Isolation Forest / One-Class SVM

数据集：Wine 数据集，采用半监督异常检测范式
  - 训练集：仅 class_0 和 class_1（视为"正常"）
  - 测试集：包含 class_2（视为"异常"），混入正常样本
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
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
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

# ============================================================
# 1. 加载数据 & 构造异常检测场景
# ============================================================
print("加载 Wine 数据集，构造异常检测场景...")
wine = load_wine()
X_all = wine.data
y_all = wine.target
feature_names = wine.feature_names

# 策略：class_0 和 class_1 视为"正常"，class_2 视为"异常"
# 训练集：仅正常样本
# 测试集：正常样本 + 异常样本混合
normal_mask = y_all != 2
X_normal = X_all[normal_mask]   # 正常: class_0 + class_1 (130 条)
X_anomaly = X_all[~normal_mask] # 异常: class_2 (48 条)

print(f"正常样本: {X_normal.shape[0]}, 异常样本: {X_anomaly.shape[0]}")
print(f"异常比例: {X_anomaly.shape[0] / X_all.shape[0]:.1%}")

# ============================================================
# 2. 数据探索：PCA 2D 可视化
# ============================================================
scaler_viz = StandardScaler()
X_all_scaled_viz = scaler_viz.fit_transform(X_all)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all_scaled_viz)

fig, ax = plt.subplots(figsize=(7, 6))
for label, c, marker, name in [
    (0, "#2196F3", "o", "Class 0 (正常)"),
    (1, "#4CAF50", "o", "Class 1 (正常)"),
    (2, "#F44336", "x", "Class 2 (异常)"),
]:
    mask = y_all == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=25, alpha=0.7,
               c=c, marker=marker, label=name, edgecolors="white", linewidth=0.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("Wine 数据集 PCA 2D — 正常 vs 异常", fontsize=13)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "ground_truth.png"), dpi=120)
plt.close()
print("Saved: ground_truth.png")

# ============================================================
# 3. 预处理 & 划分
# ============================================================
# 正常样本中 70% 做训练，30% 留到测试集
X_train_norm, X_test_norm = train_test_split(
    X_normal, test_size=0.3, random_state=42,
)

# 测试集 = 保留的正常样本 + 所有异常样本
X_test = np.vstack([X_test_norm, X_anomaly])
y_test = np.hstack([
    np.ones(X_test_norm.shape[0]),    # 1 = 正常
    -np.ones(X_anomaly.shape[0]),     # -1 = 异常
])

print(f"训练集（仅正常）: {X_train_norm.shape[0]}")
print(f"测试集（正常+异常）: {X_test.shape[0]} (正常: {X_test_norm.shape[0]}, 异常: {X_anomaly.shape[0]})")

# 标准化：在训练集上 fit，应用到测试集
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_norm)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. 模型训练 & 评估
# ============================================================
models = {
    "Isolation Forest": IsolationForest(
        n_estimators=100, contamination=0.15, random_state=42,
    ),
    "One-Class SVM": OneClassSVM(
        kernel="rbf", nu=0.15, gamma="scale",
    ),
}

results = []
all_y_pred = {}

for name, model in models.items():
    model.fit(X_train_scaled)
    y_pred = model.predict(X_test_scaled)  # 1=正常, -1=异常

    all_y_pred[name] = y_pred

    # 转为 0/1 (1=异常) 用于计算 AUC
    y_test_binary = (y_test == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)

    acc = (y_pred == y_test).mean()
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    auc = roc_auc_score(y_test_binary, y_pred_binary)

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
    print(f"  Precision: {precision:.4f}  (说异常的人中，多少真是异常)")
    print(f"  Recall:    {recall:.4f}  (真正异常中，找出了多少)")
    print(f"  Confusion Matrix (TN/FP/FN/TP):")
    print(f"  {cm}")

# ============================================================
# 5. 可视化
# ============================================================
# PCA 降维 + 模型预测
X_test_pca = pca.transform(scaler.inverse_transform(X_test_scaled)
                           if hasattr(scaler, 'inverse_transform') else X_test_scaled)
# 简化：直接用 raw X_test 做 PCA
X_test_raw = np.vstack([X_test_norm, X_anomaly])
X_test_pca = pca.transform(StandardScaler().fit_transform(X_test_raw))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for label, c, marker in [(1, "#4CAF50", "o"), (-1, "#F44336", "x")]:
    mask = y_test == label
    axes[0].scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], s=15, alpha=0.6,
                    c=c, marker=marker)
axes[0].set_title("Ground Truth")

for ax, (name, y_pred) in zip(axes[1:], all_y_pred.items()):
    for label, c, marker in [(1, "#4CAF50", "o"), (-1, "#F44336", "x")]:
        mask = y_pred == label
        ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], s=15, alpha=0.6,
                   c=c, marker=marker)
    ax.set_title(f"{name}")

for ax in axes:
    ax.set_xlabel(f"PC1"); ax.set_ylabel(f"PC2")
fig.suptitle("异常检测 — Ground Truth vs 模型预测 (Wine 数据集)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "anomaly_detection_results.png"), dpi=120)
plt.close()
print("Saved: anomaly_detection_results.png")

pd.DataFrame(results).to_csv(os.path.join(BASE, "results", "metrics.csv"), index=False)
print("\nSaved: results/metrics.csv")
print("Done!")
