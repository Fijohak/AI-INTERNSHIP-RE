"""
Iris 分类 — 多模型对比 (KNN / Logistic Regression / SVM)

数据集：sklearn 内置 Iris（150 条，3 类，4 特征）
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE = os.path.dirname(__file__)
FIGS = os.path.join(BASE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"类别: {target_names}")
print(f"特征: {feature_names}")

# ============================================================
# 2. 数据探索 & 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
colors = ["#2196F3", "#4CAF50", "#FF9800"]
for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
    for k in range(3):
        ax.hist(X[y == k, i], bins=12, alpha=0.6, color=colors[k], label=target_names[k])
    ax.set_title(name)
    ax.legend(fontsize=7)
fig.suptitle("Iris 特征分布直方图", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "feature_distribution.png"), dpi=120)
plt.close()
print("Saved: feature_distribution.png")

# 散点图矩阵
fig, axes = plt.subplots(4, 4, figsize=(11, 10))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        if i == j:
            for k in range(3):
                ax.hist(X[y == k, i], bins=10, alpha=0.6, color=colors[k])
            ax.set_xlabel(feature_names[i])
        else:
            for k in range(3):
                ax.scatter(X[y == k, j], X[y == k, i], s=3, alpha=0.6, color=colors[k])
            if i == 3:
                ax.set_xlabel(feature_names[j])
            if j == 0:
                ax.set_ylabel(feature_names[i])
fig.suptitle("Iris 散点图矩阵", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "scatter_matrix.png"), dpi=120)
plt.close()
print("Saved: scatter_matrix.png")

# ============================================================
# 3. 数据预处理
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. 模型训练 & 评估
# ============================================================
models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", random_state=42),
}

results = []
all_y_pred = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    all_y_pred[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": f"{acc:.4f}"})
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  准确率: {acc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

# ============================================================
# 5. 混淆矩阵可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, y_pred) in zip(axes, all_y_pred.items()):
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_title(name, fontsize=11)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(target_names)
    ax.set_yticklabels(target_names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    fontsize=11, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
fig.suptitle("混淆矩阵对比", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "confusion_matrices.png"), dpi=120)
plt.close()
print("Saved: confusion_matrices.png")

# ============================================================
# 6. 准确率对比条形图
# ============================================================
accs = [float(r["Accuracy"]) for r in results]
names = [r["Model"] for r in results]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(names, accs, color=colors[:3], width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{acc:.4f}", ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0.8, 1.05)
ax.set_ylabel("Accuracy")
ax.set_title("Iris 分类 — 多模型准确率对比", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "accuracy_comparison.png"), dpi=120)
plt.close()
print("Saved: accuracy_comparison.png")

# ============================================================
# 7. 保存指标
# ============================================================
df = pd.DataFrame(results)
df.to_csv(os.path.join(BASE, "results", "metrics.csv"), index=False)
print("\nSaved: results/metrics.csv")
print("\nDone!")
