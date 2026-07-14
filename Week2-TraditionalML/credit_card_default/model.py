"""
信用卡违约预测 — 多模型对比

数据集：UCI Default of Credit Card Clients (30,000 条, 23 特征)
模型：Logistic Regression, Decision Tree, Random Forest, XGBoost
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(__file__)
FIGS = os.path.join(BASE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
print("下载 UCI Credit Card Default 数据集...")
df = pd.read_excel(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
    header=1,
)
df.rename(columns={"default payment next month": "default"}, inplace=True)
print(f"样本数: {df.shape[0]}, 特征数: {df.shape[1] - 1}")
print(f"违约率: {df['default'].mean():.2%}")

X = df.drop(columns=["ID", "default"])
y = df["default"]
feature_names = list(X.columns)

# ============================================================
# 2. 数据探索
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
y.value_counts().plot.pie(
    autopct="%1.1f%%", labels=["未违约", "违约"],
    colors=["#4CAF50", "#F44336"], ax=axes[0],
)
axes[0].set_ylabel("")
axes[0].set_title("类别分布")

# 按违约状态对比关键特征
key_features = ["LIMIT_BAL", "AGE", "PAY_0"]
for i, feat in enumerate(key_features):
    g = df.groupby("default")[feat].mean()
    axes[1].bar([f"{feat}\n未违约", f"{feat}\n违约"], [g[0], g[1]], color=["#4CAF50", "#F44336"])
axes[1].set_title("关键特征均值对比")
axes[1].set_ylabel("均值")
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "eda.png"), dpi=120)
plt.close()
print("Saved: eda.png")

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
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
    ),
}

results = []
all_y_pred = {}
all_y_proba = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    all_y_pred[name] = y_pred
    all_y_proba[name] = y_proba

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results.append({
        "Model": name, "Accuracy": f"{acc:.4f}", "AUC": f"{auc:.4f}",
    })
    print(f"\n{'=' * 50}")
    print(f"  {name} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred))

# ============================================================
# 5. 可视化
# ============================================================
# ROC 曲线
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#2196F3", "#FF9800", "#4CAF50"]
for (name, proba), c in zip(all_y_proba.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={auc_val:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC 曲线对比", fontsize=13)
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "roc_curves.png"), dpi=120)
plt.close()
print("Saved: roc_curves.png")

# 混淆矩阵
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, y_pred) in zip(axes, all_y_pred.items()):
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_title(name, fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticklabels(["No Default", "Default"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
fig.suptitle("混淆矩阵对比", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "confusion_matrices.png"), dpi=120)
plt.close()
print("Saved: confusion_matrices.png")

# RF 特征重要性
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
top_n = 15
idx = np.argsort(importances)[-top_n:]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(top_n), importances[idx], color="#2196F3")
ax.set_yticks(range(top_n))
ax.set_yticklabels(np.array(feature_names)[idx])
ax.set_xlabel("Importance")
ax.set_title(f"Random Forest Top-{top_n} 特征重要性", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "feature_importance.png"), dpi=120)
plt.close()
print("Saved: feature_importance.png")

# 准确率对比
fig, ax = plt.subplots(figsize=(7, 4))
accs = [float(r["Accuracy"]) for r in results]
names = [r["Model"] for r in results]
bars = ax.bar(names, accs, color=colors[:3], width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{acc:.4f}", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_title("信用卡违约预测 — 模型准确率对比", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "accuracy_comparison.png"), dpi=120)
plt.close()
print("Saved: accuracy_comparison.png")

pd.DataFrame(results).to_csv(os.path.join(BASE, "results", "metrics.csv"), index=False)
print("\nSaved: results/metrics.csv")
print("Done!")
