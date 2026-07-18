"""
糖尿病预测 — 多模型对比 + GridSearchCV + SHAP

数据集：Pima Indians Diabetes (768 条, 8 特征)
模型：Logistic Regression, Decision Tree, Random Forest, XGBoost
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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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
print(f"样本数: {df.shape[0]}, 特征数: {len(FEATURE_NAMES)}")
print(f"患病率: {df['Outcome'].mean():.2%}")

X = df[FEATURE_NAMES].copy()
y = df["Outcome"]

# ============================================================
# 2. 数据探索 (EDA)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, (ax, name) in enumerate(zip(axes.flat, FEATURE_NAMES)):
    for k, label, color in [(0, "Healthy", "#4CAF50"), (1, "Diabetes", "#F44336")]:
        subset = X[y == k][name]
        subset = subset[subset > 0]
        ax.hist(subset, bins=20, alpha=0.6, color=color, label=label)
    ax.set_title(name, fontsize=9)
    if i == 0:
        ax.legend(fontsize=7)
fig.suptitle("Pima Indians Diabetes — 特征分布（按患病状态）", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "feature_distribution.png"), dpi=120)
plt.close()
print("Saved: feature_distribution.png")

# ============================================================
# 3. 数据预处理
# ============================================================
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    X[col] = X[col].replace(0, np.nan)
    X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"\n训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. 模型训练（含 GridSearchCV）
# ============================================================

# 4a. Logistic Regression
print("\n--- Logistic Regression ---")
lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_scaled, y_train)

# 4b. Decision Tree
print("--- Decision Tree (GridSearchCV) ---")
dt_params = {"max_depth": [3, 4, 5, 7], "min_samples_split": [2, 5]}
dt_gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42), dt_params,
    cv=3, scoring="roc_auc",
)
dt_gs.fit(X_train_scaled, y_train)
print(f"  最佳参数: {dt_gs.best_params_}")

# 4c. Random Forest
print("--- Random Forest (GridSearchCV) ---")
rf_params = {"n_estimators": [50, 100], "max_depth": [4, 6, 8]}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=42), rf_params,
    cv=3, scoring="roc_auc",
)
rf_gs.fit(X_train_scaled, y_train)
print(f"  最佳参数: {rf_gs.best_params_}")

# 4d. XGBoost
print("--- XGBoost (GridSearchCV) ---")
xgb_params = {
    "n_estimators": [50, 100],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1],
}
xgb_gs = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"), xgb_params,
    cv=3, scoring="roc_auc",
)
xgb_gs.fit(X_train_scaled, y_train)
print(f"  最佳参数: {xgb_gs.best_params_}")

# ============================================================
# 5. 测试集评估
# ============================================================
models = {
    "Logistic Regression": lr,
    "Decision Tree": dt_gs.best_estimator_,
    "Random Forest": rf_gs.best_estimator_,
    "XGBoost": xgb_gs.best_estimator_,
}

results = []
all_y_pred = {}
all_y_proba = {}

for name, model in models.items():
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
# 6. 可视化
# ============================================================
# ROC 曲线
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
for (name, proba), c in zip(all_y_proba.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={auc_val:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC 曲线对比（含 XGBoost）", fontsize=13); ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "roc_curves.png"), dpi=120)
plt.close()
print("Saved: roc_curves.png")

# 混淆矩阵
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, y_pred) in zip(axes, all_y_pred.items()):
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_title(name, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy", "Diabetes"])
    ax.set_yticklabels(["Healthy", "Diabetes"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.suptitle("混淆矩阵对比（含 XGBoost）", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "confusion_matrices.png"), dpi=120)
plt.close()
print("Saved: confusion_matrices.png")

# XGBoost 特征重要性
xgb_model = models["XGBoost"]
importances = xgb_model.feature_importances_
idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(range(len(FEATURE_NAMES)), importances[idx], color="#E91E63")
ax.set_yticks(range(len(FEATURE_NAMES)))
ax.set_yticklabels(np.array(FEATURE_NAMES)[idx])
ax.set_xlabel("Importance")
ax.set_title("XGBoost — 特征重要性", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "feature_importance.png"), dpi=120)
plt.close()
print("Saved: feature_importance.png")

pd.DataFrame(results).to_csv(os.path.join(BASE, "results", "metrics.csv"), index=False)
print("\nSaved: results/metrics.csv")
print("Done!")
