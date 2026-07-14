"""
聚类分析 — KMeans / PCA / 层次聚类

数据集：Wine 数据集 (178 条, 13 特征, 3 类)
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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(__file__)
FIGS = os.path.join(BASE, "results", "figures")
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 1. 加载数据
# ============================================================
wine = load_wine()
X = wine.data
y_true = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"类别: {target_names}")

# ============================================================
# 2. 预处理
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降维到 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA 解释方差比: {pca.explained_variance_ratio_}")
print(f"PCA 累计解释方差: {pca.explained_variance_ratio_.sum():.3f}")

# ============================================================
# 3. Elbow Method — 确定最佳 K
# ============================================================
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(K_range, inertias, "bo-", markersize=6)
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method (Inertia)")

axes[1].plot(K_range, silhouettes, "go-", markersize=6)
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score")

best_k = K_range[np.argmax(silhouettes)]
axes[1].axvline(x=best_k, color="r", linestyle="--", label=f"Best K={best_k}")
axes[1].legend()
fig.suptitle("KMeans — 最佳 K 值选择", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "elbow_method.png"), dpi=120)
plt.close()
print(f"Saved: elbow_method.png (Best K = {best_k})")

# ============================================================
# 4. KMeans 聚类
# ============================================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

ari_km = adjusted_rand_score(y_true, y_kmeans)
sil_km = silhouette_score(X_scaled, y_kmeans)
homo_km = homogeneity_score(y_true, y_kmeans)
v_km = v_measure_score(y_true, y_kmeans)

print(f"\n{'=' * 50}")
print(f"  KMeans (K=3)")
print(f"{'=' * 50}")
print(f"  Adjusted Rand Index: {ari_km:.4f}")
print(f"  Silhouette Score:    {sil_km:.4f}")
print(f"  Homogeneity:         {homo_km:.4f}")
print(f"  V-Measure:           {v_km:.4f}")

# ============================================================
# 5. 层次聚类 (Agglomerative)
# ============================================================
agg = AgglomerativeClustering(n_clusters=3)
y_agg = agg.fit_predict(X_scaled)

ari_agg = adjusted_rand_score(y_true, y_agg)
sil_agg = silhouette_score(X_scaled, y_agg)

print(f"\n{'=' * 50}")
print(f"  Agglomerative Clustering (K=3, Ward)")
print(f"{'=' * 50}")
print(f"  Adjusted Rand Index: {ari_agg:.4f}")
print(f"  Silhouette Score:    {sil_agg:.4f}")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 颜色映射
cluster_colors = np.array(["#2196F3", "#4CAF50", "#FF9800"])

# PCA: Ground Truth
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_colors[y_true],
                   s=30, alpha=0.8, edgecolors="white", linewidth=0.5)
axes[0, 0].set_title("Ground Truth (PCA 2D)")
axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

# PCA: KMeans
axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_colors[y_kmeans],
                   s=30, alpha=0.8, edgecolors="white", linewidth=0.5)
axes[0, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                   s=200, c="red", marker="X", edgecolors="white", linewidth=1)
axes[0, 1].set_title(f"KMeans (ARI={ari_km:.3f})")
axes[0, 1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")

# PCA: Agglomerative
axes[0, 2].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_colors[y_agg],
                   s=30, alpha=0.8, edgecolors="white", linewidth=0.5)
axes[0, 2].set_title(f"Agglomerative (ARI={ari_agg:.3f})")
axes[0, 2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")

# KMeans: 原始特征空间的前 2 维
axes[1, 0].scatter(X[:, 0], X[:, 1], c=cluster_colors[y_true],
                   s=25, alpha=0.8, edgecolors="white", linewidth=0.5)
axes[1, 0].set_xlabel(feature_names[0]); axes[1, 0].set_ylabel(feature_names[1])
axes[1, 0].set_title("Feature Space: Alcohol vs Malic Acid (GT)")

axes[1, 1].scatter(X[:, 0], X[:, 1], c=cluster_colors[y_kmeans],
                   s=25, alpha=0.8, edgecolors="white", linewidth=0.5)
axes[1, 1].set_xlabel(feature_names[0])
axes[1, 1].set_title("Feature Space: KMeans Clusters")

axes[1, 2].scatter(X[:, 0], X[:, 1], c=cluster_colors[y_agg],
                   s=25, alpha=0.8, edgecolors="white", linewidth=0.5)
axes[1, 2].set_xlabel(feature_names[0])
axes[1, 2].set_title("Feature Space: Agglomerative Clusters")

fig.suptitle("Wine 数据集 — 聚类结果对比", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "clustering_comparison.png"), dpi=120)
plt.close()
print("Saved: clustering_comparison.png")

# 层次聚类树状图
fig, ax = plt.subplots(figsize=(12, 5))
Z = linkage(X_scaled, method="ward")
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_font_size=9)
ax.axhline(y=15, color="r", linestyle="--", lw=1, label="Cut (K=3)")
ax.set_title("层次聚类 — Ward 树状图（Top 30 节点）", fontsize=13)
ax.set_xlabel("Sample Index"); ax.set_ylabel("Distance")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(FIGS, "dendrogram.png"), dpi=120)
plt.close()
print("Saved: dendrogram.png")

# 模型对比
results = [
    {"Model": "KMeans (K=3)", "ARI": f"{ari_km:.4f}",
     "Silhouette": f"{sil_km:.4f}", "Homogeneity": f"{homo_km:.4f}",
     "V-Measure": f"{v_km:.4f}"},
    {"Model": "Agglomerative (K=3)", "ARI": f"{ari_agg:.4f}",
     "Silhouette": f"{sil_agg:.4f}", "Homogeneity": "-", "V-Measure": "-"},
]

pd.DataFrame(results).to_csv(
    os.path.join(BASE, "results", "metrics.csv"), index=False,
)
print("\nSaved: results/metrics.csv")
print("Done!")
