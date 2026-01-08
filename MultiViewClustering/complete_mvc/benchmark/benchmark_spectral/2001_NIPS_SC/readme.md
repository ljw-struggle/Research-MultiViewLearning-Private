from sklearn.cluster import SpectralClustering

sc = SpectralClustering(
    n_clusters=5,
    affinity="precomputed",      # 或 "rbf" 等
    assign_labels="discretize",  # 这里切换
    random_state=0,
)
labels = sc.fit_predict(K)       # K 是亲和矩阵或特征矩阵

from sklearn.cluster import discretize

# vectors: (N, k) 的特征向量矩阵（Laplacian 的前 k 个特征向量）
labels = discretize(vectors, random_state=0)

import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import discretize

K = ...  # (N, N) 亲和矩阵
n_clusters = 5

L = laplacian(K, normed=True)
eigenvalues, eigenvectors = eigh(L)
H = eigenvectors[:, :n_clusters]  # 前 k 个最小特征值对应的特征向量

# 方式 A：用 discretize
labels_a = discretize(H)

# 方式 B：用 kmeans
from sklearn.cluster import KMeans
labels_b = KMeans(n_clusters=n_clusters, n_init=10).fit_predict(H)

原始数据 X
    │
    ▼  (1) 构建亲和矩阵
亲和矩阵 W          ← precomputed 跳过的是这一步
    │
    ▼  (2) 计算 Laplacian
Laplacian L = D - W  （或归一化版本）
    │
    ▼  (3) 特征分解
特征向量 H ∈ R^{N×k}  （前 k 个最小特征值对应的）
    │
    ▼  (4) 聚类
KMeans(H) → labels

# 手动实现 = spectral clustering
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans

L = laplacian(W, normed=True)          # (2)
_, H = eigh(L, subset_by_index=[0, k-1])  # (3)
labels = KMeans(n_clusters=k).fit_predict(H)  # (4)