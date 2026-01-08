"""
测试 K-means 是否可能产生少于目标聚类数的情况
"""
import numpy as np
from sklearn.cluster import KMeans
from _utils import check_kmeans_clusters

print("=" * 60)
print("测试：K-means 是否可能产生少于目标聚类数的情况")
print("=" * 60)

# 情况1：正常情况
print("\n【情况1】正常数据，目标聚类数 = 5")
np.random.seed(42)
data1 = np.random.randn(100, 10)
kmeans1 = KMeans(n_clusters=5, n_init=10, random_state=42)
preds1 = kmeans1.fit_predict(data1)
check_kmeans_clusters(preds1, 5)

# 情况2：样本数少于聚类数
print("\n【情况2】样本数 < 聚类数，目标聚类数 = 10，样本数 = 5")
np.random.seed(42)
data2 = np.random.randn(5, 10)
kmeans2 = KMeans(n_clusters=10, n_init=10, random_state=42)
preds2 = kmeans2.fit_predict(data2)
check_kmeans_clusters(preds2, 10)

# 情况3：数据高度聚集
print("\n【情况3】数据高度聚集，目标聚类数 = 5")
np.random.seed(42)
data3 = np.random.randn(20, 10) * 0.1 + np.array([1] * 10)
kmeans3 = KMeans(n_clusters=5, n_init=10, random_state=42)
preds3 = kmeans3.fit_predict(data3)
check_kmeans_clusters(preds3, 5)

# 情况4：演示标签不连续的情况
print("\n【情况4】演示标签不连续的情况（手动创建）")
y_pred_manual = np.array([0, 0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4])  # 缺少标签2
check_kmeans_clusters(y_pred_manual, 5)

print("\n" + "=" * 60)
print("结论：")
print("1. sklearn 的 KMeans 通常会确保产生 k 个聚类（即使有空聚类也会重新初始化）")
print("2. 但在极端情况下（样本数 < 聚类数），可能会产生少于 k 个聚类")
print("3. 聚类标签可能不连续（如 [0, 1, 3, 4, 5]），但这是正常的")
print("4. 修复后的 clustering_acc 使用 np.unique(y_pred) 而不是 y_pred.max()+1")
print("   这样可以避免因为标签不连续而高估聚类数")
print("=" * 60)


