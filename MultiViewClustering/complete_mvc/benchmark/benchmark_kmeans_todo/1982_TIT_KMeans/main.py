import numpy as np


class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=None):
        """
        Parameters
        ----------
        n_clusters : int
            聚类簇数
        max_iter : int
            最大迭代次数
        tol : float
            收敛阈值，当聚类中心整体移动量小于该值时停止
        random_state : int or None
            随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None  # 所有样本到其所属中心的平方距离和

    def _init_centroids(self, X):
        """
        从样本中随机选择 n_clusters 个点作为初始中心
        """
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _compute_distances(self, X, centroids):
        """
        计算每个样本到每个聚类中心的欧氏距离平方
        返回 shape: (n_samples, n_clusters)
        """
        # X: (n_samples, n_features)
        # centroids: (n_clusters, n_features)
        # broadcasting 后得到: (n_samples, n_clusters, n_features)
        distances = np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        return distances

    def fit(self, X):
        """
        训练 K-Means
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")

        centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            # 1. 分配样本到最近的聚类中心
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            # 2. 更新聚类中心
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]

                # 如果某个簇没有样本，重新随机选一个点作为中心
                if len(cluster_points) == 0:
                    rng = np.random.default_rng(self.random_state)
                    new_centroids[k] = X[rng.integers(0, X.shape[0])]
                else:
                    new_centroids[k] = np.mean(cluster_points, axis=0)

            # 3. 判断收敛
            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids

            if shift < self.tol:
                break

        # 保存结果
        final_distances = self._compute_distances(X, centroids)
        final_labels = np.argmin(final_distances, axis=1)

        self.cluster_centers_ = centroids
        self.labels_ = final_labels
        self.inertia_ = np.sum(final_distances[np.arange(X.shape[0]), final_labels])

        return self

    def predict(self, X):
        """
        预测新样本属于哪个簇
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X, dtype=float)
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
if __name__ == "__main__":
    np.random.seed(42)

    # 构造三团二维数据
    X1 = np.random.randn(50, 2) + np.array([0, 0])
    X2 = np.random.randn(50, 2) + np.array([5, 5])
    X3 = np.random.randn(50, 2) + np.array([0, 5])
    X = np.vstack([X1, X2, X3])

    kmeans = MyKMeans(n_clusters=3, max_iter=100, tol=1e-4, random_state=42)
    labels = kmeans.fit_predict(X)

    print("聚类中心:")
    print(kmeans.cluster_centers_)
    print("\n每个样本的簇标签:")
    print(labels)
    print("\nInertia:")
    print(kmeans.inertia_)