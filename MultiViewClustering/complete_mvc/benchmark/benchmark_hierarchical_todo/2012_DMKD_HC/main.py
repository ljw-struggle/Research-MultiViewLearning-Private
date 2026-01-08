import numpy as np

class MyHierarchicalClustering:
    def __init__(self, n_clusters=2, linkage="single"):
        """
        Parameters
        ----------
        n_clusters : int
            最终保留的簇数
        linkage : str
            簇间距离定义，可选:
            - 'single'   : 最短距离
            - 'complete' : 最长距离
            - 'average'  : 平均距离
        """
        if linkage not in ["single", "complete", "average"]:
            raise ValueError("linkage must be 'single', 'complete', or 'average'")
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.clusters_ = None

    def _euclidean_distance_matrix(self, X):
        """
        计算样本两两之间的欧氏距离矩阵
        shape: (n_samples, n_samples)
        """
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        return dist

    def _cluster_distance(self, cluster_a, cluster_b, sample_dist):
        """
        计算两个簇之间的距离
        cluster_a, cluster_b: list[int]
        sample_dist: 样本间距离矩阵
        """
        dists = sample_dist[np.ix_(cluster_a, cluster_b)]

        if self.linkage == "single":
            return np.min(dists)
        elif self.linkage == "complete":
            return np.max(dists)
        elif self.linkage == "average":
            return np.mean(dists)

    def fit(self, X):
        """
        执行凝聚型层次聚类
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot be greater than n_samples")

        # 1. 先计算样本间距离矩阵
        sample_dist = self._euclidean_distance_matrix(X)

        # 2. 初始时每个样本自成一个簇
        clusters = [[i] for i in range(n_samples)]

        # 3. 不断合并最近的两个簇，直到簇数变成 n_clusters
        while len(clusters) > self.n_clusters:
            min_dist = np.inf
            merge_i, merge_j = -1, -1

            # 找到最近的两个簇
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(clusters[i], clusters[j], sample_dist)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # 合并这两个簇
            new_cluster = clusters[merge_i] + clusters[merge_j]

            # 删除旧簇，加入新簇
            new_clusters = []
            for k in range(len(clusters)):
                if k != merge_i and k != merge_j:
                    new_clusters.append(clusters[k])
            new_clusters.append(new_cluster)
            clusters = new_clusters

        # 4. 生成 labels
        labels = np.empty(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for sample_id in cluster:
                labels[sample_id] = cluster_id

        self.clusters_ = clusters
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
if __name__ == "__main__":
    np.random.seed(42)

    X1 = np.random.randn(5, 2) + np.array([0, 0])
    X2 = np.random.randn(5, 2) + np.array([5, 5])
    X3 = np.random.randn(5, 2) + np.array([0, 5])

    X = np.vstack([X1, X2, X3])

    model = MyHierarchicalClustering(n_clusters=3, linkage="average")
    labels = model.fit_predict(X)

    print("labels:")
    print(labels)
    print("\nclusters:")
    print(model.clusters_)