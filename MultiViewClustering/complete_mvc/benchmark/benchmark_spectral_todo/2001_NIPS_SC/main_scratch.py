import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, discretize
from scipy.sparse.csgraph import laplacian

########################################################################################
### Generate data
########################################################################################
def make_3d_spirals(n_samples=1000, noise=0.05, n_turns=3, height=5.0, radius=1.0, random_state=42):
    # Generate 3D spirals data.
    t = np.linspace(0, n_turns * 2 * np.pi, n_samples // 2)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height * t / (n_turns * 2 * np.pi)
    # Stack the x, y, z coordinates to get the 3D data. (Two spirals: one up, one down)
    X_1 = np.stack([x, y, z], axis=1); X_2 = np.stack([x, y, z - height], axis=1)
    X = np.vstack([X_1, X_2]) # X: (n_samples, 3)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2)) # y: (n_samples,), 0-indexed.
    rng = np.random.default_rng(random_state) # random number generator only for this function.
    X += rng.normal(scale=noise, size=X.shape) # Add noise to the data. (X: (n_samples, 3))
    return X, y # X: (n_samples, 3), y: (n_samples,), 0-indexed. (Two spirals: one up, one down)

########################################################################################
### Construct the similarity matrix
########################################################################################
def rbf_affinity(X, gamma=15.0):
    # Construct the RBF similarity matrix: W_ij = exp(-gamma * ||x_i - x_j||^2)
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2 * X @ X.T
    sq_dists = np.maximum(sq_dists, 0.0) # make sure the squared distances are non-negative.
    W = np.exp(-gamma * sq_dists) # W: (n_samples, n_samples)
    np.fill_diagonal(W, 0.0) # Set the diagonal elements to 0.
    return W # W: (n_samples, n_samples)

def knn_affinity(X, n_neighbors=10, gamma=15.0, mutual=False):
    # Construct the kNN graph similarity matrix, and then use RBF as the edge weight.
    sq_norms = np.sum(X ** 2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2 * X @ X.T
    sq_dists = np.maximum(sq_dists, 0.0)
    W = np.exp(-gamma * sq_dists) # W: (n_samples, n_samples)
    # Find the n_neighbors nearest neighbors for each data point (excluding itself).
    neighbors = np.argsort(sq_dists, axis=1)[:, 1:n_neighbors + 1]
    mask = np.zeros((X.shape[0], X.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        mask[i, neighbors[i]] = 1.0
    mask = np.logical_and(mask, mask.T).astype(float) if mutual else np.logical_or(mask, mask.T).astype(float) # mutual: True: mutual kNN, False: symmetric kNN.
    # Mask the similarity matrix W.
    W = W * mask
    np.fill_diagonal(W, 0.0) # Set the diagonal elements to 0.
    return W # W: (n_samples, n_samples)

########################################################################################
### Spectral clustering
########################################################################################
# Spectral clustering algorithm flow:
# 1. Construct the similarity matrix W
# 2. Construct the degree matrix D
# 3. Construct the symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
# 4. Eigen decomposition
# 5. Take the eigenvectors corresponding to the smallest n_clusters eigenvalues
# 6. Row normalization: embedding = U / row_norms
# 7. KMeans clustering or discretize the labels
def spectral_clustering_numpy(X, n_clusters=2, affinity="knn", gamma=15.0, n_neighbors=10, random_state=42):
    # Implement spectral clustering using numpy and sklearn KMeans.
    # 1. Construct the similarity matrix W
    assert affinity in ["rbf", "knn"], "affinity must be 'rbf' or 'knn'"
    affinity_dict = {"rbf": lambda X: rbf_affinity(X, gamma=gamma), "knn": lambda X: knn_affinity(X, n_neighbors=n_neighbors, gamma=gamma, mutual=False)}
    W = affinity_dict[affinity](X)
    # 2. Construct the degree matrix D
    degrees = np.sum(W, axis=1) # degrees: (n_samples,)
    D = np.diag(degrees) # D: (n_samples, n_samples)
    # 3. Construct the symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-12)) # D_inv_sqrt: (n_samples, n_samples)
    L_sym = np.eye(X.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt # L_sym: (n_samples, n_samples)
    # L_sym = laplacian(W, normed=True) # L_sym: (n_samples, n_samples)
    # Laplacian matrix has three forms: L_unnormalized = D - W, L_sym = I - D^{-1/2} W D^{-1/2}, L_rw = D^{-1} W.
    # if use L_rw, should use the np.linalg.eig to get the eigenvalues and eigenvectors. And should sort the eigenvalues in decreasing order.
    # 4. Eigen decomposition
    # L_sym is a symmetric matrix, use eigh to get the eigenvalues and eigenvectors. sorted by the eigenvalues in increasing order.
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym) # eigenvalues: (n_samples,), eigenvectors: (n_samples, n_samples)
    # print("The first 10 eigenvalues:", eigenvalues[:10])
    # 5. Take the eigenvectors corresponding to the smallest n_clusters eigenvalues
    U = eigenvectors[:, :n_clusters] # U: (n_samples, n_clusters)
    # 6. Row normalization: embedding = U / row_norms
    row_norms = np.linalg.norm(U, axis=1, keepdims=True) # row_norms: (n_samples, 1)
    row_norms[row_norms == 0] = 1.0 # set the zero elements to 1.
    embedding = U / row_norms # embedding: (n_samples, n_clusters)
    # 7. KMeans clustering or discretize the labels
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state) # kmeans: KMeans object
    labels = kmeans.fit_predict(embedding) # labels: (n_samples,)
    # labels = discretize(embedding, random_state=random_state) # labels: (n_samples,)
    return labels, embedding

if __name__ == "__main__":
    # 1. Generate data
    # X, y_true = make_moons(n_samples=300, noise=0.08, random_state=42) # X: (300, 2), y_true: (300,)
    # X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05) # X: (500, 2), y_true: (500,)
    X, y_true = make_3d_spirals(n_samples=1000, noise=0.05, n_turns=1, height=1.2, radius=1.0, random_state=42) # X: (500, 3), y_true: (500,)

    # 2. Spectral clustering
    labels, embedding = spectral_clustering_numpy(X, n_clusters=2, affinity="knn", gamma=15.0, n_neighbors=10, random_state=42)
    # 3. Visualization
    if X.shape[1] == 3: # Project the 3D data to 2D plane. 45 degrees rotation around the x-axis.
        X_2d = X[:, [0, 1, 2]] @ np.array([[1, 0, 0], [0, np.cos(np.pi/4), -np.sin(np.pi/4)], [0, np.sin(np.pi/4), np.cos(np.pi/4)]]) # X_2d: (n_samples, 2)
    else:
        X_2d = X # X_2d: (n_samples, 2)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, s=10)
    plt.title("Ground Truth")
    plt.subplot(1, 3, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, s=10)
    plt.title("Spectral Clustering")
    plt.subplot(1, 3, 3)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, s=10)
    plt.title("Spectral Embedding")
    plt.tight_layout()
    plt.savefig("./result/spectral_clustering_scratch.png")
