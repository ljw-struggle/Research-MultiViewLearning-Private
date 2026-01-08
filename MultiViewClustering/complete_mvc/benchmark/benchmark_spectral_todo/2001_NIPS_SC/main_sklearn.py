import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

########################################################################################
### Construct the similarity matrix
########################################################################################
def affinity_rbf(X, gamma=1.0):
    # RBF kernel: W_ij = exp(-gamma * ||x_i - x_j||^2)
    W = rbf_kernel(X, gamma=gamma) # W: (n_samples, n_samples)
    np.fill_diagonal(W, 0.0)
    return W # W: (n_samples, n_samples)

def affinity_knn(X, n_neighbors=10, gamma=1.0):
    # Symmetric KNN graph with RBF edge weights.
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity", include_self=False) # A: (n_samples, n_samples), sparse
    A = 0.5 * (A + A.T) # make symmetric. A: (n_samples, n_samples), sparse
    W = rbf_kernel(X, gamma=gamma) * A.toarray() # W: (n_samples, n_samples)
    np.fill_diagonal(W, 0.0)
    return W # W: (n_samples, n_samples)

def affinity_adaptive_rbf(X, k=7):
    # Self-tuning RBF (Zelnik-Manor & Perona, 2004): sigma_i = dist(x_i, k-th neighbor).
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X) # distances: (n_samples, k+1)
    sigma = np.maximum(distances[:, k], 1e-10) # sigma: (n_samples,), the distance to the k-th neighbor.
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2) # sq_dists: (n_samples, n_samples)
    W = np.exp(-sq_dists / np.outer(sigma, sigma)) # W: (n_samples, n_samples)
    np.fill_diagonal(W, 0.0)
    return W # W: (n_samples, n_samples)

########################################################################################
### Spectral clustering (sklearn, precomputed affinity)
########################################################################################
# sklearn SpectralClustering with affinity="precomputed":
# 1. Take the precomputed affinity matrix W as input.
# 2. Construct the symmetric normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
# 3. Eigen decomposition of L_sym, take the smallest n_clusters eigenvectors.
# 4. KMeans clustering on the eigenvector embedding.
def spectral_clustering_sklearn(X, W, n_clusters=2, random_state=42):
    sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=random_state, assign_labels="kmeans") 
    # sc = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=random_state, assign_labels="discretize")
    labels = sc.fit_predict(W) # labels: (n_samples,)
    return labels # labels: (n_samples,)

########################################################################################
### Main
########################################################################################
if __name__ == "__main__":
    # 1. Generate data
    X, y_true = make_moons(n_samples=500, noise=0.06, random_state=42) # X: (500, 2), y_true: (500,)
    # X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42) # X: (500, 2), y_true: (500,)

    # 2. Construct precomputed affinity matrices and run spectral clustering.
    affinities = {"rbf_1": lambda X: affinity_rbf(X, gamma=1.0), "rbf_10": lambda X: affinity_rbf(X, gamma=10.0), "rbf_50": lambda X: affinity_rbf(X, gamma=50.0), 
                  "knn_10": lambda X: affinity_knn(X, n_neighbors=10, gamma=10.0), "adaptive_rbf": lambda X: affinity_adaptive_rbf(X, k=7)} # adaptive_rbf: self-tuning RBF.
    results = {}
    for aff_name, aff_fn in affinities.items():
        W = aff_fn(X) # W: (n_samples, n_samples)
        labels = spectral_clustering_sklearn(X, W, n_clusters=len(np.unique(y_true)), random_state=42) # labels: (n_samples,)
        nmi = normalized_mutual_info_score(y_true, labels)
        ari = adjusted_rand_score(y_true, labels)
        results[aff_name] = {"labels": labels, "nmi": nmi, "ari": ari}

    # 3. KMeans baseline
    y_km = KMeans(n_clusters=len(np.unique(y_true)), n_init=10, random_state=42).fit_predict(X) # y_km: (n_samples,)
    km_nmi = normalized_mutual_info_score(y_true, y_km)
    km_ari = adjusted_rand_score(y_true, y_km)

    # 4. Visualization
    n_cols = 2 + len(affinities) # ground truth + kmeans + each affinity
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap="tab10")
    axes[0].set_title("Ground Truth")
    axes[1].scatter(X[:, 0], X[:, 1], c=y_km, s=10, cmap="tab10")
    axes[1].set_title(f"KMeans\nNMI={km_nmi:.3f} ARI={km_ari:.3f}")
    for col, (aff_name, res) in enumerate(results.items(), start=2):
        axes[col].scatter(X[:, 0], X[:, 1], c=res["labels"], s=10, cmap="tab10")
        axes[col].set_title(f"SC ({aff_name})\nNMI={res['nmi']:.3f} ARI={res['ari']:.3f}")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("./result/spectral_clustering_sklearn.png", dpi=150)
