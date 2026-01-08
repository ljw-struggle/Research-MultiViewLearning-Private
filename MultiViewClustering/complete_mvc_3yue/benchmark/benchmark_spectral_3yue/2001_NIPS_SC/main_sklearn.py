import os, sys, argparse, random, numpy as np
import torch
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, polynomial_kernel, linear_kernel
from sklearn.neighbors import kneighbors_graph

try:
    from ...dataset import load_data
    from ...metric import evaluate
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

KERNEL_REGISTRY = {}

def register_kernel(name):
    def decorator(fn):
        KERNEL_REGISTRY[name] = fn
        return fn
    return decorator

@register_kernel("rbf")
def kernel_rbf(X, gamma=None):
    """Gaussian RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)"""
    return rbf_kernel(X, gamma=gamma)

@register_kernel("cosine")
def kernel_cosine(X, **kwargs):
    """Cosine similarity kernel: K(x,y) = x^T y / (||x|| ||y||), shifted to [0,1]"""
    S = cosine_similarity(X)
    return (S + 1.0) / 2.0

@register_kernel("polynomial")
def kernel_polynomial(X, degree=3, gamma=None, coef0=1):
    """Polynomial kernel: K(x,y) = (gamma * x^T y + coef0)^degree"""
    return polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)

@register_kernel("linear")
def kernel_linear(X, **kwargs):
    """Linear kernel: K(x,y) = x^T y"""
    K = linear_kernel(X)
    K_min = K.min()
    K_max = K.max()
    if K_max - K_min > 0:
        K = (K - K_min) / (K_max - K_min)
    return K

@register_kernel("knn")
def kernel_knn(X, n_neighbors=10, **kwargs):
    """KNN-based affinity: symmetric k-nearest-neighbor graph with RBF weights"""
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=False)
    A = 0.5 * (A + A.T)  # make symmetric
    return A.toarray()

@register_kernel("adaptive_rbf")
def kernel_adaptive_rbf(X, k=7, **kwargs):
    """Adaptive (self-tuning) RBF kernel: sigma_i = dist(x_i, x_i^k)"""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    sigma = distances[:, k]
    sigma = np.maximum(sigma, 1e-10)
    dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    denom = np.outer(sigma, sigma)
    K = np.exp(-dist_sq / denom)
    return K


def compute_affinity(X, kernel_name, **kernel_kwargs):
    if kernel_name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel '{kernel_name}'. Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[kernel_name](X, **kernel_kwargs)


def benchmark_2001_NIPS_SC_sklearn(dataset_name='BDGP',
                                    kernel='rbf',
                                    gamma=None,
                                    degree=3,
                                    coef0=1,
                                    n_neighbors=10,
                                    k_adaptive=7,
                                    seed=42,
                                    verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    data = torch.cat(data, dim=1).cpu().numpy()
    label = label.cpu().numpy()

    ## 3. Compute precomputed affinity matrix.
    kernel_kwargs = dict(gamma=gamma, degree=degree, coef0=coef0,
                         n_neighbors=n_neighbors, k=k_adaptive)
    affinity_matrix = compute_affinity(data, kernel, **kernel_kwargs)
    if verbose:
        print(f"Kernel: {kernel} | Affinity shape: {affinity_matrix.shape} | "
              f"min={affinity_matrix.min():.4f}, max={affinity_matrix.max():.4f}")

    ## 4. Run Spectral Clustering with precomputed affinity.
    model = SpectralClustering(n_clusters=class_num, affinity='precomputed', random_state=seed)
    y_pred = model.fit_predict(affinity_matrix)

    ## 5. Evaluate.
    nmi, ari, acc, pur = evaluate(label, y_pred)
    if verbose:
        print(f"[{kernel}] ACC={acc:.4f}; NMI={nmi:.4f}; ARI={ari:.4f}; PUR={pur:.4f}")
    return nmi, ari, acc, pur


def run_all_kernels(dataset_name='BDGP', seed=42, verbose=False, **kwargs):
    """Run spectral clustering with every registered kernel and return a results dict."""
    results = {}
    for kernel_name in KERNEL_REGISTRY:
        if verbose:
            print(f"\n{'='*50}\nRunning kernel: {kernel_name}\n{'='*50}")
        nmi, ari, acc, pur = benchmark_2001_NIPS_SC_sklearn(
            dataset_name=dataset_name, kernel=kernel_name, seed=seed, verbose=verbose, **kwargs)
        results[kernel_name] = dict(nmi=nmi, ari=ari, acc=acc, pur=pur)
        print(f"[{kernel_name:>12s}] ACC={acc:.4f}; NMI={nmi:.4f}; ARI={ari:.4f}; PUR={pur:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Clustering (sklearn, precomputed kernels)")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--kernel", default="rbf", type=str,
                        choices=list(KERNEL_REGISTRY.keys()) + ["all"],
                        help="Kernel for precomputed affinity ('all' to run every kernel)")
    parser.add_argument("--gamma", default=None, type=float, help="Gamma for RBF / polynomial kernel (None=auto)")
    parser.add_argument("--degree", default=3, type=int, help="Degree for polynomial kernel")
    parser.add_argument("--coef0", default=1, type=float, help="Independent term for polynomial kernel")
    parser.add_argument("--n_neighbors", default=10, type=int, help="Number of neighbors for KNN kernel")
    parser.add_argument("--k_adaptive", default=7, type=int, help="k-th neighbor for adaptive RBF sigma")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.kernel == "all":
        run_all_kernels(dataset_name=args.dataset, seed=args.seed, verbose=args.verbose,
                        gamma=args.gamma, degree=args.degree, coef0=args.coef0,
                        n_neighbors=args.n_neighbors, k_adaptive=args.k_adaptive)
    else:
        nmi, ari, acc, pur = benchmark_2001_NIPS_SC_sklearn(
            dataset_name=args.dataset, kernel=args.kernel,
            gamma=args.gamma, degree=args.degree, coef0=args.coef0,
            n_neighbors=args.n_neighbors, k_adaptive=args.k_adaptive,
            seed=args.seed, verbose=args.verbose)
        print(f"NMI={nmi:.4f}; ARI={ari:.4f}; ACC={acc:.4f}; PUR={pur:.4f}")
