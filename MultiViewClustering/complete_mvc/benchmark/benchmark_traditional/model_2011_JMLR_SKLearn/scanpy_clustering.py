import numpy as np
import torch
import scanpy as sc
import pandas as pd
import random
from anndata import AnnData
from _utils import load_data, evaluate


def _estimate_resolution(adata, target_clusters, method, initial_resolution=0.5, max_iter=20):
    # Estimate the resolution value required to achieve the target number of clusters. (resolution smaller, fewer clusters; resolution larger, more clusters)
    resolution = initial_resolution; step = 0.1
    for _ in range(max_iter):
        if method.lower() == 'leiden':
            sc.tl.leiden(adata, resolution=resolution, key_added='clusters', flavor='igraph', n_iterations=2, directed=False)
        if method.lower() == 'louvain':
            sc.tl.louvain(adata, resolution=resolution, key_added='clusters')
        n_clusters = len(adata.obs['clusters'].unique()); 
        if n_clusters == target_clusters: break; 
        if n_clusters < target_clusters: resolution += step; 
        if n_clusters > target_clusters: resolution -= step; step *= 0.5; 
        if resolution < 0.01: resolution = 0.01; break; 
        if resolution > 3.0: resolution = 3.0; break; 
    return resolution; 


def scanpy_clustering_with_ann_data(X, n_clusters=None, method='leiden', resolution=0.5, n_neighbors=15, n_pcs=50, normalize=True, log_transform=True):
    """
    Use Scanpy to perform clustering (full process)
    Parameters:
        X: numpy array, shape (n_samples, n_features) - input data matrix
        n_clusters: int, optional - expected number of clusters (for automatic resolution adjustment)
        method: str, 'leiden' or 'louvain' - clustering method
        resolution: float - clustering resolution, controlling cluster granularity (larger values result in more clusters)
        n_neighbors: int - number of neighbors when building the neighbor graph
        n_pcs: int - number of principal components
        normalize: bool - whether to normalize the data
        log_transform: bool - whether to perform log transformation
    Returns:
        labels: numpy array - clustering labels
        adata: AnnData object - AnnData object containing the full analysis results
    """
    # 1. Create AnnData object # Scanpy uses AnnData as the data container
    adata = AnnData(X)
    # 2. Preprocessing (optional, but recommended)
    if normalize: # Standardize the total counts per cell to 1e4 (simulating the standardization of single-cell data)
        sc.pp.normalize_total(adata, target_sum=1e4)
    if log_transform: # Log transformation log(x + 1)
        sc.pp.log1p(adata)
    # 3. Feature selection (optional, recommended for high-dimensional data)
    # Select highly variable genes (commonly used in single-cell data)
    # For general data, can skip or use variance filtering
    if X.shape[1] > 2000:  # If the number of features is many, perform feature selection
        sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, X.shape[1]), flavor='seurat')
        adata = adata[:, adata.var.highly_variable]
    # 4. Scale data (optional, but recommended for PCA)
    sc.pp.scale(adata, max_value=10) # scale the data to the range of 0-1
    # 5. PCA dimensionality reduction
    # n_comps must satisfy three constraints:
    # 1. n_pcs: user-specified number of components (default 50)
    # 2. X.shape[1]: cannot exceed the number of features (columns) - PCA can only extract at most p components from p features
    # 3. X.shape[0]-1: cannot exceed (n_samples - 1) - for n samples, the covariance matrix has rank at most min(n-1, p-1)
    #    This is because the mean-centering reduces the rank by 1, so maximum non-zero eigenvalues = min(n-1, p-1)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=min(n_pcs, X.shape[1], X.shape[0]-1))
    # 6. Build neighbor graph (this is a prerequisite for clustering)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.obsm['X_pca'].shape[1]))
    # 7. Perform clustering
    if method.lower() == 'leiden':
        sc.tl.leiden(adata, resolution=resolution, key_added='clusters', flavor='igraph', n_iterations=2, directed=False)
        labels = adata.obs['clusters'].astype(int).values
    elif method.lower() == 'louvain':
        sc.tl.louvain(adata, resolution=resolution, key_added='clusters')
        labels = adata.obs['clusters'].astype(int).values
    # 8. If needed, automatically adjust the resolution to achieve the expected number of clusters
    if n_clusters is not None:
        unique_clusters = len(np.unique(labels))
        if unique_clusters != n_clusters:
            resolution = _estimate_resolution(adata, n_clusters, method, resolution)
            if method.lower() == 'leiden':
                sc.tl.leiden(adata, resolution=resolution, key_added='clusters', flavor='igraph', n_iterations=2, directed=False)
            if method.lower() == 'louvain':
                sc.tl.louvain(adata, resolution=resolution, key_added='clusters')
            labels = adata.obs['clusters'].astype(int).values
    return labels, adata


def scanpy_clustering_simple(X, method='leiden', resolution=0.5, n_neighbors=15, n_pcs=50):
    """
    Simplified version of Scanpy clustering (skipping some preprocessing steps)
    For already preprocessed data, only perform PCA and neighbor graph construction
    """
    adata = AnnData(X)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=min(n_pcs, X.shape[1], X.shape[0]-1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.obsm['X_pca'].shape[1]))
    if method.lower() == 'leiden':
        sc.tl.leiden(adata, resolution=resolution, key_added='clusters', flavor='igraph', n_iterations=2, directed=False)
    if method.lower() == 'louvain':
        sc.tl.louvain(adata, resolution=resolution, key_added='clusters')
    labels = adata.obs['clusters'].astype(int).values
    return labels


if __name__ == "__main__":    
    # Set global random seed for reproducibility
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    sc.settings.verbosity = 1  # Reduce output
    sc.set_figure_params(dpi=80, facecolor='white')
    DATASETS = ["BDGP", "MNIST-USPS"]
    print("=" * 80)
    print("Scanpy clustering algorithm research")
    print("=" * 80)
    print("\nScanpy provides two main clustering algorithms:")
    print("1. Leiden clustering (sc.tl.leiden) - recommended, ensures cluster connectivity")
    print("2. Louvain clustering (sc.tl.louvain) - traditional method, faster")
    print("\nNote: The clustering in Scanpy is based on the graph structure, so the neighbor graph needs to be built first.")
    print("=" * 80)
    results = {}
    for dataset_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        from torch.utils.data import DataLoader
        dataset, dims, view, data_size, class_num = load_data(dataset_name)
        dataloader = DataLoader(dataset, batch_size=data_size, shuffle=False)
        batch = next(iter(dataloader))
        multi_view_data = batch[0]
        labels = batch[1].numpy()
        X_concat = torch.cat(multi_view_data, dim=1).numpy()
        print(f"Dataset shape: {X_concat.shape}")
        print(f"Number of classes: {class_num}, Number of views: {view}")
        dataset_results = {}
        for method in ['leiden', 'louvain']:
            print(f"\nTesting {method.upper()} clustering...")
            test_resolutions = [0.3, 0.5, 0.8]
            method_results = {}
            for resolution in test_resolutions:
                print(f"Using resolution={resolution}...")
                labels_pred = scanpy_clustering_simple(
                    X_concat, 
                    method=method, 
                    resolution=resolution,
                    n_neighbors=min(15, data_size // 10), # adjust the number of neighbors based on the data size
                    n_pcs=min(50, X_concat.shape[1], X_concat.shape[0]-1) # adjust the number of principal components based on the data size
                )
                nmi, ari, acc, pur = evaluate(labels, labels_pred)
                n_clusters = len(np.unique(labels_pred))
                method_results[resolution] = {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters}
                print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, Purity: {pur:.4f}, Number of clusters: {n_clusters}")
            dataset_results[method] = method_results
        results[dataset_name] = dataset_results
    print(f"\n{'='*80}")
    print("Summary Results")
    print(f"{'='*80}")
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        for method, method_results in dataset_results.items():
            print(f"\n  {method.upper()}:")
            for resolution, metrics in method_results.items():
                print(f"    resolution={resolution}: "
                      f"NMI={metrics['NMI']:.4f}, ARI={metrics['ARI']:.4f}, "
                      f"ACC={metrics['ACC']:.4f}, Purity={metrics['Purity']:.4f}, "
                      f"Number of clusters={metrics['n_clusters']}")
