import torch
import random
import numpy as np
import scanpy as sc
from anndata import AnnData
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix, spdiags, lil_matrix
from scipy.spatial.distance import cdist
from sklearn.cluster import (KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering, Birch)
from sklearn.mixture import GaussianMixture
from _dataset import load_data
from _evaluate import evaluate

# ============================================================================
# Helper Functions for Random Seed Setting
# ============================================================================

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ============================================================================
# Sklearn Clustering Functions
# ============================================================================

def kmeans_clustering(X, y_true, n_clusters, random_state=42, init='k-means++', n_init=10):
    """
    KMeans clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        n_clusters : int. Number of clusters
        random_state : int, default=42. Random seed
        init : str, default='k-means++'. Initialization method
        n_init : int, default=10. Number of initializations
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    set_random_seed(random_state)
    model = KMeans(n_clusters=n_clusters, init=init, random_state=random_state, n_init=n_init)
    y_pred = model.fit_predict(X)
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

def minibatch_kmeans_clustering(X, y_true, n_clusters, random_state=42, n_init=10, batch_size=1024):
    """
    MiniBatchKMeans clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        n_clusters : int. Number of clusters
        random_state : int, default=42. Random seed
        n_init : int, default=10. Number of initializations
        batch_size : int, default=1024. Batch size for mini-batch
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    set_random_seed(random_state)
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, batch_size=batch_size)
    y_pred = model.fit_predict(X)
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

def spectral_clustering(X, y_true, n_clusters, random_state=42, n_neighbors=10, affinity='nearest_neighbors'):
    """
    Spectral Clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        n_clusters : int. Number of clusters
        random_state : int, default=42. Random seed
        n_neighbors : int, default=10. Number of neighbors for affinity matrix
        affinity : str, default='nearest_neighbors'. Affinity matrix type
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    set_random_seed(random_state)
    model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, 
                              n_neighbors=n_neighbors, affinity=affinity)
    y_pred = model.fit_predict(X)
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

def agglomerative_clustering(X, y_true, n_clusters, linkage='ward', affinity='euclidean'):
    """
    Agglomerative Clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        n_clusters : int. Number of clusters
        linkage : str, default='ward'. Linkage criterion
        affinity : str, default='euclidean'. Distance metric
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
    y_pred = model.fit_predict(X)
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

def birch_clustering(X, y_true, n_clusters, threshold=0.5, branching_factor=50):
    """
    Birch Clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        n_clusters : int. Number of clusters
        threshold : float, default=0.5. Threshold for CF node
        branching_factor : int, default=50. Maximum number of CF subclusters
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    model = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor)
    y_pred = model.fit_predict(X)
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

def gaussian_mixture_clustering(X, y_true, n_clusters, random_state=42, covariance_type='full', max_iter=100):
    """
    Gaussian Mixture Model Clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        n_clusters : int. Number of clusters
        random_state : int, default=42. Random seed
        covariance_type : str, default='full'. Type of covariance parameters
        max_iter : int, default=100. Maximum number of iterations
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    set_random_seed(random_state)
    model = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, 
                           random_state=random_state, max_iter=max_iter)
    y_pred = model.fit_predict(X)
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

# ============================================================================
# Scanpy Clustering Functions
# ============================================================================

def scanpy_leiden_clustering(X, y_true, resolution=0.5, n_neighbors=15, n_pcs=50, random_state=42):
    """
    Scanpy Leiden Clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        resolution : float, default=0.5. Resolution parameter for Leiden
        n_neighbors : int, default=15. Number of neighbors for graph construction
        n_pcs : int, default=50. Number of principal components
        random_state : int, default=42. Random seed
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    set_random_seed(random_state)
    adata = AnnData(X)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=min(n_pcs, X.shape[1], X.shape[0]-1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.obsm['X_pca'].shape[1]))
    sc.tl.leiden(adata, resolution=resolution, key_added='clusters', flavor='igraph', n_iterations=2, directed=False)
    y_pred = adata.obs['clusters'].astype(int).values
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

def scanpy_louvain_clustering(X, y_true, resolution=0.5, n_neighbors=15, n_pcs=50, random_state=42):
    """
    Scanpy Louvain Clustering
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        resolution : float, default=0.5. Resolution parameter for Louvain
        n_neighbors : int, default=15. Number of neighbors for graph construction
        n_pcs : int, default=50. Number of principal components
        random_state : int, default=42. Random seed
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    set_random_seed(random_state)
    adata = AnnData(X)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=min(n_pcs, X.shape[1], X.shape[0]-1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.obsm['X_pca'].shape[1]))
    sc.tl.louvain(adata, resolution=resolution, key_added='clusters')
    y_pred = adata.obs['clusters'].astype(int).values
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

# ============================================================================
# LSC Clustering Helper Functions
# ============================================================================

def get_landmarks(X, p, method='Random', random_state=42):
    """ Get landmarks from the data matrix X.
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data matrix
        p : int. Number of landmarks to select
        method : str, optional (default='Random'). Method to select landmarks: 'Random' or 'Kmeans'
        random_state : int, default=42. Random seed
    Returns:
        landmarks : array-like, shape (p, n_features). Selected landmarks
    """
    set_random_seed(random_state)
    if method == 'Random':
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)[:p]
        landmarks = X[indices, :]
        return landmarks
    elif method == 'Kmeans':
        kmeans = KMeans(n_clusters=p, random_state=random_state, n_init=10).fit(X)
        landmarks = kmeans.cluster_centers_
        return landmarks
    else:
        raise ValueError("method can only be 'Kmeans' or 'Random'")

def gaussian_kernel(distance, bandwidth):
    """ Compute Gaussian kernel value. Gaussian kernel: exp(-distance / (2 * bandwidth ** 2))
    Parameters:
        distance : float or array-like. Distance value(s)
        bandwidth : float. Bandwidth parameter for the Gaussian kernel
    Returns:
        kernel_value : float or array-like. Gaussian kernel value(s)
    """
    return np.exp(-distance / (2 * bandwidth ** 2))

def get_linear_coding(X, landmarks, bandwidth, r):
    """ Compute linear coding matrix using landmarks.
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data matrix
        landmarks : array-like, shape (p, n_features). Landmark points
        bandwidth : float. Bandwidth parameter for Gaussian kernel
        r : int. Number of top landmarks to use for each point
    Returns:
        Z_hat : sparse matrix, shape (p, n_samples). Linear coding matrix (p, n_samples)
    """
    # Compute pairwise distances between landmarks and all points
    distances = cdist(landmarks, X, metric='euclidean')
    # Compute similarities using Gaussian kernel
    similarities = gaussian_kernel(distances, bandwidth)
    # Initialize sparse matrix (use lil_matrix for efficient construction)
    p, n_samples = similarities.shape
    Z_hat = lil_matrix((p, n_samples), dtype=np.float64)
    # For each point, select top r landmarks and normalize
    for i in range(n_samples):
        # Get top r landmarks (largest similarities)
        top_indices = np.argsort(similarities[:, i])[-r:][::-1]
        top_coefficients = similarities[top_indices, i]
        # Normalize coefficients
        top_coefficients = top_coefficients / np.sum(top_coefficients)
        # Store in sparse matrix
        Z_hat[top_indices, i] = top_coefficients
    # Convert to csr_matrix for efficient matrix operations
    Z_hat = Z_hat.tocsr()
    # Normalize by row sums: D^(-1/2) * Z_hat
    row_sums = np.array(Z_hat.sum(axis=1)).flatten()
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    D_inv_sqrt = spdiags(1.0 / np.sqrt(row_sums), 0, p, p)
    return D_inv_sqrt @ Z_hat

# ============================================================================
# LSC Clustering Functions
# ============================================================================

def lsc_clustering(X, y_true, n_clusters, n_landmarks, method='Random', 
                   non_zero_landmark_weights=5, bandwidth=0.5, random_state=42):
    """
    Large Scale Spectral Clustering with Landmark-Based Representation
    
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data
        y_true : array-like, shape (n_samples,). True labels
        n_clusters : int. Number of clusters
        n_landmarks : int. Number of landmarks to use
        method : str, default='Random'. Method to select landmarks: 'Random' or 'Kmeans'
        non_zero_landmark_weights : int, default=5. Number of top landmarks to use for each point (r parameter)
        bandwidth : float, default=0.5. Bandwidth parameter for Gaussian kernel
        random_state : int, default=42. Random seed
    
    Returns:
        dict: Evaluation metrics {'NMI': float, 'ARI': float, 'ACC': float, 'Purity': float, 'n_clusters': int}
    """
    set_random_seed(random_state)
    # Get landmarks
    landmarks = get_landmarks(X, n_landmarks, method, random_state)
    # Get linear coding matrix
    Z_hat = get_linear_coding(X, landmarks, bandwidth, non_zero_landmark_weights)
    # Convert sparse matrix to dense for SVD
    Z_hat_dense = Z_hat.T.toarray() if hasattr(Z_hat, 'toarray') else Z_hat.T
    # Perform SVD on Z_hat^T (which is (n_samples, p))
    U, s, Vt = np.linalg.svd(Z_hat_dense, full_matrices=False)
    # Use top n_clusters eigenvectors
    U_top = U[:, :n_clusters]
    # Perform K-means on the embedded space
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(U_top)
    y_pred = kmeans.labels_
    nmi, ari, acc, pur = evaluate(y_true, y_pred)
    n_clusters_pred = len(np.unique(y_pred))
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Purity': pur, 'n_clusters': n_clusters_pred}

if __name__ == "__main__":
    # Set global random seed for reproducibility
    RANDOM_SEED = 42
    set_random_seed(RANDOM_SEED)
    
    sc.settings.verbosity = 1  # Reduce output
    sc.set_figure_params(dpi=80, facecolor='white')
    
    print("=" * 80)
    print("Traditional Clustering Algorithms Benchmark")
    print("=" * 80)
    print("\nThis script evaluates multiple clustering algorithms:")
    print("1. Sklearn algorithms: KMeans, MiniBatchKMeans, SpectralClustering,")
    print("   AgglomerativeClustering, Birch, GaussianMixture")
    print("2. Scanpy algorithms: Leiden, Louvain")
    print("3. LSC (Large Scale Spectral Clustering)")
    print("=" * 80)
    DATASETS = ["BDGP", "MNIST-USPS", "CCV", "Fashion", "Caltech-2V", "Caltech-3V", "Caltech-4V", "Caltech-5V"] 
    print(f"Processing datasets: {DATASETS}")
    results = {}
    for dataset_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Load dataset
        dataset, dims, view, data_size, class_num = load_data(dataset_name)
        dataloader = DataLoader(dataset, batch_size=data_size, shuffle=False)
        batch = next(iter(dataloader))
        multi_view_data = batch[0]  # Multi-view data list
        labels = batch[1].numpy()  # True labels
        X_concat = torch.cat(multi_view_data, dim=1).numpy()  # shape: (data_size, sum(dims))
        
        print(f"Dataset shape: {X_concat.shape}")
        print(f"Number of classes: {class_num}, Number of views: {view}")
        
        dataset_results = {}
        
        # ========================================================================
        # 1. Run Sklearn clustering algorithms
        # ========================================================================
        print(f"\n--- Running Sklearn Clustering Algorithms ---")
        
        print(f"\n  Running KMeans...")
        result = kmeans_clustering(X_concat, labels, class_num, random_state=RANDOM_SEED)
        dataset_results['KMeans'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        print(f"\n  Running MiniBatchKMeans...")
        result = minibatch_kmeans_clustering(X_concat, labels, class_num, random_state=RANDOM_SEED)
        dataset_results['MiniBatchKMeans'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        print(f"\n  Running SpectralClustering...")
        result = spectral_clustering(X_concat, labels, class_num, random_state=RANDOM_SEED)
        dataset_results['SpectralClustering'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        print(f"\n  Running AgglomerativeClustering...")
        result = agglomerative_clustering(X_concat, labels, class_num)
        dataset_results['AgglomerativeClustering'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        print(f"\n  Running Birch...")
        result = birch_clustering(X_concat, labels, class_num)
        dataset_results['Birch'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        print(f"\n  Running GaussianMixture...")
        result = gaussian_mixture_clustering(X_concat, labels, class_num, random_state=RANDOM_SEED)
        dataset_results['GaussianMixture'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        # ========================================================================
        # 2. Run Scanpy clustering algorithms
        # ========================================================================
        print(f"\n--- Running Scanpy Clustering Algorithms ---")
        
        # Leiden with multiple resolutions
        print(f"\n  Testing Leiden clustering...")
        leiden_results = {}
        for resolution in [0.3, 0.5, 0.8]:
            print(f"    Using resolution={resolution}...")
            result = scanpy_leiden_clustering(
                X_concat, labels, resolution=resolution,
                n_neighbors=min(15, max(5, data_size // 10)),
                n_pcs=min(50, X_concat.shape[1], X_concat.shape[0]-1),
                random_state=RANDOM_SEED
            )
            leiden_results[resolution] = result
            print(f"      NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
                  f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
                  f"Number of clusters: {result['n_clusters']}")
        dataset_results['Scanpy_Leiden'] = leiden_results
        
        # Louvain with multiple resolutions
        print(f"\n  Testing Louvain clustering...")
        louvain_results = {}
        for resolution in [0.3, 0.5, 0.8]:
            print(f"    Using resolution={resolution}...")
            result = scanpy_louvain_clustering(
                X_concat, labels, resolution=resolution,
                n_neighbors=min(15, max(5, data_size // 10)),
                n_pcs=min(50, X_concat.shape[1], X_concat.shape[0]-1),
                random_state=RANDOM_SEED
            )
            louvain_results[resolution] = result
            print(f"      NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
                  f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
                  f"Number of clusters: {result['n_clusters']}")
        dataset_results['Scanpy_Louvain'] = louvain_results
        
        # ========================================================================
        # 3. Run LSC clustering
        # ========================================================================
        print(f"\n--- Running LSC Clustering ---")
        n_landmarks = min(max(class_num * 10, 50), data_size // 2, 500)  # Adaptive number of landmarks
        
        print(f"\n  Running LSC with Random landmarks (n_landmarks={n_landmarks})...")
        result = lsc_clustering(
            X_concat, labels, n_clusters=class_num,
            n_landmarks=n_landmarks, method='Random',
            non_zero_landmark_weights=5, bandwidth=0.5,
            random_state=RANDOM_SEED
        )
        dataset_results['LSC_Random'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        print(f"\n  Running LSC with Kmeans landmarks (n_landmarks={n_landmarks})...")
        result = lsc_clustering(
            X_concat, labels, n_clusters=class_num,
            n_landmarks=n_landmarks, method='Kmeans',
            non_zero_landmark_weights=5, bandwidth=0.5,
            random_state=RANDOM_SEED
        )
        dataset_results['LSC_Kmeans'] = result
        print(f"    NMI: {result['NMI']:.4f}, ARI: {result['ARI']:.4f}, "
              f"ACC: {result['ACC']:.4f}, Purity: {result['Purity']:.4f}, "
              f"Number of clusters: {result['n_clusters']}")
        
        results[dataset_name] = dataset_results
    
    # ========================================================================
    # Print Summary Results
    # ========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        for method_name, metrics in dataset_results.items():
            if isinstance(metrics, dict) and any(isinstance(v, dict) for v in metrics.values()):
                # Handle Scanpy results with multiple resolutions
                print(f"  {method_name}:")
                for param, param_metrics in metrics.items():
                    print(f"    {param}: NMI={param_metrics['NMI']:.4f}, "
                          f"ARI={param_metrics['ARI']:.4f}, "
                          f"ACC={param_metrics['ACC']:.4f}, "
                          f"Purity={param_metrics['Purity']:.4f}, "
                          f"n_clusters={param_metrics['n_clusters']}")
            else:
                # Handle single result metrics
                print(f"  {method_name}: NMI={metrics['NMI']:.4f}, "
                      f"ARI={metrics['ARI']:.4f}, "
                      f"ACC={metrics['ACC']:.4f}, "
                      f"Purity={metrics['Purity']:.4f}, "
                      f"n_clusters={metrics['n_clusters']}")
