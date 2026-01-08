import torch, random, numpy as np
from scipy.sparse import csr_matrix, spdiags, lil_matrix
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from ..dataset import load_data
from ..metric import evaluate
    
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

def lsc_clustering(X, y_true, n_clusters, n_landmarks, method='Random', non_zero_landmark_weights=5, bandwidth=0.5, random_state=42):
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
    return nmi, ari, acc, pur

def benchmark_2011_AAAI_LSC(dataset_name='BDGP', use_view=-1, method='Kmeans', non_zero_landmark_weights=5, bandwidth=0.5, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
        
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    n_landmarks = min(max(class_num * 10, 50), data_size // 2, 500)  # Adaptive number of landmarks
    nmi, ari, acc, pur = lsc_clustering(data, label, class_num, n_landmarks, method, non_zero_landmark_weights, bandwidth, random_state)
    return nmi, ari, acc, pur

