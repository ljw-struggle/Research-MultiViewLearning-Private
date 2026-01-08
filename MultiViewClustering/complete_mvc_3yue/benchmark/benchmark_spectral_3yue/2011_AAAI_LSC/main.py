import os, numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, spdiags, lil_matrix
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score

########################################################################################
### NMI Metric Implementation
########################################################################################

def entropy(partition_set, N):
    """ Calculate entropy of a partition (n_clusters, n_samples).
    Parameters:
        partition_set : list of lists. Each inner list contains indices of points in that cluster (n_clusters, n_samples)
        N : int. Total number of points
    Returns:
        entropy_value : float. Entropy of the partition
    """
    entropy_value = 0.0
    for cluster in partition_set:
        if len(cluster) > 0:
            p = len(cluster) / N
            entropy_value -= p * np.log(p)
    return entropy_value

def mutual_information(partition_set_a, partition_set_b, N):
    """ Calculate mutual information between two partitions (n_clusters, n_samples).
    Parameters:
        partition_set_a : list of lists. First partition
        partition_set_b : list of lists. Second partition
        N : int. Total number of points
    Returns:
        mi_value : float. Mutual information value
    """
    mi_value = 0.0
    for i, cluster_a in enumerate(partition_set_a):
        for j, cluster_b in enumerate(partition_set_b):
            # Calculate intersection
            intersection = len(set(cluster_a) & set(cluster_b))
            if intersection > 0:
                p_ab = intersection / N
                p_a = len(cluster_a) / N
                p_b = len(cluster_b) / N
                if p_a > 0 and p_b > 0:
                    mi_value += p_ab * np.log((intersection * N) / (len(cluster_a) * len(cluster_b)))
    return mi_value

def normalized_mutual_information(partition_set_a, partition_set_b, N):
    """ Calculate normalized mutual information (NMI) between two partitions (n_clusters, n_samples).
    Parameters:
        partition_set_a : list of lists. First partition    
        partition_set_b : list of lists. Second partition
        N : int. Total number of points
    Returns:
        nmi_value : float. Normalized mutual information value (between 0 and 1)
    """
    mi = mutual_information(partition_set_a, partition_set_b, N)
    entropy_a = entropy(partition_set_a, N)
    entropy_b = entropy(partition_set_b, N)
    # Avoid division by zero
    if entropy_a + entropy_b == 0:
        return 1.0 if mi > 0 else 0.0
    nmi_value = mi / ((entropy_a + entropy_b) / 2)
    return nmi_value

def get_partition_set(vector, n_clusters):
    """ Convert cluster assignment vector to partition set representation (n_clusters, n_samples).
    Parameters:
        vector : array-like, shape (n_samples,). Cluster assignment vector (0-indexed or 1-indexed)
        n_clusters : int. Number of clusters
    Returns:
        partition_set : list of lists. Each inner list contains indices of points in that cluster (n_clusters, n_samples)
    """
    # Convert to 0-indexed if needed
    vector = np.array(vector)
    if vector.min() > 0:
        vector = vector - 1
    partition_set = [[] for _ in range(n_clusters)]
    for i, cluster_id in enumerate(vector):
        if 0 <= cluster_id < n_clusters:
            partition_set[int(cluster_id)].append(i)
    return partition_set

# labels = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
# predicted_labels = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0])
# # Evaluate using Normalized Mutual Information
# predicted_partition = get_partition_set(predicted_labels, n_clusters=2) # shape: (10, n_samples)
# true_partition = get_partition_set(labels, n_clusters=2)  # shape: (10, n_samples)
# nmi_value = normalized_mutual_information(predicted_partition, true_partition, len(labels))
# print(f"Normalized Mutual Information: {nmi_value:.4f}")
# nmi_value = normalized_mutual_info_score(labels, predicted_labels)
# print(f"Normalized Mutual Information: {nmi_value:.4f}")

########################################################################################
### LSC Algorithm Implementation 
########################################################################################

def get_landmarks(X, p, method='Random'):
    """ Get landmarks from the data matrix X.
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data matrix
        p : int. Number of landmarks to select
        method : str, optional (default='Random'). Method to select landmarks: 'Random' or 'Kmeans'
    Returns:
        landmarks : array-like, shape (p, n_features). Selected landmarks
    """
    if method == 'Random':
        n_samples = X.shape[0] # n_samples: number of samples in X
        indices = np.random.permutation(n_samples)[:p] # indices: (p,)
        landmarks = X[indices, :] # landmarks: (p, n_features) from X
        return landmarks # landmarks: (p, n_features)
    elif method == 'Kmeans':
        kmeans = KMeans(n_clusters=p, random_state=42, n_init=10).fit(X) # kmeans: KMeans object
        landmarks = kmeans.cluster_centers_ # landmarks: (p, n_features)
        return landmarks # landmarks: (p, n_features)
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
    return np.exp(-distance / (2 * bandwidth ** 2)) # kernel_value: (p, n_samples)

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
    # landmarks: (p, n_features), X: (n_samples, n_features)
    # Result: (p, n_samples)
    distances = cdist(landmarks, X, metric='euclidean') # distances: (p, n_samples)
    # Compute similarities using Gaussian kernel
    similarities = gaussian_kernel(distances, bandwidth) # similarities: (p, n_samples)
    # Initialize sparse matrix (use lil_matrix for efficient construction)
    p, n_samples = similarities.shape # p: number of landmarks, n_samples: number of samples in X
    Z_hat = lil_matrix((p, n_samples), dtype=np.float64) # Z_hat: (p, n_samples)
    # For each point, select top r landmarks and normalize
    for i in range(n_samples):
        # Get top r landmarks (largest similarities)
        top_indices = np.argsort(similarities[:, i])[-r:][::-1] # top_indices: (r,)
        top_coefficients = similarities[top_indices, i] # top_coefficients: (r,)
        # Normalize coefficients
        top_coefficients = top_coefficients / np.sum(top_coefficients) # top_coefficients: (r,)
        # Store in sparse matrix
        Z_hat[top_indices, i] = top_coefficients # Z_hat: (p, n_samples)
    # Convert to csr_matrix for efficient matrix operations
    Z_hat = Z_hat.tocsr() # Z_hat: (p, n_samples)
    # Normalize by row sums: D^(-1/2) * Z_hat
    row_sums = np.array(Z_hat.sum(axis=1)).flatten() # row_sums: (p,)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1 # row_sums: (p,)
    D_inv_sqrt = spdiags(1.0 / np.sqrt(row_sums), 0, p, p) # D_inv_sqrt: (p, p)
    return D_inv_sqrt @ Z_hat # D_inv_sqrt @ Z_hat: (p, n_samples)

def lsc_clustering(X, n_clusters, n_landmarks, method='Random', non_zero_landmark_weights=5, bandwidth=0.5):
    """ Perform Large Scale Spectral Clustering with Landmark-Based Representation.
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data matrix
        n_clusters : int. Number of clusters
        n_landmarks : int. Number of landmarks to use
        method : str, optional (default='Random'). Method to select landmarks: 'Random' or 'Kmeans'
        non_zero_landmark_weights : int, optional (default=5). Number of top landmarks to use for each point (r parameter)
        bandwidth : float, optional (default=0.5). Bandwidth parameter for Gaussian kernel
    Returns:
        clustering_result : KMeans object. KMeans clustering result with assignments in .labels_ attribute (n_samples,), 0-indexed
    """
    # Get landmarks
    landmarks = get_landmarks(X, n_landmarks, method) # landmarks: (p, n_features)
    # Get linear coding matrix
    Z_hat = get_linear_coding(X, landmarks, bandwidth, non_zero_landmark_weights) # Z_hat: (p, n_samples)
    # Convert sparse matrix to dense for SVD (or use scipy.sparse.linalg.svds for sparse)
    # For large matrices, consider using scipy.sparse.linalg.svds instead
    # Z_hat is (p, n_samples), we need (n_samples, p) for SVD
    Z_hat_dense = Z_hat.T.toarray() if hasattr(Z_hat, 'toarray') else Z_hat.T # Z_hat_dense: (n_samples, p)
    # Perform SVD on Z_hat^T (which is (n_samples, p))
    U, s, Vt = np.linalg.svd(Z_hat_dense, full_matrices=False) # U: (n_samples, n_samples), s: (n_samples,), Vt: (n_samples, p)
    # Use top n_clusters eigenvectors
    U_top = U[:, :n_clusters] # U_top: (n_samples, n_clusters)
    # Perform K-means on the embedded space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(U_top) # kmeans: KMeans object with assignments in .labels_ attribute (n_samples,), 0-indexed
    return kmeans # kmeans: KMeans object

def elbow_method_to_find_optimal_n_landmarks(X, start=1, end=10):
    """ Find the optimal number of landmarks using the elbow method. Make an elbow graph to determine the optimal number of landmarks. The number of landmarks should be the first one with a dramatic drop from ~10^2 to ~10^1.
        - Function: Use Elbow Method to determine the optimal number of landmarks
        - How it works: 1. Run KMeans for different numbers of clusters k (from start to end); 2. Calculate the inertia (within-cluster sum of squares) for each k; 3. Plot the curve of inertia vs k; 4. Find the "elbow" point where the rate of decrease in inertia slows down;
        - Meaning of Inertia: 1. Inertia = Sum of squared distances from all points to their cluster centers; 2. Lower inertia means tighter clusters; 3. Inertia typically decreases as k increases;
        - How to choose optimal k: 1. Find the point where the rate of decrease in inertia suddenly slows (elbow point); 2. Usually after this point, increasing k does not significantly improve clustering
    Parameters:
        X : array-like, shape (n_samples, n_features). Input data matrix
        start : int, optional (default=1). Start number of landmarks
        end : int, optional (default=10). End number of landmarks
    """
    cluster_numbers=[]
    inertia=[]
    for i in range(start, end+1):
        cluster_numbers.append(i)   
        inertia.append(KMeans(n_clusters=i).fit(X).inertia_)
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_numbers, inertia)
    plt.xticks(np.arange(start, end+1))
    plt.xlabel('Number of landmarks'); plt.ylabel('Inertia'); plt.title('The elbow method')
    plt.savefig('elbow_graph_to_find_optimal_n_landmarks.png')
    

if __name__ == "__main__":
    os.makedirs('result', exist_ok=True)
    X, y = make_blobs(n_samples=1000, n_features=100, centers=10, random_state=42) # X: (n_samples, n_features), y: (n_samples,), 0-indexed
    
    ## 1. Normalize the data.
    X = zscore(X, axis=0, ddof=0)  # biased estimation (n), z-score standardization along axis=0 (per feature)
    # X = StandardScaler().fit_transform(X)  # equivalent to zscore along axis=0
    # X = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X - np.mean(X, axis=0, keepdims=True), axis=0, keepdims=True)  # equivalent to zscore along axis=0
    
    ## 2. Find the optimal number of landmarks
    # elbow_method_to_find_optimal_n_landmarks(X)

    # 3. Perform LSC clustering
    lsc_result = lsc_clustering(X, n_clusters=10, n_landmarks=350, method='Kmeans', non_zero_landmark_weights=5, bandwidth=0.5)
    nmi_value = normalized_mutual_info_score(y, lsc_result.labels_)
    print(f"Normalized Mutual Information: {nmi_value:.4f}")
    
    # # 4. Plot the clustering result
    # plt.figure(figsize=(10, 5))
    # plt.scatter(X[:, 0], X[:, 1], c=lsc_result.labels_, cmap='viridis')
    # plt.colorbar()
    # plt.savefig('clustering_result.png')
    