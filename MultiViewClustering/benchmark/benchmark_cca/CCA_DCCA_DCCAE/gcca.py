import random, argparse, numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Standardize the data to have zero mean and unit variance at each feature
from _utils import load_data, evaluate

class GCCA:
    def __init__(self, n_components=1, r=1e-4, eps=1e-8):
        self.n_components = n_components
        self.r = r
        self.eps = eps
        self.U = None  # Mapping matrices for each view
        self.m = None  # Mean vectors for each view

    def fit(self, X_list): # X_list: list of arrays, each X[i].shape: (n_samples, n_features_i)
        """
        Fit GCCA model on multiple views
        X_list: list of numpy arrays, each representing one view
        """
        num_views = len(X_list)
        n_samples = X_list[0].shape[0]
        
        # Calculate means for each view
        self.m = [np.mean(X, axis=0, keepdims=True) for X in X_list] # Each m[i].shape: [1, n_features_i]
        
        # Center the data
        H_list = [X - self.m[i] for i, X in enumerate(X_list)] # Each H[i].shape: [n_samples, n_features_i]
        
        # Compute SVD for each view
        AT_list = []
        for i, H in enumerate(H_list):
            # SVD: H = A @ S @ B^T
            A, S, B = np.linalg.svd(H, full_matrices=False)
            
            # Take top_k components
            top_k = min(self.n_components, min(S.shape[0], A.shape[1]))
            A = A[:, :top_k]  # A.shape: (n_samples, top_k)
            S_thin = S[:top_k]  # S_thin.shape: (top_k,)
            
            # Compute T matrix: T = diag(sqrt(S_thin^2 / (S_thin^2 + eps)))
            S2_inv = 1.0 / (S_thin ** 2 + self.eps)
            T2 = S_thin * S2_inv * S_thin
            T2 = np.maximum(T2, self.eps)  # Ensure T2 >= eps
            T = np.diag(np.sqrt(T2))
            
            # AT = A @ T
            AT = A @ T  # AT.shape: (n_samples, top_k)
            AT_list.append(AT)
        
        # Concatenate all AT matrices
        M_tilde = np.concatenate(AT_list, axis=1)  # M_tilde.shape: (n_samples, sum of top_k for all views)
        
        # Compute QR decomposition of M_tilde
        Q, R = np.linalg.qr(M_tilde)
        
        # Compute SVD of R
        U, lbda, _ = np.linalg.svd(R, full_matrices=False)
        
        # Take top_k components
        top_k = min(self.n_components, U.shape[1])
        G = Q @ U[:, :top_k]  # G.shape: (n_samples, top_k)
        
        # Compute mapping matrices U for each view
        self.U = []
        for i, H in enumerate(H_list):
            # Compute QR decomposition of H
            Q_H, R_H = np.linalg.qr(H)
            # Compute pseudo-inverse: Cjj_inv = (R_H^T @ R_H + eps*I)^(-1)
            Cjj = R_H.T @ R_H + self.eps * np.eye(R_H.shape[1])
            Cjj_inv = np.linalg.inv(Cjj)
            pinv = Cjj_inv @ R_H.T @ Q_H.T
            # U[i] maps view i to shared space
            U_i = pinv @ G  # U[i].shape: (n_features_i, top_k)
            self.U.append(U_i)
        
        return self

    def transform(self, X_list):
        """
        Transform multiple views to shared space
        X_list: list of numpy arrays, each representing one view
        Returns: list of transformed arrays, each Z[i].shape: (n_samples, n_components)
        """
        results = []
        for i, X in enumerate(X_list):
            # Center the data
            H = X - self.m[i]
            # Transform to shared space
            Z = H @ self.U[i]  # Z.shape: (n_samples, n_components)
            results.append(Z)
        return results

def calc_corr_multi(Z_list): # Calculate the correlation between multiple views
    """
    Calculate average pairwise correlation between all views
    Z_list: list of arrays, each Z[i].shape: (n_samples, n_features)
    """
    num_views = len(Z_list)
    if num_views < 2:
        return []
    
    corrs_all = []
    for i in range(num_views):
        for j in range(i+1, num_views):
            Z1, Z2 = Z_list[i], Z_list[j]
            # Calculate correlation for each feature dimension
            corrs = [np.corrcoef(Z1[:, f], Z2[:, f])[0, 1] for f in range(Z1.shape[1])]
            corrs_all.extend(corrs)
    
    return corrs_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCCA')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--seed', default=2, type=int)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## 1. Set seed for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)))
    
    # Convert to numpy and standardize
    X_list = [x_list[v].numpy() for v in range(view)]
    X_list = [StandardScaler().fit_transform(X) for X in X_list]
    y = y.numpy()
    
    # Determine n_components
    min_dim = min([X.shape[1] for X in X_list])
    n_components = min(10, min_dim)
    
    model = GCCA(n_components=n_components)
    model.fit(X_list)
    
    ## 3. Evaluate the model.
    Z_list = model.transform(X_list)
    corrs = calc_corr_multi(Z_list)
    if len(corrs) > 0:
        print(f"\nCorrelation: mean={np.mean(corrs):.4f}, std={np.std(corrs):.4f}")
    
    # Fuse multiple views by averaging
    Z_fused = np.mean(Z_list, axis=0)  # shape: (num_samples, n_components)
    kmeans = KMeans(n_clusters=class_num, random_state=42, n_init=10)
    pred = kmeans.fit_predict(Z_fused)
    nmi, ari, acc, pur = evaluate(y, pred)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, PUR: {pur:.4f}") # print the clustering results
