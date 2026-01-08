import random, argparse, numpy as np
import torch
import sys
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Standardize the data to have zero mean and unit variance at each feature
from itertools import combinations
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class CCA:
    def __init__(self, n_components=1, r1=1e-4, r2=1e-4):
        self.n_components = n_components
        self.r1 = r1 # regularization parameter for view 1
        self.r2 = r2 # regularization parameter for view 2
        self.w = [None, None] # mapping matrices for two views
        self.m = [None, None] # mean vectors for two views

    def fit(self, X1, X2): # X1.shape: (n_samples, n_features_1); X2.shape: (n_samples, n_features_2)
        # Get the number of samples and features of two views
        n_samples = X1.shape[0]; n_features_1 = X1.shape[1]; n_features_2 = X2.shape[1]
        # Calculate the mean of two views
        self.m[0] = np.mean(X1, axis=0, keepdims=True) # [1, n_features_1]
        self.m[1] = np.mean(X2, axis=0, keepdims=True) # [1, n_features_2]
        # Calculate the centered data
        H1bar = X1 - self.m[0] # [n_samples, n_features_1]
        H2bar = X2 - self.m[1] # [n_samples, n_features_2]
        # Calculate the covariance matrices
        SigmaHat12 = (1.0 / (n_samples - 1)) * np.dot(H1bar.T, H2bar) # [n_features_1, n_features_2]
        SigmaHat11 = (1.0 / (n_samples - 1)) * np.dot(H1bar.T, H1bar) + self.r1 * np.identity(n_features_1) # [n_features_1, n_features_1]
        SigmaHat22 = (1.0 / (n_samples - 1)) * np.dot(H2bar.T, H2bar) + self.r2 * np.identity(n_features_2) # [n_features_2, n_features_2]
        # Calculate the eigenvalues and eigenvectors of the covariance matrices
        [D1, V1] = np.linalg.eigh(SigmaHat11) # D1.shape: (n_features_1,); V1.shape: (n_features_1, n_features_1)
        [D2, V2] = np.linalg.eigh(SigmaHat22) # D2.shape: (n_features_2,); V2.shape: (n_features_2, n_features_2)
        SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T) # [n_features_1, n_features_1]
        SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T) # [n_features_2, n_features_2]
        Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv) # [n_features_1, n_features_2]
        # Calculate the singular values and eigenvectors of the covariance matrices by using SVD. SVD: Tval = U @ D @ V^T. 
        # Attention: SVD will sort the singular values in descending order, so we need to take the first n_components singular values and eigenvectors
        [U, D, V] = np.linalg.svd(Tval) # U.shape: (n_features_1, n_features_1); D.shape: (n_features_1,); V.shape: (n_features_2, n_features_2)
        V = V.T # V.shape: (n_features_2, n_features_2)
        self.w[0] = np.dot(SigmaHat11RootInv, U[:, 0:self.n_components]) # [n_features_1, n_components]
        self.w[1] = np.dot(SigmaHat22RootInv, V[:, 0:self.n_components]) # [n_features_2, n_components]
        D = D[0:self.n_components] # D.shape: (n_components,)

    def transform(self, X1, X2):
        result_1 = X1 - self.m[0].reshape([1, -1]).repeat(len(X1), axis=0) # [n_samples, n_features]
        result_1 = np.dot(result_1, self.w[0]) # [n_samples, n_components]
        result_2 = X2 - self.m[1].reshape([1, -1]).repeat(len(X2), axis=0) # [n_samples, n_features]
        result_2 = np.dot(result_2, self.w[1]) # [n_samples, n_components]
        return result_1, result_2 # result_1.shape: (num_samples, n_components); result_2.shape: (num_samples, n_components)

class GCCA:
    def __init__(self, n_components=1, r=1e-4, eps=1e-8):
        self.n_components = n_components
        self.r = r # regularization parameter
        self.eps = eps # epsilon for stability
        self.U = None  # Mapping matrices for each view; not used in this implementation
        self.m = None  # Mean vectors for each view

    def fit(self, X_list): # X_list: list of arrays, each X[i].shape: (n_samples, n_features_i)
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
        U, _, _ = np.linalg.svd(R, full_matrices=False)
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

    def transform(self, X_list):
        results = []
        for i, X in enumerate(X_list):
            H = X - self.m[i] # Center the data
            Z = H @ self.U[i] # Transform to shared space; Z.shape: (n_samples, n_components)
            results.append(Z)
        return results

def calc_corr_multi(Z_list): # Calculate the correlation between multiple views
    corrs_all = []
    for Z1, Z2 in combinations(Z_list, 2): # Z1.shape: (num_samples, num_features); Z2.shape: (num_samples, num_features)
        # np.corrcoef(z1, z2) = [[corr(z1, z1), corr(z1, z2)], [corr(z2, z1), corr(z2, z2)]]
        corrs = [np.corrcoef(z1, z2)[0, 1] for z1, z2 in zip(Z1.T, Z2.T)] # [0, 1] means only cross-cov
        corrs_all.append(corrs)
    return corrs_all # corrs_all.shape: (combinations(num_views, 2), num_features)

def benchmark_1992_BOOK_CCA(dataset_name="BDGP", 
                            n_components=10, 
                            regularization=1e-4, 
                            seed=42, 
                            verbose=False):
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    assert view > 1, "CCA(GCCA) only supports two views or more, but got {view} views"
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)))
    x_list = [x.numpy() for x in x_list]; y = y.numpy()
    if view == 2:
        x_list = [StandardScaler().fit_transform(x) for x in x_list]
        model = CCA(n_components=n_components, r1=regularization, r2=regularization)
        model.fit(x_list[0], x_list[1])
        Z1, Z2 = model.transform(x_list[0], x_list[1])
        corrs_all = calc_corr_multi([Z1, Z2]) # corrs_all.shape: (num_features,)
        Z_fused = (Z1 + Z2) / 2 # shape: (num_samples, n_components)
    else:
        x_list = [StandardScaler().fit_transform(x) for x in x_list]
        model = GCCA(n_components=n_components, r=regularization, eps=1e-8)
        model.fit(x_list)
        Z_list = model.transform(x_list)
        corrs_all = calc_corr_multi(Z_list) # corrs_all.shape: (num_features,)
        Z_fused = np.mean(Z_list, axis=0) # shape: (num_samples, n_components)

    ## 3. Evaluate the model.
    print(f"Correlation: mean={np.mean(corrs_all):.4f}, std={np.std(corrs_all):.4f}") if verbose else None # print the correlation results
    kmeans = KMeans(n_clusters=class_num, random_state=seed, n_init=10)
    pred = kmeans.fit_predict(Z_fused)
    nmi, ari, acc, pur = evaluate(y, pred)
    print('Clustering results: ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCCA')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--n_components', default=10, type=int)
    parser.add_argument('--regularization', default=1e-4, type=float)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_1992_BOOK_CCA(
        dataset_name=args.dataset,
        n_components=args.n_components,
        regularization=args.regularization,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
    