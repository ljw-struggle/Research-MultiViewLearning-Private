import random, argparse, numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA as CCA_Sklearn
from sklearn.preprocessing import StandardScaler # Standardize the data to have zero mean and unit variance at each feature
from _utils import load_data, evaluate

class CCA_Scratch:
    def __init__(self, n_components=1, r1=1e-4, r2=1e-4):
        self.n_components = n_components
        self.r1 = r1
        self.r2 = r2
        self.w = [None, None]
        self.m = [None, None]

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

def calc_corr(Z1, Z2): # Calculate the correlation of each feature between two views
    N, F = Z1.shape # Z1.shape: (num_samples, num_features); Z2.shape: (num_samples, num_features)
    # np.corrcoef(z1, z2) = [[corr(z1, z1), corr(z1, z2)], [corr(z2, z1), corr(z2, z2)]]
    corrs = [np.corrcoef(z1, z2)[0, 1] for z1, z2 in zip(Z1.T, Z2.T)] # only cross-cov
    return corrs # corrs.shape: (num_features,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CCA')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--mode', default='scratch', type=str, choices=['sklearn', 'scratch'])
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
    X1 = x_list[0].numpy(); X2 = x_list[1].numpy(); y = y.numpy()
    X1 = StandardScaler().fit_transform(X1)
    X2 = StandardScaler().fit_transform(X2)
    if args.mode == 'sklearn':
        model = CCA_Sklearn(n_components=min(10, min(X1.shape[1], X2.shape[1])), scale=False)
        Z1, Z2 = model.fit_transform(X1, X2) # Z1.shape: (num_samples, n_components); Z2.shape: (num_samples, n_components)
    elif args.mode == 'scratch':
        model = CCA_Scratch(n_components=min(10, min(X1.shape[1], X2.shape[1])))
        model.fit(X1, X2) # CCA components cannot exceed the minimum dimension of two views
        Z1, Z2 = model.transform(X1, X2) # Z1.shape: (num_samples, n_components); Z2.shape: (num_samples, n_components)
    
    ## 3. Evaluate the model.
    corrs = calc_corr(Z1, Z2)
    print(f"\nCorrelation: mean={np.mean(corrs):.4f}, std={np.std(corrs):.4f}")
    Z_fused = (Z1 + Z2) / 2 # shape: (num_samples, n_components)
    kmeans = KMeans(n_clusters=class_num, random_state=42, n_init=10)
    pred = kmeans.fit_predict(Z_fused)
    nmi, ari, acc, pur = evaluate(y, pred)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, PUR: {pur:.4f}") # print the clustering results
