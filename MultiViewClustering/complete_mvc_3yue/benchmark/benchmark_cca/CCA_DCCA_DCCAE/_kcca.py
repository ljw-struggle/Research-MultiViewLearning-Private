import random, argparse, numpy as np
import torch
from scipy.linalg import eigh
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from cca_zoo.nonparametric import KCCA as KCCA_Zoo
from _utils import load_data, evaluate

# Fix for cca_zoo compatibility issue: add missing _get_tags method
def _get_tags(self):
    return {'multiview': False}

# Add _get_tags method to KCCA class
KCCA_Zoo._get_tags = _get_tags

def combined_kernel(X, Y, weight_linear=0.5, weight_polynomial=0.3, weight_rbf=0.2):
    """
    Combined kernel: weighted sum of linear, polynomial, and RBF kernels.
    """
    from sklearn.metrics.pairwise import linear_kernel as lin_kernel_func, rbf_kernel as rbf_kernel_func, polynomial_kernel as poly_kernel_func
    lin_k = lin_kernel_func(X, Y) # linear kernel formulation: X @ Y^T
    poly_k = poly_kernel_func(X, Y, degree=3, gamma=1/X.shape[1], coef0=0) # polynomial kernel formulation: (gamma * X @ Y^T + coef0)^degree
    rbf_k = rbf_kernel_func(X, Y, gamma=1/X.shape[1]) # rbf kernel formulation: exp(-gamma * ||X - Y||^2)
    return weight_linear * lin_k + weight_polynomial * poly_k + weight_rbf * rbf_k

class KCCA_Scratch:
    def __init__(self, n_components=2, kernel='linear', sigma=1.0, degree=3, c=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma # kernel width parameter for the rbf kernel
        self.degree = degree # degree of the polynomial kernel for the polynomial kernel
        self.c = c # regularization parameter

    def make_kernel(self, X: np.array, Y: np.array):
        """
        Compute kernel matrix between X and Y.
        Supports: 'linear', 'rbf'/'gaussian', 'poly'/'polynomial', 'combined'
        """
        if self.kernel == 'linear':
            return linear_kernel(X, Y) # linear kernel formulation: X @ Y^T
        if self.kernel == 'rbf' or self.kernel == 'gaussian':
            # RBF kernel: exp(-gamma * ||X - Y||^2)
            # gamma = 1/(2*sigma^2) for sigma parameterization
            gamma = 1.0 / (2.0 * self.sigma ** 2) if self.sigma > 0 else 1.0 / X.shape[1]
            return rbf_kernel(X, Y, gamma=gamma)
        if self.kernel == 'poly' or self.kernel == 'polynomial':
            # Polynomial kernel: (gamma * X @ Y^T + coef0)^degree
            # Default gamma = 1/n_features, coef0 = 0
            gamma = 1.0 / X.shape[1] if X.shape[1] > 0 else 1.0
            return polynomial_kernel(X, Y, degree=self.degree, gamma=gamma, coef0=0)
        if self.kernel == 'combined':
            return combined_kernel(X, Y)
        # Default: linear kernel
        return linear_kernel(X, Y)

    def fit(self, X1, X2): # X1.shape: (n_samples, n_features_1); X2.shape: (n_samples, n_features_2)
        N = X1.shape[0] # number of samples
        # Save training data for transform
        self.X1 = X1; self.X2 = X2
        # Calculate the kernel matrices
        K1 = self.make_kernel(X1, X1) # K1.shape: (n_samples, n_samples)
        K2 = self.make_kernel(X2, X2) # K2.shape: (n_samples, n_samples)
        # Center the kernel matrices (kernel centering)
        self.N0 = np.eye(N) - 1. / N * np.ones((N, N)) # centering matrix; N0.shape: (n_samples, n_samples)
        K1 = np.dot(np.dot(self.N0, K1), self.N0) # K1.shape: (n_samples, n_samples)
        K2 = np.dot(np.dot(self.N0, K2), self.N0) # K2.shape: (n_samples, n_samples)
        # Store centered kernel matrices for transform
        self.K1 = K1
        self.K2 = K2
        # Calculate the eigenvalues and eigenvectors using Hardoon's method
        R, D = self.hardoon_method(K1, K2) # R.shape: (2*n_samples, 2*n_samples); D.shape: (2*n_samples, 2*n_samples)
        # Solve generalized eigenvalue problem: R @ alpha = beta @ D @ alpha
        betas, alphas = eigh(R, D) # betas.shape: (2*n_samples,); alphas.shape: (2*n_samples, 2*n_samples)
        # Get real parts (in case of numerical issues)
        betas = np.real(betas) # get the real part of the eigenvalues, shape: (2*n_samples,)
        alphas = np.real(alphas) # get the real part of the eigenvectors, shape: (2*n_samples, 2*n_samples)
        # Sort eigenvalues in descending order (we want the largest correlations)
        ind = np.argsort(betas)[::-1] # sort in descending order, shape: (2*n_samples,)
        betas = betas[ind] # reorder eigenvalues
        alphas = alphas[:, ind] # reorder eigenvectors, shape: (2*n_samples, 2*n_samples)
        # Take top n_components (largest eigenvalues correspond to highest correlations)
        n_comp = min(self.n_components, alphas.shape[1])
        alpha = alphas[:, :n_comp] # alpha.shape: (2*n_samples, n_comp)
        
        # In generalized eigenvalue problem R @ alpha = beta @ D @ alpha,
        # eigenvectors are normalized such that alpha^T @ D @ alpha = 1
        # However, we may need to ensure numerical stability and proper scaling
        # Split into two views
        self.alpha1 = alpha[:N, :] # alpha1.shape: (n_samples, n_comp)
        self.alpha2 = alpha[N:, :] # alpha2.shape: (n_samples, n_comp)
        
        # Optional: normalize to unit L2 norm for numerical stability
        # This doesn't change the correlation but helps with numerical issues
        for i in range(n_comp):
            norm1 = np.linalg.norm(self.alpha1[:, i])
            norm2 = np.linalg.norm(self.alpha2[:, i])
            if norm1 > 1e-10:
                self.alpha1[:, i] = self.alpha1[:, i] / norm1
            if norm2 > 1e-10:
                self.alpha2[:, i] = self.alpha2[:, i] / norm2

    def hardoon_method(self, K1, K2):
        """
        Hardoon's method for solving KCCA generalized eigenvalue problem.
        The problem is: maximize alpha1^T * K1^T * K2 * alpha2
        subject to: c * alpha_i^T * K_i * alpha_i + (1-c) * alpha_i^T * K_i^T * K_i * alpha_i = 1
        
        Returns:
        R: Cross-covariance matrix (2N x 2N)
        D: Regularization matrix (2N x 2N)
        """
        N = K1.shape[0] # number of samples
        I = np.eye(N) # identity matrix
        Z = np.zeros((N, N)) # zero matrix
        eps = 1e-8 # small epsilon for numerical stability
        
        # Construct R matrix: cross-covariance between views
        # R = [[0, K1^T @ K2], [K2^T @ K1, 0]]
        # Note: For centered kernels, K1^T = K1 and K2^T = K2 (symmetric)
        R1 = np.concatenate([Z, K1 @ K2], axis=1) # R1.shape: (n_samples, 2*n_samples)
        R2 = np.concatenate([K2 @ K1, Z], axis=1) # R2.shape: (n_samples, 2*n_samples)
        R = np.concatenate([R1, R2], axis=0) # R.shape: (2*n_samples, 2*n_samples)
        
        # Construct D matrix: regularization matrix
        # D = block_diag([(1-c)*K1^T*K1 + c*K1, (1-c)*K2^T*K2 + c*K2])
        # For centered kernels: K^T = K (symmetric), so K^T*K = K^2
        # According to Hardoon's method: D = 0.5 * block_diag([...])
        D1_block = (1 - self.c) * (K1 @ K1) + self.c * K1
        D2_block = (1 - self.c) * (K2 @ K2) + self.c * K2
        D1 = np.concatenate([D1_block, Z], axis=1) # D1.shape: (n_samples, 2*n_samples)
        D2 = np.concatenate([Z, D2_block], axis=1) # D2.shape: (n_samples, 2*n_samples)
        D = 0.5 * np.concatenate([D1, D2], axis=0) # D.shape: (2*n_samples, 2*n_samples)
        
        # Ensure D is positive definite (add small value to diagonal if needed)
        D_min_eig = np.linalg.eigvalsh(D).min()
        if D_min_eig <= 0:
            D = D - D_min_eig * np.eye(2*N) + eps * np.eye(2*N)
        
        return R, D # R.shape: (2*n_samples, 2*n_samples); D.shape: (2*n_samples, 2*n_samples)

    def transform(self, X1, X2):
        """
        Transform new data to canonical space.
        X1, X2: test data with same number of samples
        Returns: transformed representations Z1, Z2
        """
        # Compute kernel matrices between test and training data
        K1_test = self.make_kernel(X1, self.X1) # K1_test.shape: (n_test_samples, n_train_samples)
        K2_test = self.make_kernel(X2, self.X2) # K2_test.shape: (n_test_samples, n_train_samples)
        # Center the test kernel matrices using training centering matrix
        # For kernel centering: K_test_centered = (I - 1/N_test * 1) @ K_test @ (I - 1/N_train * 1)
        N_test = X1.shape[0] # number of samples in the test data
        N_train = self.X1.shape[0] # number of samples in the training data
        N0_test = np.eye(N_test) - 1. / N_test * np.ones((N_test, N_test)) # test centering matrix
        # Center test kernel: K_test_centered = N0_test @ K_test @ N0_train
        K1_test = np.dot(np.dot(N0_test, K1_test), self.N0) # K1_test.shape: (n_test_samples, n_train_samples)
        K2_test = np.dot(np.dot(N0_test, K2_test), self.N0) # K2_test.shape: (n_test_samples, n_train_samples)
        # Project to canonical space: Z = K_test @ alpha
        Z1 = np.dot(K1_test, self.alpha1) # Z1.shape: (n_test_samples, n_components)
        Z2 = np.dot(K2_test, self.alpha2) # Z2.shape: (n_test_samples, n_components)
        return Z1, Z2 # Z1.shape: (n_test_samples, n_components); Z2.shape: (n_test_samples, n_components)

def calc_corr(Z1, Z2): # Calculate the correlation of each feature between two views
    N, F = Z1.shape # Z1.shape: (num_samples, num_features); Z2.shape: (num_samples, num_features)
    # np.corrcoef(z1, z2) = [[corr(z1, z1), corr(z1, z2)], [corr(z2, z1), corr(z2, z2)]]
    corrs = [np.corrcoef(z1, z2)[0, 1] for z1, z2 in zip(Z1.T, Z2.T)] # only cross-cov
    return corrs # corrs.shape: (num_features,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KCCA')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--mode', default='scratch', type=str, choices=['cca_zoo', 'scratch'])
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
    
    # model = KCCA(latent_dimensions=n_comp, kernel=kernel, kernel_params=kernel_params, c=c)
    # model.fit([X1, X2])
    # Z1, Z2 = model.transform([X1, X2])
    if args.mode == 'scratch':
        model = KCCA_Scratch(n_components=min(10, min(X1.shape[1], X2.shape[1])), kernel='linear', sigma=1.0, c=1) # KCCA components cannot exceed the minimum dimension of two views
        model.fit(X1, X2)
        Z1, Z2 = model.transform(X1, X2) # Z1.shape: (num_samples, n_components); Z2.shape: (num_samples, n_components)
    elif args.mode == 'cca_zoo':
        model = KCCA_Zoo(latent_dimensions=min(10, min(X1.shape[1], X2.shape[1])), kernel='linear', kernel_params=None, c=1) # KCCA components cannot exceed the minimum dimension of two views
        model.fit([X1, X2])
        Z1, Z2 = model.transform([X1, X2]) # Z1.shape: (num_samples, n_components); Z2.shape: (num_samples, n_components)
    
    ## 3. Evaluate the model.
    corrs = calc_corr(Z1, Z2)
    print(f"\nCorrelation: mean={np.mean(corrs):.4f}, std={np.std(corrs):.4f}")
    Z_fused = (Z1 + Z2) / 2 # shape: (num_samples, n_components)
    kmeans = KMeans(n_clusters=class_num, random_state=42, n_init=10)
    pred = kmeans.fit_predict(Z_fused)
    nmi, ari, acc, pur = evaluate(y, pred)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, PUR: {pur:.4f}") # print the clustering results
    