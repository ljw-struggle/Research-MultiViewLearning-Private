"""
Demo: SVM with kernel="precomputed" and custom kernel functions.

When using kernel="precomputed", you pass a precomputed Gram matrix K
instead of raw features X. This allows full control over the kernel function,
which is essential for multi-view learning where kernels from different
views can be combined (e.g., averaged or weighted).

Usage:
    - fit(K_train, y_train)   where K_train[i,j] = k(x_i, x_j)
    - predict(K_test)         where K_test[i,j]  = k(x_test_i, x_train_j)
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
# from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, rbf_kernel, polynomial_kernel, linear_kernel, sigmoid_kernel

# ============================================================
# Custom Kernel Functions
# ============================================================
def linear_kernel(A, B): # K(x, y) = x^T y
    return A @ B.T


def polynomial_kernel(A, B, degree=3, coef0=1.0, gamma=1.0): # K(x, y) = (gamma * x^T y + coef0)^degree
    return (gamma * (A @ B.T) + coef0) ** degree


def rbf_kernel(A, B, gamma=1.0): # K(x, y) = exp(-gamma * ||x - y||^2)
    sq_dist = cdist(A, B, metric="sqeuclidean")  # (n_a, n_b)
    return np.exp(-gamma * sq_dist)

def sigmoid_kernel(A, B, gamma=1.0, coef0=1.0): # K(x, y) = tanh(gamma * x^T y + coef0)
    return np.tanh(gamma * (A @ B.T) + coef0)

KERNELS = {
    "linear":     lambda A, B: linear_kernel(A, B),
    "polynomial": lambda A, B: polynomial_kernel(A, B, degree=3, coef0=1.0, gamma=1.0),
    "rbf":        lambda A, B: rbf_kernel(A, B, gamma=0.5),
}


# ============================================================
# Evaluation with 5-Fold Cross-Validation
# ============================================================
def evaluate_precomputed_svm(X, y, kernel_fn, C=1.0, n_splits=5, random_state=0):
    """
    Evaluate SVM(kernel='precomputed') using stratified k-fold CV.
    Parameters
    ----------
    X : ndarray (N, d)
    y : ndarray (N,)
    kernel_fn : callable (A, B) -> (n_a, n_b)
    C : float, regularisation parameter
    n_splits : int
    random_state : int

    Returns
    -------
    mean_acc : float
    std_acc  : float
    fold_accs: list[float]
    """
    K_full = kernel_fn(X, X)  # (N, N)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        K_train = K_full[np.ix_(train_idx, train_idx)]  # (n_tr, n_tr)
        K_test = K_full[np.ix_(test_idx, train_idx)]     # (n_te, n_tr)
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K_train, y[train_idx])
        y_pred = clf.predict(K_test)
        fold_accs.append(accuracy_score(y[test_idx], y_pred))
    return np.mean(fold_accs), np.std(fold_accs), fold_accs


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    print(f"Dataset: Iris  |  N={X.shape[0]}, d={X.shape[1]}, classes={np.unique(y)}")
    print(f"{'Kernel':<15} {'Mean Acc':>10} {'Std':>10}")
    print("-" * 37)
    for name, kfn in KERNELS.items():
        mean_acc, std_acc, _ = evaluate_precomputed_svm(X, y, kfn, C=1.0)
        print(f"{name:<15} {mean_acc:>10.4f} {std_acc:>10.4f}")
