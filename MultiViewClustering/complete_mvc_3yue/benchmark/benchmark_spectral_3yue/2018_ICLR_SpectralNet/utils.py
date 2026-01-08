import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

def load_mnist() -> tuple:
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root="../data", train=True, download=True, transform=tensor_transform)
    test_set = datasets.MNIST(root="../data", train=False, download=True, transform=tensor_transform)
    x_train, y_train = zip(*train_set)
    x_train, y_train = torch.cat(x_train), torch.Tensor(y_train)
    x_test, y_test = zip(*test_set)
    x_test, y_test = torch.cat(x_test), torch.Tensor(y_test)
    return x_train, y_train, x_test, y_test

def load_twomoon() -> tuple:
    data, y = make_moons(n_samples=7000, shuffle=True, noise=0.075, random_state=None)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
    x_train, x_test = torch.Tensor(x_train), torch.Tensor(x_test)
    y_train, y_test = torch.Tensor(y_train), torch.Tensor(y_test)
    return x_train, y_train, x_test, y_test

def load_from_path(dpath: str, lpath: str = None) -> tuple:
    X = np.loadtxt(dpath, delimiter=",", dtype=np.float32)
    n_train = int(0.9 * len(X))
    x_train, x_test = X[:n_train], X[n_train:]
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    if lpath is not None:
        y = np.loadtxt(lpath, delimiter=",", dtype=np.float32)
        y_train, y_test = y[:n_train], y[n_train:]
        y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    else:
        y_train, y_test = None, None
    return x_train, y_train, x_test, y_test

def load_data(dataset: str) -> tuple:
    if dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist()
    elif dataset == "twomoons":
        x_train, y_train, x_test, y_test = load_twomoon()
    else:
        try:
            data_path = dataset["dpath"]
            if "lpath" in dataset:
                label_path = dataset["lpath"]
            else:
                label_path = None
        except:
            raise ValueError("Could not find dataset path. Check your config file.")
        x_train, y_train, x_test, y_test = load_from_path(data_path, label_path)
    return x_train, x_test, y_train, y_test

def build_ann(X: torch.Tensor):
    # Builds approximate-nearest-neighbors object that can be used to calculate the k-nearest neighbors of a data-point
    X = X.view(X.size(0), -1)
    t = AnnoyIndex(X[0].shape[0], "euclidean")
    for i, x_i in enumerate(X):
        t.add_item(i, x_i)
    t.build(50)
    t.save("ann_index.ann")

def get_laplacian(W: torch.Tensor) -> np.ndarray:
    # Computes the unnormalized Laplacian matrix, given the affinity matrix W.
    W = W.detach().cpu().numpy()
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L

def sort_laplacian(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Sorts the columns and rows of the Laplacian by the true labels in order to see whether the sorted Laplacian is a block diagonal matrix.
    i = np.argsort(y)
    L = L[i, :]
    L = L[:, i]
    return L

def sort_matrix_rows(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Sorts the rows of a matrix by a given order.
    i = np.argsort(y)
    A = A[i, :]
    return A

def get_eigenvalues(A: np.ndarray) -> np.ndarray:
    # Computes the eigenvalues of a given matrix A and sorts them in increasing order.
    _, vals, _ = np.linalg.svd(A)
    sorted_vals = vals[np.argsort(vals)]
    return sorted_vals

def get_eigenvectors(A: np.ndarray) -> np.ndarray:
    # Computes the eigenvectors of a given matrix A and sorts them by the eigenvalues.
    vecs, vals, _ = np.linalg.svd(A)
    vecs = vecs[:, np.argsort(vals)]
    return vecs

def plot_eigenvalues(vals: np.ndarray):
    # Plot the eigenvalues of the Laplacian.
    rang = range(len(vals))
    plt.plot(rang, vals)
    plt.show()

def get_laplacian_eigenvectors(V: torch.Tensor, y: np.ndarray) -> np.ndarray:
    # Returns eigenvectors of the Laplacian when the data is sorted in increasing order by the true label.
    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    return V, rang

def plot_laplacian_eigenvectors(V: np.ndarray, y: np.ndarray):
    # Plot the eigenvectors of the Laplacian when the data is sorted in increasing order by the true label.
    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    plt.plot(rang, V)
    plt.show()
    return plt

def plot_sorted_laplacian(W: torch.Tensor, y: np.ndarray):
    # Plot the block diagonal matrix obtained from the sorted Laplacian.
    L = get_laplacian(W)
    L = sort_laplacian(L, y)
    plt.imshow(L, cmap="hot", norm=colors.LogNorm())
    plt.imshow(L, cmap="flag")
    plt.show()

def get_grassman_distance(A: np.ndarray, B: np.ndarray) -> float:
    # Computes the Grassmann distance between the subspaces spanned by the columns of A and B.
    M = np.dot(np.transpose(A), B)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = 1 - np.square(s)
    grassmann = np.sum(s)
    return grassmann

def get_t_kernel(D: torch.Tensor, Ids: np.ndarray, device: torch.device, is_local: bool = True) -> torch.Tensor:
    # Computes the t similarity function according to a given distance matrix D and a given scale.
    # D: Distance matrix.
    # Ids: Indices of the k nearest neighbors of each sample.
    # device: Defaults to torch.device("cpu").
    # is_local: Determines whether the given scale is global or local. Defaults to True.
    # Returns: Matrix W with t similarities.
    W = torch.pow(1 + torch.pow(D, 2), -1)
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + W.T) / 2.0
    return sym_W

def get_nearest_neighbors(X: torch.Tensor, Y: torch.Tensor = None, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    # Computes the distances and the indices of the k nearest neighbors of each data point.
    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids

def get_affinity_matrix(X: torch.Tensor, n_neighbors: int, device: torch.device) -> torch.Tensor:
    # Computes the affinity matrix for the data X.
    Dx = torch.cdist(X, X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    W = get_t_kernel(Dx, indices, device=device)
    return W

def plot_data_by_assignments(X, assignments: np.ndarray):
    # Plots the data with the assignments obtained from SpectralNet. Relevant only for 2D data.
    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.show()

def write_assignments_to_file(assignments: np.ndarray):
    # Saves SpectralNet cluster assignments to a file.
    np.savetxt("cluster_assignments.csv", assignments.astype(int), fmt="%i", delimiter=",")
