import os, h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
from annoy import AnnoyIndex
import sklearn.metrics as metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import trange
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score as nmi
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def build_ann(X: torch.Tensor):
    """
    Builds approximate-nearest-neighbors object
    that can be used to calculate the k-nearest neighbors of a data-point
    """
    X = X.view(X.size(0), -1)
    t = AnnoyIndex(X[0].shape[0], "euclidean")
    for i, x_i in enumerate(X):
        t.add_item(i, x_i)
    t.build(50)
    t.save("ann_index.ann")


def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a new batch of data points from the given batch (batch_x)
    in case that the graph-laplacian obtained from the given batch is sparse.
    The new batch is computed based on the nearest neighbors of 0.25
    of the given batch.

    Parameters
    ----------
    batch_x : torch.Tensor
        Batch of data points.

    Returns
    -------
    torch.Tensor
        New batch of data points.
    """

    batch_size = batch_x.shape[0]
    batch_size //= 5
    new_batch_x = batch_x[:batch_size]
    batch_x = new_batch_x
    n_neighbors = 5

    u = AnnoyIndex(batch_x[0].shape[0], "euclidean")
    u.load("ann_index.ann")
    for x in batch_x:
        x = x.detach().cpu().numpy()
        nn_indices = u.get_nns_by_vector(x, n_neighbors)
        nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
        nn_tensors = torch.tensor(nn_tensors, device=batch_x.device)
        new_batch_x = torch.cat((new_batch_x, nn_tensors))

    return new_batch_x


def get_laplacian(W: torch.Tensor) -> np.ndarray:
    """
    Computes the unnormalized Laplacian matrix, given the affinity matrix W.

    Parameters
    ----------
    W : torch.Tensor
        Affinity matrix.

    Returns
    -------
    np.ndarray
        Laplacian matrix.
    """

    W = W.detach().cpu().numpy()
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L


def sort_laplacian(L: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the columns and rows of the Laplacian by the true labels in order
    to see whether the sorted Laplacian is a block diagonal matrix.

    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix.
    y : np.ndarray
        Labels.

    Returns
    -------
    np.ndarray
        Sorted Laplacian.
    """

    i = np.argsort(y)
    L = L[i, :]
    L = L[:, i]
    return L


def sort_matrix_rows(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the rows of a matrix by a given order.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.
    y : np.ndarray
        True labels.

    Returns
    -------
    np.ndarray
        Sorted matrix.
    """

    i = np.argsort(y)
    A = A[i, :]
    return A


def get_eigenvalues(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvalues of a given matrix A and sorts them in increasing order.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.

    Returns
    -------
    np.ndarray
        Sorted eigenvalues.
    """

    _, vals, _ = np.linalg.svd(A)
    sorted_vals = vals[np.argsort(vals)]
    return sorted_vals


def get_eigenvectors(A: np.ndarray) -> np.ndarray:
    """
    Computes the eigenvectors of a given matrix A and sorts them by the eigenvalues.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.

    Returns
    -------
    np.ndarray
        Sorted eigenvectors.
    """

    vecs, vals, _ = np.linalg.svd(A)
    vecs = vecs[:, np.argsort(vals)]
    return vecs


def plot_eigenvalues(vals: np.ndarray):
    """
    Plot the eigenvalues of the Laplacian.

    Parameters
    ----------
    vals : np.ndarray
        Eigenvalues.
    """

    rang = range(len(vals))
    plt.plot(rang, vals)
    plt.show()


def get_laplacian_eigenvectors(V: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """
    Returns eigenvectors of the Laplacian when the data is sorted in increasing
    order by the true label.

    Parameters
    ----------
    V : torch.Tensor
        Eigenvectors matrix.
    y : np.ndarray
        True labels.

    Returns
    -------
    np.ndarray
        Sorted eigenvectors matrix and range.

    """

    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    return V, rang


def plot_laplacian_eigenvectors(V: np.ndarray, y: np.ndarray):
    """
    Plot the eigenvectors of the Laplacian when the data is sorted in increasing
    order by the true label.

    Parameters
    ----------
    V : np.ndarray
        Eigenvectors matrix.
    y : np.ndarray
        True labels.

    Returns
    -------
    plt.Axes
        The matplotlib Axes object containing the plot.
    """

    V = sort_matrix_rows(V, y)
    rang = range(len(y))
    plt.plot(rang, V)
    plt.show()
    return plt


def plot_sorted_laplacian(W: torch.Tensor, y: np.ndarray):
    """
    Plot the block diagonal matrix obtained from the sorted Laplacian.

    Parameters
    ----------
    W : torch.Tensor
        Affinity matrix.
    y : np.ndarray
        True labels.
    """
    L = get_laplacian(W)
    L = sort_laplacian(L, y)
    plt.imshow(L, cmap="hot", norm=colors.LogNorm())
    plt.imshow(L, cmap="flag")
    plt.show()


def get_nearest_neighbors(
    X: torch.Tensor, Y: torch.Tensor = None, k: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the distances and the indices of the k nearest neighbors of each data point.

    Parameters
    ----------
    X : torch.Tensor
        Batch of data points.
    Y : torch.Tensor, optional
        Defaults to None.
    k : int, optional
        Number of nearest neighbors to calculate. Defaults to 3.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Distances and indices of each data point.
    """
    if Y is None:
        Y = X
    if len(X) < k:
        k = len(X)
    X = X.cpu().detach().numpy()
    Y = Y.cpu().detach().numpy()
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    Dis, Ids = nbrs.kneighbors(X)
    return Dis, Ids


def get_grassman_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes the Grassmann distance between the subspaces spanned by the columns of A and B.

    Parameters
    ----------
    A : np.ndarray
        Numpy ndarray.
    B : np.ndarray
        Numpy ndarray.

    Returns
    -------
    float
        The Grassmann distance.
    """

    M = np.dot(np.transpose(A), B)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    s = 1 - np.square(s)
    grassmann = np.sum(s)
    return grassmann


def compute_scale(
    Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True
) -> np.ndarray:
    """
    Computes the scale for the Gaussian similarity function.

    Parameters
    ----------
    Dis : np.ndarray
        Distances of the k nearest neighbors of each data point.
    k : int, optional
        Number of nearest neighbors for the scale calculation. Relevant for global scale only.
    med : bool, optional
        Scale calculation method. Can be calculated by the median distance from a data point to its neighbors,
        or by the maximum distance. Defaults to True.
    is_local : bool, optional
        Local distance (different for each data point), or global distance. Defaults to True.

    Returns
    -------
    np.ndarray
        Scale (global or local).
    """

    if is_local:
        if not med:
            scale = np.max(Dis, axis=1)
        else:
            scale = np.median(Dis, axis=1)
    else:
        if not med:
            scale = np.max(Dis[:, k - 1])
        else:
            scale = np.median(Dis[:, k - 1])
    return scale


def get_gaussian_kernel(
    D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the Gaussian similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    scale :
        Scale.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with Gaussian similarities.
    """

    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales
        W = torch.exp(
            -torch.pow(D, 2).to(device)
            / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2)
        )
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    return sym_W


def get_t_kernel(
    D: torch.Tensor, Ids: np.ndarray, device: torch.device, is_local: bool = True
) -> torch.Tensor:
    """
    Computes the t similarity function according to a given distance matrix D and a given scale.

    Parameters
    ----------
    D : torch.Tensor
        Distance matrix.
    Ids : np.ndarray
        Indices of the k nearest neighbors of each sample.
    device : torch.device
        Defaults to torch.device("cpu").
    is_local : bool, optional
        Determines whether the given scale is global or local. Defaults to True.

    Returns
    -------
    torch.Tensor
        Matrix W with t similarities.
    """

    W = torch.pow(1 + torch.pow(D, 2), -1)
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + W.T) / 2.0
    return sym_W


def get_affinity_matrix(
    X: torch.Tensor, n_neighbors: int, device: torch.device
) -> torch.Tensor:
    """
    Computes the affinity matrix for the data X.

    Parameters
    ----------
    X : torch.Tensor
        Data.
    n_neighbors : int
        Number of nearest neighbors to calculate.
    device : torch.device
        Defaults to torch.device("cpu").

    Returns
    -------
    torch.Tensor
        Affinity matrix.
    """

    Dx = torch.cdist(X, X)
    Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
    W = get_t_kernel(Dx, indices, device=device)
    return W


def plot_data_by_assignments(X, assignments: np.ndarray):
    """
    Plots the data with the assignments obtained from SpectralNet. Relevant only for 2D data.

    Parameters
    ----------
    X :
        Data.
    assignments : np.ndarray
        Cluster assignments.
    """

    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    plt.show()


def calculate_cost_matrix(C: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm.

    Parameters
    ----------
    C : np.ndarray
        Confusion matrix.
    n_clusters : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cost matrix.
    """

    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices.

    Parameters
    ----------
    indices : np.ndarray
        Indices of the clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """

    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def write_assignments_to_file(assignments: np.ndarray):
    """
    Saves SpectralNet cluster assignments to a file.

    Parameters
    ----------
    assignments : np.ndarray
        The assignments that obtained from SpectralNet.
    """

    np.savetxt(
        "cluster_assignments.csv", assignments.astype(int), fmt="%i", delimiter=","
    )


def create_weights_dir():
    """
    Creates a directory for the weights of the Autoencoder and the Siamese network
    """
    if not os.path.exists("weights"):
        os.makedirs("weights")


def calculate_cost_matrix(C: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Calculates the cost matrix for the Munkres algorithm.
    """
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices: np.ndarray) -> np.ndarray:
    """
    Gets the cluster labels from their indices.
    """
    num_clusters = len(indices)
    cluster_labels = np.zeros(num_clusters)
    for i in range(num_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

class Metrics:
    @staticmethod
    def acc_score(cluster_assignments: np.ndarray, y: np.ndarray, n_clusters: int) -> float:
        confusion_matrix = metrics.confusion_matrix(y, cluster_assignments, labels=None)
        cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters=n_clusters)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
        y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
        print(metrics.confusion_matrix(y, y_pred))
        accuracy = np.mean(y_pred == y)
        return accuracy

    @staticmethod
    def nmi_score(cluster_assignments: np.ndarray, y: np.ndarray) -> float:
        return nmi(cluster_assignments, y)
    
class AEModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(AEModel, self).__init__()
        self.architecture = architecture
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        current_dim = input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.encoder.append(nn.Sequential(nn.Linear(current_dim, next_dim)))
            else:
                self.encoder.append(
                    nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                )
                current_dim = next_dim

        last_dim = input_dim
        current_dim = self.architecture[-1]
        for i, layer in enumerate(reversed(self.architecture[:-1])):
            next_dim = layer
            self.decoder.append(
                nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            )
            current_dim = next_dim
        self.decoder.append(nn.Sequential(nn.Linear(current_dim, last_dim)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x

class AETrainer:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.ae_config = config
        self.lr = self.ae_config["lr"]
        self.epochs = self.ae_config["epochs"]
        self.min_lr = self.ae_config["min_lr"]
        self.lr_decay = self.ae_config["lr_decay"]
        self.patience = self.ae_config["patience"]
        self.architecture = self.ae_config["hiddens"]
        self.batch_size = self.ae_config["batch_size"]
        self.weights_dir = "spectralnet/_trainers/weights"
        self.weights_path = "spectralnet/_trainers/weights/ae_weights.pth"
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

    def train(self, X: torch.Tensor) -> AEModel:
        self.X = X.view(X.size(0), -1)
        self.criterion = nn.MSELoss()

        self.ae_net = AEModel(self.architecture, input_dim=self.X.shape[1]).to(
            self.device
        )

        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        if os.path.exists(self.weights_path):
            self.ae_net.load_state_dict(torch.load(self.weights_path))
            return self.ae_net

        train_loader, valid_loader = self._get_data_loader()

        print("Training Autoencoder:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            train_loss = 0.0
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                self.optimizer.zero_grad()
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.min_lr:
                break

            t.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            t.refresh()

        torch.save(self.ae_net.state_dict(), self.weights_path)
        return self.ae_net

    def validate(self, valid_loader: DataLoader) -> float:
        self.ae_net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x in valid_loader:
                batch_x = batch_x.to(self.device)
                batch_x = batch_x.view(batch_x.size(0), -1)
                output = self.ae_net(batch_x)
                loss = self.criterion(output, batch_x)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        print("Embedding data ...")
        self.ae_net.eval()
        with torch.no_grad():
            X = X.view(X.size(0), -1)
            encoded_data = self.ae_net.encode(X.to(self.device))
        return encoded_data

    def _get_data_loader(self) -> tuple:
        trainset_len = int(len(self.X) * 0.9)
        validset_len = len(self.X) - trainset_len
        trainset, validset = random_split(self.X, [trainset_len, validset_len])
        train_loader = DataLoader(
            trainset, batch_size=self.ae_config["batch_size"], shuffle=True
        )
        valid_loader = DataLoader(
            validset, batch_size=self.ae_config["batch_size"], shuffle=False
        )
        return train_loader, valid_loader
    


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean = nn.functional.pairwise_distance(output1, output2)
        positive_distance = torch.pow(euclidean, 2)
        negative_distance = torch.pow(torch.clamp(self.margin - euclidean, min=0.0), 2)
        loss = torch.mean((label * positive_distance) + ((1 - label) * negative_distance))
        return loss

class SiameseNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SiameseNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for layer in self.architecture:
            next_dim = layer
            self.layers.append(nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU()))
            current_dim = next_dim

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

class SiameseDataset:
    def __init__(self, pairs: list):
        self.pairs = pairs

    def __getitem__(self, index: int):
        x1 = self.pairs[index][0]
        x2 = self.pairs[index][1]
        label = self.pairs[index][2]
        return x1, x2, label

    def __len__(self):
        return len(self.pairs)


class SiameseTrainer:
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.siamese_config = config
        self.lr = self.siamese_config["lr"]
        self.n_nbg = self.siamese_config["n_nbg"]
        self.min_lr = self.siamese_config["min_lr"]
        self.epochs = self.siamese_config["epochs"]
        self.lr_decay = self.siamese_config["lr_decay"]
        self.patience = self.siamese_config["patience"]
        self.architecture = self.siamese_config["hiddens"]
        self.batch_size = self.siamese_config["batch_size"]
        self.use_approx = self.siamese_config["use_approx"]
        self.weights_path = "spectralnet/_trainers/weights/siamese_weights.pth"

    def train(self, X: torch.Tensor) -> SiameseNetModel:
        self.X = X.view(X.size(0), -1)
        # self.X = X

        self.criterion = ContrastiveLoss()
        self.siamese_net = SiameseNetModel(
            self.architecture, input_dim=self.X.shape[1]
        ).to(self.device)

        self.optimizer = optim.Adam(self.siamese_net.parameters(), lr=self.lr)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience
        )

        if os.path.exists(self.weights_path):
            self.siamese_net.load_state_dict(torch.load(self.weights_path))
            return self.siamese_net

        train_loader, valid_loader = self._get_data_loader()

        print("Training Siamese Network:")
        t = trange(self.epochs, leave=True)
        self.siamese_net.train()
        for epoch in t:
            train_loss = 0.0
            for x1, x2, label in train_loader:
                x1 = x1.to(self.device)
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.to(self.device)
                x2 = x2.view(x2.size(0), -1)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                output1, output2 = self.siamese_net(x1, x2)
                loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.min_lr:
                break
            t.set_description(
                "Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(
                    train_loss, valid_loss, current_lr
                )
            )
            t.refresh()

        torch.save(self.siamese_net.state_dict(), self.weights_path)
        return self.siamese_net

    def validate(self, valid_loader: DataLoader) -> float:
        valid_loss = 0.0
        self.siamese_net.eval()
        with torch.no_grad():
            for x1, x2, label in valid_loader:
                x1 = x1.to(self.device)
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.to(self.device)
                x2 = x2.view(x2.size(0), -1)
                label = label.to(self.device)
                output1, output2 = self.siamese_net(x1, x2)
                loss = self.criterion(output1, output2, label)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_knn_pairs(self) -> list:
        pairs = []
        X = self.X.detach().cpu().numpy()
        data_indices = np.arange(len(X))
        n_neighbors = self.n_nbg
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(X)
        _, neighbors_indices = nbrs.kneighbors(X)
        for i in range(len(X)):
            non_neighbors_indices = np.delete(data_indices, neighbors_indices[i])
            non_neighbors_random_chosen_indices = np.random.choice(non_neighbors_indices, n_neighbors)
            positive_pairs = [[self.X[i], self.X[n], 1] for n in neighbors_indices[i][1 : n_neighbors + 1]]
            negative_pairs = [[self.X[i], self.X[n], 0] for n in non_neighbors_random_chosen_indices]
            pairs += positive_pairs
            pairs += negative_pairs
        return pairs

    def _get_approx_nn_pairs(self) -> list:
        pairs = []
        n_samples = self.X.shape[0]
        n_neighbors = self.n_nbg
        indices = torch.randperm(self.X.shape[0])[:n_samples]
        x_train = self.X[indices]
        X_numpy = self.X[indices].detach().cpu().numpy()
        data_indices = np.arange(len(x_train))
        ann = AnnoyIndex(X_numpy.shape[1], "euclidean")
        for i, x_ in enumerate(X_numpy):
            ann.add_item(i, x_)
        ann.build(50)
        neighbors_indices = np.empty((len(X_numpy), n_neighbors + 1))
        for i in range(len(X_numpy)):
            nn_i = ann.get_nns_by_item(i, n_neighbors + 1, include_distances=False)
            neighbors_indices[i, :] = np.array(nn_i)
        neighbors_indices = neighbors_indices.astype(int)
        print("Building dataset for the siamese network ...")
        for i in range(len(X_numpy)):
            non_neighbors_indices = np.delete(data_indices, neighbors_indices[i])
            neighbor_idx = np.random.choice(neighbors_indices[i][1:], 1)
            non_nbr_idx = np.random.choice(non_neighbors_indices, 1)
            positive_pairs = [[x_train[i], x_train[neighbor_idx], 1]]
            negative_pairs = [[x_train[i], x_train[non_nbr_idx], 0]]
            pairs += positive_pairs
            pairs += negative_pairs
        return pairs

    def _get_pairs(self) -> list:
        """Gets the pairs of data points to be used for training the siamese network.
        This method internally calls either _get_knn_pairs() or _get_approx_nn_pairs() based on the value of the 'use_approx' attribute.
        """
        should_use_approx = self.use_approx
        if should_use_approx:
            return self._get_approx_nn_pairs()
        else:
            return self._get_knn_pairs()

    def _get_data_loader(self) -> tuple:
        pairs = self._get_pairs()
        siamese_dataset = SiameseDataset(pairs)
        siamese_trainset_len = int(len(siamese_dataset) * 0.9)
        siamese_validset_len = len(siamese_dataset) - siamese_trainset_len
        siamese_trainset, siamese_validset = random_split(siamese_dataset, [siamese_trainset_len, siamese_validset_len])
        siamese_trainloader = DataLoader(siamese_trainset, batch_size=self.siamese_config["batch_size"], shuffle=True)
        siamese_validloader = DataLoader(siamese_validset, batch_size=self.siamese_config["batch_size"], shuffle=False)
        return siamese_trainloader, siamese_validloader

class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        """ This function computes the loss of the SpectralNet model. The loss is the rayleigh quotient of the Laplacian matrix obtained from W, and the orthonormalized output of the network. """
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D)[:, None]
        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)
        return loss

class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self.input_dim = input_dim

        current_dim = self.input_dim
        for i, layer in enumerate(self.architecture):
            next_dim = layer
            if i == len(self.architecture) - 1:
                self.layers.append(nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh()))
            else:
                self.layers.append(nn.Sequential(nn.Linear(current_dim, next_dim), nn.LeakyReLU()))
                current_dim = next_dim

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Orthonormalize the output of the network using the Cholesky decomposition.
        This function applies QR decomposition to orthonormalize the output (`Y`) of the network.
        The inverse of the R matrix is returned as the orthonormalization weights.
        """
        m = Y.shape[0]
        _, R = torch.linalg.qr(Y)
        orthonorm_weights = np.sqrt(m) * torch.inverse(R)
        return orthonorm_weights

    def forward(self, x: torch.Tensor, should_update_orth_weights: bool = True) -> torch.Tensor:
        """
        Perform the forward pass of the model.
        This function takes an input tensor `x` and computes the forward pass of the model.
        If `should_update_orth_weights` is set to True, the orthonormalization weights are updated
        using the QR decomposition. The output tensor is returned.
        """
        for layer in self.layers:
            x = layer(x)
        Y_tilde = x
        if should_update_orth_weights:
            self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde)
        Y = Y_tilde @ self.orthonorm_weights
        return Y

class SpectralTrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse: bool = False):
        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config
        self.lr = self.spectral_config["lr"]
        self.n_nbg = self.spectral_config["n_nbg"]
        self.min_lr = self.spectral_config["min_lr"]
        self.epochs = self.spectral_config["epochs"]
        self.scale_k = self.spectral_config["scale_k"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.architecture = self.spectral_config["hiddens"]
        self.batch_size = self.spectral_config["batch_size"]
        self.is_local_scale = self.spectral_config["is_local_scale"]

    def train(self, X: torch.Tensor, y: torch.Tensor, siamese_net: nn.Module = None) -> SpectralNetModel:
        self.X = X.view(X.size(0), -1)
        self.y = y
        self.counter = 0
        self.siamese_net = siamese_net
        self.criterion = SpectralNetLoss()
        self.spectral_net = SpectralNetModel(self.architecture, input_dim=self.X.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience)
        train_loader, ortho_loader, valid_loader = self._get_data_loader()
        print("Training SpectralNet:")
        t = trange(self.epochs, leave=True)
        for epoch in t:
            train_loss = 0.0
            for (X_grad, _), (X_orth, _) in zip(train_loader, ortho_loader):
                X_grad = X_grad.to(device=self.device)
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_orth = X_orth.to(device=self.device)
                X_orth = X_orth.view(X_orth.size(0), -1)
                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)
                    X_orth = make_batch_for_sparse_grapsh(X_orth)
                # Orthogonalization step
                self.spectral_net.eval()
                self.spectral_net(X_orth, should_update_orth_weights=True)
                # Gradient step
                self.spectral_net.train()
                self.optimizer.zero_grad()
                Y = self.spectral_net(X_grad, should_update_orth_weights=False)
                if self.siamese_net is not None:
                    with torch.no_grad():
                        X_grad = self.siamese_net.forward_once(X_grad)
                W = self._get_affinity_matrix(X_grad)
                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]:
                break
            t.set_description("Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(train_loss, valid_loss, current_lr))
            t.refresh()

        return self.spectral_net

    def validate(self, valid_loader: DataLoader) -> float:
        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)
                Y = self.spectral_net(X, should_update_orth_weights=False)
                with torch.no_grad():
                    if self.siamese_net is not None:
                        X = self.siamese_net.forward_once(X)
                W = self._get_affinity_matrix(X)
                loss = self.criterion(W, Y)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor:
        """ This function computes the affinity matrix W using the Gaussian kernel. """
        is_local = self.is_local_scale
        n_neighbors = self.n_nbg
        scale_k = self.scale_k
        Dx = torch.cdist(X, X)
        Dis, indices = get_nearest_neighbors(X, k=n_neighbors + 1)
        scale = compute_scale(Dis, k=scale_k, is_local=is_local)
        W = get_gaussian_kernel(Dx, scale, indices, device=self.device, is_local=is_local)
        return W

    def _get_data_loader(self) -> tuple:
        if self.y is None:
            self.y = torch.zeros(len(self.X))
        train_size = int(0.9 * len(self.X))
        valid_size = len(self.X) - train_size
        dataset = TensorDataset(self.X, self.y)
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        ortho_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, ortho_loader, valid_loader

class SpectralNet:
    def __init__(
        self,
        n_clusters: int,
        should_use_ae: bool = False,
        should_use_siamese: bool = False,
        is_sparse_graph: bool = False,
        ae_hiddens: list = [512, 512, 2048, 10],
        ae_epochs: int = 40,
        ae_lr: float = 1e-3,
        ae_lr_decay: float = 0.1,
        ae_min_lr: float = 1e-7,
        ae_patience: int = 10,
        ae_batch_size: int = 256,
        siamese_hiddens: list = [1024, 1024, 512, 10],
        siamese_epochs: int = 30,
        siamese_lr: float = 1e-3,
        siamese_lr_decay: float = 0.1,
        siamese_min_lr: float = 1e-7,
        siamese_patience: int = 10,
        siamese_n_nbg: int = 2,
        siamese_use_approx: bool = False,
        siamese_batch_size: int = 128,
        spectral_hiddens: list = [1024, 1024, 512, 10],
        spectral_epochs: int = 30,
        spectral_lr: float = 1e-3,
        spectral_lr_decay: float = 0.1,
        spectral_min_lr: float = 1e-8,
        spectral_patience: int = 10,
        spectral_batch_size: int = 1024,
        spectral_n_nbg: int = 30,
        spectral_scale_k: int = 15,
        spectral_is_local_scale: bool = True,
    ):
        """SpectralNet is a class for implementing a Deep learning model that performs spectral clustering.
        This model optionally utilizes Autoencoders (AE) and Siamese networks for training.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to be generated by the SpectralNet algorithm.
            Also used for the dimention of the projection subspace.

        should_use_ae : bool, optional (default=False)
            Specifies whether to use the Autoencoder (AE) network as part of the training process.

        should_use_siamese : bool, optional (default=False)
                Specifies whether to use the Siamese network as part of the training process.

        is_sparse_graph : bool, optional (default=False)
            Specifies whether the graph Laplacian created from the data is sparse.

        ae_hiddens : list, optional (default=[512, 512, 2048, 10])
            The number of hidden units in each layer of the Autoencoder network.

        ae_epochs : int, optional (default=30)
            The number of epochs to train the Autoencoder network.

        ae_lr : float, optional (default=1e-3)
            The learning rate for the Autoencoder network.

        ae_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Autoencoder network.

        ae_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Autoencoder network.

        ae_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Autoencoder network.

        ae_batch_size : int, optional (default=256)
            The batch size used during training of the Autoencoder network.

        siamese_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Siamese network.

        siamese_epochs : int, optional (default=30)
            The number of epochs to train the Siamese network.

        siamese_lr : float, optional (default=1e-3)
            The learning rate for the Siamese network.

        siamese_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Siamese network.

        siamese_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Siamese network.

        siamese_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Siamese network.

        siamese_n_nbg : int, optional (default=2)
            The number of nearest neighbors to consider as 'positive' pairs by the Siamese network.

        siamese_use_approx : bool, optional (default=False)
            Specifies whether to use Annoy instead of KNN for computing nearest neighbors,
            particularly useful for large datasets.

        siamese_batch_size : int, optional (default=256)
            The batch size used during training of the Siamese network.

        spectral_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Spectral network.

        spectral_epochs : int, optional (default=30)
            The number of epochs to train the Spectral network.

        spectral_lr : float, optional (default=1e-3)
            The learning rate for the Spectral network.

        spectral_lr_decay : float, optional (default=0.1)
            The learning rate decay factor"""

        self.n_clusters = n_clusters
        self.should_use_ae = should_use_ae
        self.should_use_siamese = should_use_siamese
        self.is_sparse_graph = is_sparse_graph
        self.ae_hiddens = ae_hiddens
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_lr_decay = ae_lr_decay
        self.ae_min_lr = ae_min_lr
        self.ae_patience = ae_patience
        self.ae_batch_size = ae_batch_size
        self.siamese_hiddens = siamese_hiddens
        self.siamese_epochs = siamese_epochs
        self.siamese_lr = siamese_lr
        self.siamese_lr_decay = siamese_lr_decay
        self.siamese_min_lr = siamese_min_lr
        self.siamese_patience = siamese_patience
        self.siamese_n_nbg = siamese_n_nbg
        self.siamese_use_approx = siamese_use_approx
        self.siamese_batch_size = siamese_batch_size
        self.spectral_hiddens = spectral_hiddens
        self.spectral_epochs = spectral_epochs
        self.spectral_lr = spectral_lr
        self.spectral_lr_decay = spectral_lr_decay
        self.spectral_min_lr = spectral_min_lr
        self.spectral_patience = spectral_patience
        self.spectral_n_nbg = spectral_n_nbg
        self.spectral_scale_k = spectral_scale_k
        self.spectral_is_local_scale = spectral_is_local_scale
        self.spectral_batch_size = spectral_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._validate_spectral_hiddens()

    def _validate_spectral_hiddens(self):
        """Validates the number of hidden units in each layer of the Spectral network."""

        if self.spectral_hiddens[-1] != self.n_clusters:
            raise ValueError(
                "The number of units in the last layer of spectral_hiddens network must be equal to the number of clusters or components."
            )

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        """Performs the main training loop for the SpectralNet model.

        Parameters
        ----------
        X : torch.Tensor
            Data to train the networks on.

        y : torch.Tensor, optional
            Labels in case there are any. Defaults to None.
        """
        self._X = X
        ae_config = {
            "hiddens": self.ae_hiddens,
            "epochs": self.ae_epochs,
            "lr": self.ae_lr,
            "lr_decay": self.ae_lr_decay,
            "min_lr": self.ae_min_lr,
            "patience": self.ae_patience,
            "batch_size": self.ae_batch_size,
        }

        siamese_config = {
            "hiddens": self.siamese_hiddens,
            "epochs": self.siamese_epochs,
            "lr": self.siamese_lr,
            "lr_decay": self.siamese_lr_decay,
            "min_lr": self.siamese_min_lr,
            "patience": self.siamese_patience,
            "n_nbg": self.siamese_n_nbg,
            "use_approx": self.siamese_use_approx,
            "batch_size": self.siamese_batch_size,
        }

        spectral_config = {
            "hiddens": self.spectral_hiddens,
            "epochs": self.spectral_epochs,
            "lr": self.spectral_lr,
            "lr_decay": self.spectral_lr_decay,
            "min_lr": self.spectral_min_lr,
            "patience": self.spectral_patience,
            "n_nbg": self.spectral_n_nbg,
            "scale_k": self.spectral_scale_k,
            "is_local_scale": self.spectral_is_local_scale,
            "batch_size": self.spectral_batch_size,
        }

        if self.should_use_ae:
            self.ae_trainer = AETrainer(config=ae_config, device=self.device)
            self.ae_net = self.ae_trainer.train(X)
            X = self.ae_trainer.embed(X)

        if self.should_use_siamese:
            self.siamese_trainer = SiameseTrainer(
                config=siamese_config, device=self.device
            )
            self.siamese_net = self.siamese_trainer.train(X)
        else:
            self.siamese_net = None

        is_sparse = self.is_sparse_graph
        if is_sparse:
            build_ann(X)

        self.spectral_trainer = SpectralTrainer(
            config=spectral_config, device=self.device, is_sparse=is_sparse
        )
        self.spec_net = self.spectral_trainer.train(X, y, self.siamese_net)

    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Predicts the cluster assignments for the given data.

        Parameters
        ----------
        X : torch.Tensor
            Data to be clustered.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """
        X = X.view(X.size(0), -1)
        X = X.to(self.device)

        with torch.no_grad():
            if self.should_use_ae:
                X = self.ae_net.encode(X)
            self.embeddings_ = self.spec_net(X, should_update_orth_weights=False)
            self.embeddings_ = self.embeddings_.detach().cpu().numpy()

        cluster_assignments = self._get_clusters_by_kmeans(self.embeddings_)
        return cluster_assignments

    def get_random_batch(self, batch_size: int = 1024) -> tuple:
        """Get a batch of the input data.

        Parameters
        ----------
        batch_size : int
            The size of the batch to use.

        Returns
        -------
        tuple
            The raw batch and the encoded batch.

        """
        permuted_indices = torch.randperm(batch_size)
        X_raw = self._X.view(self._X.size(0), -1)
        X_encoded = X_raw

        if self.should_use_ae:
            X_encoded = self.ae_trainer.embed(self._X)

        if self.should_use_siamese:
            X_encoded = self.siamese_net.forward_once(X_encoded)

        X_encoded = X_encoded[permuted_indices]
        X_raw = X_raw[permuted_indices]
        X_encoded = X_encoded.to(self.device)
        return X_raw, X_encoded

    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Performs k-means clustering on the spectral-embedding space.

        Parameters
        ----------
        embeddings : np.ndarray
            The spectral-embedding space.

        Returns
        -------
        np.ndarray
            The cluster assignments for the given data.
        """

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments


class SpectralReduction:
    def __init__(
        self,
        n_components: int,
        should_use_ae: bool = False,
        should_use_siamese: bool = False,
        is_sparse_graph: bool = False,
        ae_hiddens: list = [512, 512, 2048, 10],
        ae_epochs: int = 40,
        ae_lr: float = 1e-3,
        ae_lr_decay: float = 0.1,
        ae_min_lr: float = 1e-7,
        ae_patience: int = 10,
        ae_batch_size: int = 256,
        siamese_hiddens: list = [1024, 1024, 512, 10],
        siamese_epochs: int = 30,
        siamese_lr: float = 1e-3,
        siamese_lr_decay: float = 0.1,
        siamese_min_lr: float = 1e-7,
        siamese_patience: int = 10,
        siamese_n_nbg: int = 2,
        siamese_use_approx: bool = False,
        siamese_batch_size: int = 128,
        spectral_hiddens: list = [1024, 1024, 512, 10],
        spectral_epochs: int = 30,
        spectral_lr: float = 1e-3,
        spectral_lr_decay: float = 0.1,
        spectral_min_lr: float = 1e-8,
        spectral_patience: int = 10,
        spectral_batch_size: int = 1024,
        spectral_n_nbg: int = 30,
        spectral_scale_k: int = 15,
        spectral_is_local_scale: bool = True,
    ):
        """SpectralNet is a class for implementing a Deep learning model that performs spectral clustering.
        This model optionally utilizes Autoencoders (AE) and Siamese networks for training.
        Parameters
        ----------
        n_components : int
            The number of components to keep.
        should_use_ae : bool, optional (default=False)
            Specifies whether to use the Autoencoder (AE) network as part of the training process.
        should_use_siamese : bool, optional (default=False)
                Specifies whether to use the Siamese network as part of the training process.
        is_sparse_graph : bool, optional (default=False)
            Specifies whether the graph Laplacian created from the data is sparse.
        ae_hiddens : list, optional (default=[512, 512, 2048, 10])
            The number of hidden units in each layer of the Autoencoder network.
        ae_epochs : int, optional (default=30)
            The number of epochs to train the Autoencoder network.
        ae_lr : float, optional (default=1e-3)
            The learning rate for the Autoencoder network.
        ae_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Autoencoder network.
        ae_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Autoencoder network.
        ae_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Autoencoder network.
        ae_batch_size : int, optional (default=256)
            The batch size used during training of the Autoencoder network.
        siamese_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Siamese network.
        siamese_epochs : int, optional (default=30)
            The number of epochs to train the Siamese network.
        siamese_lr : float, optional (default=1e-3)
            The learning rate for the Siamese network.
        siamese_lr_decay : float, optional (default=0.1)
            The learning rate decay factor for the Siamese network.
        siamese_min_lr : float, optional (default=1e-7)
            The minimum learning rate for the Siamese network.
        siamese_patience : int, optional (default=10)
            The number of epochs to wait before reducing the learning rate for the Siamese network.
        siamese_n_nbg : int, optional (default=2)
            The number of nearest neighbors to consider as 'positive' pairs by the Siamese network.
        siamese_use_approx : bool, optional (default=False)
            Specifies whether to use Annoy instead of KNN for computing nearest neighbors,
            particularly useful for large datasets.
        siamese_batch_size : int, optional (default=256)
            The batch size used during training of the Siamese network.
        spectral_hiddens : list, optional (default=[1024, 1024, 512, 10])
            The number of hidden units in each layer of the Spectral network.
        spectral_epochs : int, optional (default=30)
            The number of epochs to train the Spectral network.
        spectral_lr : float, optional (default=1e-3)
            The learning rate for the Spectral network.
        spectral_lr_decay : float, optional (default=0.1)
            The learning rate decay factor """

        self.n_components = n_components
        self.should_use_ae = should_use_ae
        self.should_use_siamese = should_use_siamese
        self.is_sparse_graph = is_sparse_graph
        self.ae_hiddens = ae_hiddens
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.ae_lr_decay = ae_lr_decay
        self.ae_min_lr = ae_min_lr
        self.ae_patience = ae_patience
        self.ae_batch_size = ae_batch_size
        self.siamese_hiddens = siamese_hiddens
        self.siamese_epochs = siamese_epochs
        self.siamese_lr = siamese_lr
        self.siamese_lr_decay = siamese_lr_decay
        self.siamese_min_lr = siamese_min_lr
        self.siamese_patience = siamese_patience
        self.siamese_n_nbg = siamese_n_nbg
        self.siamese_use_approx = siamese_use_approx
        self.siamese_batch_size = siamese_batch_size
        self.spectral_hiddens = spectral_hiddens
        self.spectral_epochs = spectral_epochs
        self.spectral_lr = spectral_lr
        self.spectral_lr_decay = spectral_lr_decay
        self.spectral_min_lr = spectral_min_lr
        self.spectral_patience = spectral_patience
        self.spectral_n_nbg = spectral_n_nbg
        self.spectral_scale_k = spectral_scale_k
        self.spectral_is_local_scale = spectral_is_local_scale
        self.spectral_batch_size = spectral_batch_size
        self.X_new = None

    def _fit(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """ Fit the SpectralNet model to the input data. """
        self._spectralnet = SpectralNet(
            n_clusters=self.n_components,
            should_use_ae=self.should_use_ae,
            should_use_siamese=self.should_use_siamese,
            is_sparse_graph=self.is_sparse_graph,
            ae_hiddens=self.ae_hiddens,
            ae_epochs=self.ae_epochs,
            ae_lr=self.ae_lr,
            ae_lr_decay=self.ae_lr_decay,
            ae_min_lr=self.ae_min_lr,
            ae_patience=self.ae_patience,
            ae_batch_size=self.ae_batch_size,
            siamese_hiddens=self.siamese_hiddens,
            siamese_epochs=self.siamese_epochs,
            siamese_lr=self.siamese_lr,
            siamese_lr_decay=self.siamese_lr_decay,
            siamese_min_lr=self.siamese_min_lr,
            siamese_patience=self.siamese_patience,
            siamese_n_nbg=self.siamese_n_nbg,
            siamese_use_approx=self.siamese_use_approx,
            siamese_batch_size=self.siamese_batch_size,
            spectral_hiddens=self.spectral_hiddens,
            spectral_epochs=self.spectral_epochs,
            spectral_lr=self.spectral_lr,
            spectral_lr_decay=self.spectral_lr_decay,
            spectral_min_lr=self.spectral_min_lr,
            spectral_patience=self.spectral_patience,
            spectral_n_nbg=self.spectral_n_nbg,
            spectral_scale_k=self.spectral_scale_k,
            spectral_is_local_scale=self.spectral_is_local_scale,
            spectral_batch_size=self.spectral_batch_size,
        )
        self._spectralnet.fit(X, y)

    def _predict(self, X: torch.Tensor) -> np.ndarray:
        """ Predict embeddings for the input data using the fitted SpectralNet model. """
        self._spectralnet.predict(X)
        return self._spectralnet.embeddings_

    def _transform(self, X: torch.Tensor) -> np.ndarray:
        """ Transform the input data into embeddings using the fitted SpectralNet model """
        return self._predict(X)

    def fit_transform(self, X: torch.Tensor, y: torch.Tensor = None) -> np.ndarray:
        """ Fit the SpectralNet model to the input data and transform it into embeddings. """
        self._fit(X, y)
        return self._transform(X)

    def _get_laplacian_of_small_batch(self, batch: torch.Tensor) -> np.ndarray:
        """ Get the Laplacian of a small batch of the input data """
        W = get_affinity_matrix(batch, self.spectral_n_nbg, self._spectralnet.device)
        L = get_laplacian(W)
        return L

    def _remove_smallest_eigenvector(self, V: np.ndarray) -> np.ndarray:
        """ Remove the constant eigenvector from the eigenvectors of the Laplacian of a small batch of the input data. """
        batch_raw, batch_encoded = self._spectralnet.get_random_batch()
        L_batch = self._get_laplacian_of_small_batch(batch_encoded)
        V_batch = self._predict(batch_raw)
        eigenvalues = np.diag(V_batch.T @ L_batch @ V_batch)
        indices = np.argsort(eigenvalues)
        smallest_index = indices[0]
        V = V[:, np.arange(V.shape[1]) != smallest_index]
        V = V[:,(np.arange(V.shape[1]) == indices[1]) | (np.arange(V.shape[1]) == indices[2])]
        return V

    def visualize(self, V: np.ndarray, y: torch.Tensor = None, n_components: int = 1) -> None:
        """ Visualize the embeddings of the input data using the fitted SpectralNet model. """
        V = self._remove_smallest_eigenvector(V)
        print(V.shape)
        plot_laplacian_eigenvectors(V, y)
        cluster_labels = self._get_clusters_by_kmeans(V)
        acc = Metrics.acc_score(cluster_labels, y.detach().cpu().numpy(), n_clusters=10)
        print("acc with 2 components: ", acc)
        if n_components > 1:
            x_axis = V[:, 0]
            y_axis = V[:, 1]
        elif n_components == 1:
            x_axis = V
            y_axis = np.zeros_like(V)
        else:
            raise ValueError("n_components must be a positive integer (greater than 0)")
        if y is None:
            plt.scatter(x_axis, y_axis)
        else:
            plt.scatter(x_axis, y_axis, c=y, cmap="tab10", s=3)
        plt.show()

    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Performs k-means clustering on the spectral-embedding space.
        """
        kmeans = KMeans(n_clusters=self.n_components, n_init=10).fit(embeddings)
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments
    
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

def load_reuters() -> tuple:
    with h5py.File("../data/Reuters/reutersidf_total.h5", "r") as f:
        x = np.asarray(f.get("data"), dtype="float32")
        y = np.asarray(f.get("labels"), dtype="float32")
        n_train = int(0.9 * len(x))
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
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
    elif dataset == "reuters":
        x_train, y_train, x_test, y_test = load_reuters()
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

def cluster_twomoons():
    x_train, x_test, y_train, y_test = load_data("twomoons")
    X = torch.cat([x_train, x_test])
    y = torch.cat([y_train, y_test])
    spectralnet = SpectralNet(n_clusters=2, should_use_ae=False, should_use_siamese=False, spectral_batch_size=712, spectral_epochs=40, spectral_is_local_scale=False, spectral_n_nbg=8, spectral_scale_k=2, spectral_lr=1e-2, spectral_hiddens=[128, 128, 2])
    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_
    y = y.detach().cpu().numpy()
    acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=2)
    nmi_score = Metrics.nmi_score(cluster_assignments, y)
    print(f"ACC: {np.round(acc_score, 3)}")
    print(f"NMI: {np.round(nmi_score, 3)}")

def cluster_mnist():
    x_train, x_test, y_train, y_test = load_data("mnist")
    X = torch.cat([x_train, x_test])
    y = torch.cat([y_train, y_test])
    spectralnet = SpectralNet(n_clusters=10, should_use_ae=True, should_use_siamese=True)
    spectralnet.fit(X, y)
    cluster_assignments = spectralnet.predict(X)
    embeddings = spectralnet.embeddings_
    y = y.detach().cpu().numpy()
    acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters=10)
    nmi_score = Metrics.nmi_score(cluster_assignments, y)
    print(f"ACC: {np.round(acc_score, 3)}")
    print(f"NMI: {np.round(nmi_score, 3)}")

def main_mnist_reduction():
    x_train, x_test, y_train, y_test = load_data("mnist")
    X = torch.cat([x_train, x_test])
    y = torch.cat([y_train, y_test])
    spectralreduction = SpectralReduction(n_components=3, should_use_ae=True, should_use_siamese=True, spectral_hiddens=[512, 512, 2048, 3])
    X_new = spectralreduction.fit_transform(X)
    spectralreduction.visualize(X_new, y, n_components=2)

def main_twomoons_reduction():
    x_train, x_test, y_train, y_test = load_data("twomoons")
    X = torch.cat([x_train, x_test])
    y = torch.cat([y_train, y_test])
    spectralreduction = SpectralReduction(n_components=2, should_use_ae=False, should_use_siamese=False, spectral_batch_size=712, spectral_epochs=40, spectral_is_local_scale=False, spectral_n_nbg=8, spectral_scale_k=2, spectral_lr=1e-2, spectral_hiddens=[128, 128, 2])
    X_new = spectralreduction.fit_transform(X)
    spectralreduction.visualize(X_new, y, n_components=1)

if __name__ == "__main__":
    cluster_twomoons()
    cluster_mnist()
    main_mnist_reduction()
    main_twomoons_reduction()