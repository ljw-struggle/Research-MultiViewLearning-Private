import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import trange
import os
from sklearn.neighbors import NearestNeighbors
from annoy import AnnoyIndex

########################################################
# Models: Autoencoder, Siamese, Spectral
########################################################
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
                self.encoder.append(nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU()))
                current_dim = next_dim

        last_dim = input_dim
        current_dim = self.architecture[-1]
        for i, layer in enumerate(reversed(self.architecture[:-1])):
            next_dim = layer
            self.decoder.append(nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU()))
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
    
########################################################
# Losses: SpectralNetLoss, ContrastiveLoss
########################################################
class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        # This function computes the loss of the SpectralNet model. The loss is the rayleigh quotient of the Laplacian matrix obtained from W, and the orthonormalized output of the network.
        m = Y.size(0)
        if is_normalized:
            D = torch.sum(W, dim=1)
            Y = Y / torch.sqrt(D)[:, None]
        Dy = torch.cdist(Y, Y)
        loss = torch.sum(W * Dy.pow(2)) / (2 * m)
        return loss
    
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
    
########################################################
# Trainers: AETrainer, SiameseTrainer, SpectralTrainer
########################################################
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
        self.ae_net = AEModel(self.architecture, input_dim=self.X.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.ae_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience)
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
            t.set_description("Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(train_loss, valid_loss, current_lr))
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
        train_loader = DataLoader(trainset, batch_size=self.ae_config["batch_size"], shuffle=True)
        valid_loader = DataLoader(validset, batch_size=self.ae_config["batch_size"], shuffle=False)
        return train_loader, valid_loader

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
        self.siamese_net = SiameseNetModel(self.architecture, input_dim=self.X.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.siamese_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience)
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
            t.set_description("Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(train_loss, valid_loss, current_lr))
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

def get_gaussian_kernel(D: torch.Tensor, scale, Ids: np.ndarray, device: torch.device, is_local: bool = True) -> torch.Tensor:
    # Computes the Gaussian similarity function according to a given distance matrix D and a given scale.
    # D: Distance matrix.
    # scale: Scale.
    # Ids: Indices of the k nearest neighbors of each sample.
    # device: Defaults to torch.device("cpu").
    # is_local: Determines whether the given scale is global or local. Defaults to True.
    # Returns: Matrix W with Gaussian similarities.
    if not is_local:
        # global scale
        W = torch.exp(-torch.pow(D, 2) / (scale**2))
    else:
        # local scales
        W = torch.exp(-torch.pow(D, 2).to(device) / (torch.tensor(scale).float().to(device).clamp_min(1e-7) ** 2))
    if Ids is not None:
        n, k = Ids.shape
        mask = torch.zeros([n, n]).to(device=device)
        for i in range(len(Ids)):
            mask[i, Ids[i]] = 1
        W = W * mask
    sym_W = (W + torch.t(W)) / 2.0
    return sym_W

def compute_scale(Dis: np.ndarray, k: int = 2, med: bool = True, is_local: bool = True) -> np.ndarray:
    # Computes the scale for the Gaussian similarity function.
    # Dis: Distances of the k nearest neighbors of each data point.
    # k: Number of nearest neighbors for the scale calculation. Relevant for global scale only.
    # med: Scale calculation method. Can be calculated by the median distance from a data point to its neighbors, or by the maximum distance. Defaults to True.
    # is_local: Local distance (different for each data point), or global distance. Defaults to True.
    # Returns: Scale (global or local).
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

def make_batch_for_sparse_grapsh(batch_x: torch.Tensor) -> torch.Tensor:
    # Computes a new batch of data points from the given batch (batch_x) in case that the graph-laplacian obtained from the given batch is sparse.
    # The new batch is computed based on the nearest neighbors of 0.25 of the given batch.
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
        # This function computes the affinity matrix W using the Gaussian kernel.
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
