import os, numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset, random_split
from annoy import AnnoyIndex
from tqdm import tqdm_gui, trange

########################################################################################################################
# Models for the Autoencoder, Siamese, and Spectral networks.
########################################################################################################################
class AEModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(AEModel, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        dim_list = [input_dim] + architecture
        for i in range(len(dim_list) - 1):
            self.encoder.append(nn.Sequential(nn.Linear(dim_list[i], dim_list[i+1]), nn.ReLU())) if i < len(dim_list) - 2 else None
            self.encoder.append(nn.Sequential(nn.Linear(dim_list[i], dim_list[i+1]))) if i == len(dim_list) - 2 else None
        dim_list.reverse()
        for i in range(len(dim_list) - 1):
            self.decoder.append(nn.Sequential(nn.Linear(dim_list[i], dim_list[i+1]), nn.ReLU())) if i < len(dim_list) - 2 else None
            self.decoder.append(nn.Sequential(nn.Linear(dim_list[i], dim_list[i+1]))) if i == len(dim_list) - 2 else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
    
class SiameseNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SiameseNetModel, self).__init__()
        self.encoder = nn.ModuleList()
        dim_list = [input_dim] + architecture
        for i in range(len(dim_list) - 1):
            self.encoder.append(nn.Sequential(nn.Linear(dim_list[i], dim_list[i+1]), nn.ReLU()))

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        return self.forward_once(x1), self.forward_once(x2)
    
class SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SpectralNetModel, self).__init__()
        self.layers = nn.ModuleList()
        dim_list = [input_dim] + architecture
        for i in range(len(dim_list) - 1):
            self.layers.append(nn.Sequential(nn.Linear(dim_list[i], dim_list[i+1]), nn.LeakyReLU())) if i < len(dim_list) - 2 else None
            self.layers.append(nn.Sequential(nn.Linear(dim_list[i], dim_list[i+1]), nn.Tanh())) if i == len(dim_list) - 2 else None

    def _make_orthonorm_weights(self, Y: torch.Tensor) -> torch.Tensor:
        # Orthonormalize the output of the network using QR decomposition.
        # Q, R = torch.linalg.qr(Y). Returns Q and R such that Y = QR, Q^T @ Q = I and R is upper triangular.
        # Q is orthonormal matrix: every column is orthogonal to every other column and has unit length.
        # return sqrt(N) * R^{-1} as the orthonormalization weights. (Y @ R^{-1} = Q, so Y @ sqrt(N) * R^{-1} = sqrt(N) * Q.)
        _, R = torch.linalg.qr(Y)
        return np.sqrt(Y.shape[0]) * torch.inverse(R)

    def forward(self, x: torch.Tensor, should_update_orth_weights: bool = True) -> torch.Tensor:
        # If `should_update_orth_weights` is set to True, the orthonormalization weights are updated using the QR decomposition.
        for layer in self.layers:
            x = layer(x)
        Y_tilde = x # Y_tilde.shape: (N, embedding_dim).
        if should_update_orth_weights: self.orthonorm_weights = self._make_orthonorm_weights(Y_tilde) # self.orthonorm_weights.shape: (embedding_dim, embedding_dim).
        Y = Y_tilde @ self.orthonorm_weights # Y.shape: (N, embedding_dim). Y = Y_tilde @ self.orthonorm_weights = sqrt(N) * Q, so Y @ Y^T = N * I.
        return Y
    
########################################################################################################################
# Loss functions for the Spectral network.
########################################################################################################################
class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(self, W: torch.Tensor, Y: torch.Tensor, is_normalized: bool = False) -> torch.Tensor:
        # The loss is the rayleigh quotient of the affinity matrix W and the orthonormalized output Y of the network.
        # Destination: push together the data points that are similar to each other and push apart the data points that are not similar to each other.
        N = Y.size(0) # N is the number of data points.
        if is_normalized:
            D = torch.sum(W, dim=1) # D.shape: (N,). D = sum(W_i,j) for j in N.
            Y = Y / torch.sqrt(D)[:, None] # Y = Y / sqrt(D), so Y @ Y^T = I.
        Dy = torch.cdist(Y, Y) # Dy.shape: (N, N). Dy = sqrt(||Y_i - Y_j||^2) (euclidean distance between Y_i and Y_j).
        return torch.sum(W * Dy.pow(2)) / (2 * N) # The loss is the rayleigh quotient of the affinity matrix W and the orthonormalized output Y of the network.
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Destination: push together the data points that positive pairs and push apart the data points that negative pairs.
        euclidean = nn.functional.pairwise_distance(output1, output2) # euclidean.shape: (N,). euclidean = sqrt(||output1_i - output2_j||^2)
        positive_distance = torch.pow(euclidean, 2) # positive_distance = euclidean^2.
        negative_distance = torch.pow(torch.clamp(self.margin - euclidean, min=0.0), 2) # negative_distance = max(0, margin - euclidean)^2.
        loss = torch.mean((label * positive_distance) + ((1 - label) * negative_distance)) 
        return loss

########################################################################################################################
# Trainers for the Autoencoder, Siamese, and Spectral networks.
########################################################################################################################
class AETrainer:
    def __init__(self, config: dict, device: torch.device):
        self.lr = config["lr"]; self.epochs = config["epochs"]; self.min_lr = config["min_lr"]; self.lr_decay = config["lr_decay"]
        self.patience = config["patience"]; self.architecture = config["hiddens"]; self.batch_size = config["batch_size"]
        self.device = device; self.weights_path = "./result/ae_weights.pth"

    def train(self, X: torch.Tensor) -> AEModel:
        self.ae_net = AEModel(self.architecture, input_dim=X.shape[1]).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.ae_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience)
        if os.path.exists(self.weights_path):
            self.ae_net.load_state_dict(torch.load(self.weights_path))
            return self.ae_net
        trainset, validset = random_split(X, [int(len(X) * 0.9), len(X) - int(len(X) * 0.9)])
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False)
        tqdm_range = trange(self.epochs, leave=True, desc="Training Autoencoder:")
        for _ in tqdm_range:
            self.ae_net.train()
            train_loss_list = []
            for batch_x in train_loader:
                batch_x = batch_x.to(self.device) # batch_x.shape: (batch_size, input_dim).
                self.optimizer.zero_grad()
                output = self.ae_net(batch_x) # output.shape: (batch_size, output_dim).
                loss = self.criterion(output, batch_x)
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())
            self.ae_net.eval()
            valid_loss_list = []
            with torch.no_grad():
                for batch_x in valid_loader:
                    batch_x = batch_x.to(self.device)
                    output = self.ae_net(batch_x)
                    loss = self.criterion(output, batch_x)
                    valid_loss_list.append(loss.item())
            self.scheduler.step(np.mean(valid_loss_list))
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.min_lr: break # Stop training if the learning rate is below the minimum learning rate.
            tqdm_range.set_description("Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(np.mean(train_loss_list), np.mean(valid_loss_list), current_lr))
            tqdm_range.refresh()
        torch.save(self.ae_net.state_dict(), self.weights_path)
        return self.ae_net

    def embed(self, X: torch.Tensor) -> torch.Tensor:
        self.ae_net.eval()
        with torch.no_grad():
            X_encoded = self.ae_net.encode(X.to(self.device))
        return X_encoded
    
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
        self.lr = config["lr"]; self.n_nbg = config["n_nbg"]; self.min_lr = config["min_lr"]; self.epochs = config["epochs"]
        self.lr_decay = config["lr_decay"]; self.patience = config["patience"]; self.architecture = config["hiddens"]
        self.batch_size = config["batch_size"]; self.use_approx = config["use_approx"]
        self.device = device; self.weights_path = "./result/siamese_weights.pth"

    def train(self, X: torch.Tensor) -> SiameseNetModel:
        self.siamese_net = SiameseNetModel(self.architecture, input_dim=X.shape[1]).to(self.device)
        self.criterion = ContrastiveLoss()
        self.optimizer = torch.optim.Adam(self.siamese_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience)
        if os.path.exists(self.weights_path):
            self.siamese_net.load_state_dict(torch.load(self.weights_path))
            return self.siamese_net
        data = self._get_pairs(X)
        # dataset = TensorDataset(torch.stack([x[0] for x in data]), torch.stack([x[1] for x in data]), torch.stack([x[2] for x in data]))
        dataset = SiameseDataset(data)
        trainset, validset = random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=self.batch_size, shuffle=False)
        tqdm_range = trange(self.epochs, leave=True, desc="Training Siamese Network:")
        for _ in tqdm_range:
            self.siamese_net.train()
            train_loss_list = []
            for x1, x2, label in train_loader:
                x1 = x1.to(self.device); x2 = x2.to(self.device); label = label.to(self.device)
                self.optimizer.zero_grad()
                output1, output2 = self.siamese_net(x1, x2)
                loss = self.criterion(output1, output2, label)
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())
            self.siamese_net.eval()
            valid_loss_list = []
            with torch.no_grad():
                for x1, x2, label in valid_loader:
                    x1 = x1.to(self.device); x2 = x2.to(self.device); label = label.to(self.device)
                    output1, output2 = self.siamese_net(x1, x2)
                    loss = self.criterion(output1, output2, label)
                    valid_loss_list.append(loss.item())
            self.scheduler.step(np.mean(valid_loss_list))
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.min_lr: break # Stop training if the learning rate is below the minimum learning rate.
            tqdm_range.set_description("Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(np.mean(train_loss_list), np.mean(valid_loss_list), current_lr))
            tqdm_range.refresh()
        torch.save(self.siamese_net.state_dict(), self.weights_path)
        return self.siamese_net

    def _get_pairs(self, X: torch.Tensor) -> list: # Gets the pairs of data points to be used for training the siamese network.
        should_use_approx = self.use_approx
        if should_use_approx: # Use approximate nearest neighbors search for large datasets.
            indices = torch.randperm(X.shape[0]) # indices.shape: (N,). indices = random permutation of the indices of the data points.
            x_train = X[indices]; X_numpy = X[indices].detach().cpu().numpy()
            data_indices = np.arange(len(x_train))
            ann = AnnoyIndex(X_numpy.shape[1], "euclidean") # Create an Annoy index for the data points.
            for i, x_ in enumerate(X_numpy):
                ann.add_item(i, x_) # Add the data points to the Annoy index.
            ann.build(50) # Build the Annoy index.
            neighbors_indices = np.empty((len(X_numpy), self.n_nbg + 1))
            for i in range(len(X_numpy)):
                nn_i = ann.get_nns_by_item(i, self.n_nbg + 1, include_distances=False)
                neighbors_indices[i, :] = np.array(nn_i)
            neighbors_indices = neighbors_indices.astype(int)
            pairs = [] # pairs.shape: (N, 3)
            for i in range(len(X_numpy)):
                non_neighbors_indices = np.delete(data_indices, neighbors_indices[i])
                neighbor_idx = np.random.choice(neighbors_indices[i][1:], 1) # random choice of the index of the nearest neighbor of the i-th data point.
                non_nbr_idx = np.random.choice(non_neighbors_indices, 1) # random choice of the index of the non-nearest neighbor of the i-th data point.
                positive_pairs = [[x_train[i], x_train[neighbor_idx].squeeze(0), torch.tensor(1.0)]]
                negative_pairs = [[x_train[i], x_train[non_nbr_idx].squeeze(0), torch.tensor(0.0)]]
                pairs += positive_pairs
                pairs += negative_pairs
            return pairs
        else: # Use exact nearest neighbors search for small datasets.
            X_numpy = X.detach().cpu().numpy()
            data_indices = np.arange(len(X_numpy))
            nbrs = NearestNeighbors(n_neighbors=self.n_nbg + 1, algorithm="ball_tree").fit(X_numpy)
            _, idx_nbrs = nbrs.kneighbors(X_numpy)
            pairs = [] # pairs.shape: (N * self.n_nbg, 3)
            for i in range(len(X_numpy)):
                non_neighbors_indices = np.delete(data_indices, idx_nbrs[i])
                non_neighbors_random_chosen_indices = np.random.choice(non_neighbors_indices, self.n_nbg)
                positive_pairs = [[X[i], X[n], torch.tensor(1.0)] for n in idx_nbrs[i][1 : self.n_nbg + 1]]
                negative_pairs = [[X[i], X[n], torch.tensor(0.0)] for n in non_neighbors_random_chosen_indices]
                pairs += positive_pairs
                pairs += negative_pairs
            return pairs

class SpectralTrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse: bool = False):
        self.lr = config["lr"]; self.n_nbg = config["n_nbg"]; self.min_lr = config["min_lr"]; self.epochs = config["epochs"]
        self.scale_k = config["scale_k"]; self.lr_decay = config["lr_decay"]; self.patience = config["patience"]
        self.architecture = config["hiddens"]; self.batch_size = config["batch_size"]; self.is_local_scale = config["is_local_scale"]
        self.device = device; self.is_sparse = is_sparse

    def train(self, X: torch.Tensor, y: torch.Tensor, siamese_net: nn.Module = None) -> SpectralNetModel:
        self.siamese_net = siamese_net
        self.criterion = SpectralNetLoss()
        self.spectral_net = SpectralNetModel(self.architecture, input_dim=X.shape[1]).to(self.device)
        self.optimizer = torch.optim.Adam(self.spectral_net.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.lr_decay, patience=self.patience)
        dataset = TensorDataset(X, y)
        train_dataset, valid_dataset = random_split(dataset, [int(0.9 * len(X)), len(X) - int(0.9 * len(X))])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        ortho_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        self.make_annoy_index_for_sparse_graph(X) if self.is_sparse else None
        tqdm_range = trange(self.epochs, leave=True, desc="Training SpectralNet:")
        for _ in tqdm_range:
            train_loss_list = []
            for (X_grad, _), (X_orth, _) in zip(train_loader, ortho_loader):
                X_grad = X_grad.to(self.device); X_orth = X_orth.to(self.device)
                X_grad = self.make_batch_for_sparse_grapsh(X_grad) if self.is_sparse else X_grad
                X_orth = self.make_batch_for_sparse_grapsh(X_orth) if self.is_sparse else X_orth
                self.spectral_net.eval() # Orthogonalization step
                self.spectral_net(X_orth, should_update_orth_weights=True)
                self.spectral_net.train() # Gradient step
                self.optimizer.zero_grad()
                Y = self.spectral_net(X_grad, should_update_orth_weights=False)
                if self.siamese_net is not None:
                    with torch.no_grad():
                        X_grad = self.siamese_net.forward_once(X_grad)
                W = self._get_affinity_matrix(X_grad) # W.shape: (N, N). W = affinity matrix.
                loss = self.criterion(W, Y)
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.item())
            valid_loss_list = []
            self.spectral_net.eval()
            with torch.no_grad():
                for X, y in valid_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    X = self.make_batch_for_sparse_grapsh(X) if self.is_sparse else X
                    Y = self.spectral_net(X, should_update_orth_weights=False)
                    with torch.no_grad():
                        if self.siamese_net is not None:
                            X = self.siamese_net.forward_once(X)
                    W = self._get_affinity_matrix(X)
                    loss = self.criterion(W, Y)
                    valid_loss_list.append(loss.item())
            self.scheduler.step(np.mean(valid_loss_list))
            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.min_lr: break # Stop training if the learning rate is below the minimum learning rate.
            tqdm_range.set_description("Train Loss: {:.7f}, Valid Loss: {:.7f}, LR: {:.6f}".format(np.mean(train_loss_list), np.mean(valid_loss_list), current_lr))
            tqdm_range.refresh()
        return self.spectral_net

    def _get_affinity_matrix(self, X: torch.Tensor) -> torch.Tensor: # This function computes the affinity matrix W using the Gaussian kernel.
        # Gaussian kernel: W_ij = exp(-||X_i - X_j||^2 / (2 * sigma^2))
        D = torch.cdist(X, X) # D.shape: (N, N). D = sqrt(||X_i - X_j||^2) (euclidean distance between X_i and X_j).
        nbrs = NearestNeighbors(n_neighbors=self.n_nbg + 1).fit(X.cpu().detach().numpy())
        dis_nbrs, idx_nbrs = nbrs.kneighbors(X.cpu().detach().numpy()) # return the distances and the indices of the k nearest neighbors of each data point
        med = True
        if self.is_local_scale: # Local scale: compute the scale for each data point based on the distances to its k nearest neighbors.
            # 1. original code.
            scale = np.max(dis_nbrs, axis=1) if not med else np.median(dis_nbrs, axis=1) # scale.shape: (n_samples,).
            W = torch.exp(-torch.pow(D, 2) / (torch.tensor(scale, dtype=torch.float32, device=D.device).clamp_min(1e-7) ** 2))
            # 2. scale.shape: (n_samples, 1).
            # scale = scale[:, None] # scale.shape: (n_samples, 1).
            # W = torch.exp(-torch.pow(D, 2) / (torch.tensor(scale, dtype=torch.float32, device=D.device).clamp_min(1e-7) ** 2))
            # 3. scale.shape: (n_samples, n_samples).
            # W = torch.exp(-torch.pow(D, 2) / (torch.tensor(torch.outer(scale, scale), dtype=torch.float32, device=D.device).clamp_min(1e-7) ** 2))
        else: # Global scale: compute the scale for all data points based on the distances to the k-th nearest neighbor.
            scale = np.max(dis_nbrs[:, self.scale_k - 1]) if not med else np.median(dis_nbrs[:, self.scale_k - 1]) # scale.shape: (1,).
            W = torch.exp(-torch.pow(D, 2) / (torch.tensor(scale, dtype=torch.float32, device=D.device).clamp_min(1e-7) ** 2))
        mask = torch.zeros([X.shape[0], X.shape[0]]).to(device=self.device)
        for i in range(len(idx_nbrs)):
            mask[i, idx_nbrs[i]] = 1
        W = W * mask # Sparse affinity matrix. W.shape: (n_samples, n_samples).
        W = (W + W.T) / 2.0 # Symmetric affinity matrix. W.shape: (n_samples, n_samples).
        return W # Affinity matrix. W.shape: (n_samples, n_samples).
    
    def make_annoy_index_for_sparse_graph(self, X: torch.Tensor) -> torch.Tensor:
        # Build the Annoy index for the data X to be used for approximate nearest neighbors search. (For large datasets, Annoy is much faster than KNN.)
        ann_index = AnnoyIndex(X.shape[1], "euclidean") # Initialize the Annoy index by specifying the dimension of the data and the distance metric.
        for i, x_i in enumerate(X):
            ann_index.add_item(i, x_i) # Add the data point to the Annoy index.
        ann_index.build(50) # Build the Annoy index with 50 trees. (The more trees, the more accurate the nearest neighbors search, but the slower the search.)
        ann_index.save("ann_index.ann") # Save the Annoy index to a file.
    
    def make_batch_for_sparse_grapsh(self, batch_x: torch.Tensor) -> torch.Tensor:
        # Computes a new batch of data points from the given batch (batch_x) in case that the graph-laplacian obtained from the given batch is sparse.
        # select 1/5 of the data points from the given batch and compute the 4 nearest neighbors of the selected data points.
        # then, add the 4 nearest neighbors of the selected data points to the new batch.
        batch_size = batch_x.shape[0]; batch_size //= 5
        new_batch_x = batch_x[:batch_size]; batch_x = new_batch_x; n_neighbors = 5
        u = AnnoyIndex(batch_x[0].shape[0], "euclidean")
        u.load("ann_index.ann")
        for x in batch_x:
            nn_indices = u.get_nns_by_vector(x.detach().cpu().numpy(), n_neighbors)
            nn_tensors = [u.get_item_vector(i) for i in nn_indices[1:]]
            nn_tensors = torch.tensor(nn_tensors, device=batch_x.device)
            new_batch_x = torch.cat((new_batch_x, nn_tensors))
        return new_batch_x

########################################################################################################################
# SpectralNet Class: Implementing a Deep learning model that performs spectral clustering. (Optionally utilizes Autoencoders (AE) and Siamese networks for training.)
########################################################################################################################
class SpectralNet: # SpectralNet: implementing a Deep learning model that performs spectral clustering. (Optionally utilizes Autoencoders (AE) and Siamese networks for training.)
    # Autoencoder Network: For dimensionality reduction and feature extraction.
    # Siamese Network: For learning the similarity between data points.
    # Spectral Network: For spectral clustering. (Uses the eigenvectors of the Laplacian matrix of the data to perform clustering.)
    def __init__(
        self,
        n_clusters: int, # The number of clusters to be generated by the SpectralNet algorithm. Also used for the dimention of the projection subspace.
        should_use_ae: bool = False, # Specifies whether to use the Autoencoder (AE) network as part of the training process.
        should_use_siamese: bool = False, # Specifies whether to use the Siamese network as part of the training process.
        is_sparse_graph: bool = False, # Specifies whether the graph Laplacian created from the data is sparse.
        ae_hiddens: list = [512, 512, 2048, 10], # The number of hidden units in each layer of the Autoencoder network.
        ae_epochs: int = 40, # The number of epochs to train the Autoencoder network.
        ae_lr: float = 1e-3, # The learning rate for the Autoencoder network.
        ae_lr_decay: float = 0.1, # The learning rate decay factor for the Autoencoder network.
        ae_min_lr: float = 1e-7, # The minimum learning rate for the Autoencoder network.
        ae_patience: int = 10, # The number of epochs to wait before reducing the learning rate for the Autoencoder network.
        ae_batch_size: int = 256, # The batch size used during training of the Autoencoder network.
        siamese_hiddens: list = [1024, 1024, 512, 10], # The number of hidden units in each layer of the Siamese network.
        siamese_epochs: int = 30, # The number of epochs to train the Siamese network.
        siamese_lr: float = 1e-3, # The learning rate for the Siamese network.
        siamese_lr_decay: float = 0.1, # The learning rate decay factor for the Siamese network.
        siamese_min_lr: float = 1e-7, # The minimum learning rate for the Siamese network.
        siamese_patience: int = 10, # The number of epochs to wait before reducing the learning rate for the Siamese network.
        siamese_n_nbg: int = 2, # The number of nearest neighbors to consider as 'positive' pairs by the Siamese network.
        siamese_use_approx: bool = False, # Specifies whether to use Annoy instead of KNN for computing nearest neighbors, particularly useful for large datasets.
        siamese_batch_size: int = 128, # The batch size used during training of the Siamese network.
        spectral_hiddens: list = [1024, 1024, 512, 10], # The number of hidden units in each layer of the Spectral network.
        spectral_epochs: int = 30, # The number of epochs to train the Spectral network.
        spectral_lr: float = 1e-3, # The learning rate for the Spectral network.
        spectral_lr_decay: float = 0.1, # The learning rate decay factor for the Spectral network.
        spectral_min_lr: float = 1e-8, # The minimum learning rate for the Spectral network.
        spectral_patience: int = 10, # The number of epochs to wait before reducing the learning rate for the Spectral network.
        spectral_batch_size: int = 1024, # The batch size used during training of the Spectral network.
        spectral_n_nbg: int = 30, # The number of nearest neighbors to consider as 'positive' pairs by the Spectral network.
        spectral_scale_k: int = 15, # The number of nearest neighbors to consider as 'positive' pairs by the Spectral network.
        spectral_is_local_scale: bool = True, # Specifies whether the scale is local or global.
    ):
        assert spectral_hiddens[-1] == n_clusters, "The number of units in the last layer of spectral_hiddens network must be equal to the number of clusters or components."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_clusters = n_clusters; self.should_use_ae = should_use_ae; self.should_use_siamese = should_use_siamese; self.is_sparse_graph = is_sparse_graph
        self.ae_config = {"hiddens": ae_hiddens, "epochs": ae_epochs, "lr": ae_lr, "lr_decay": ae_lr_decay, "min_lr": ae_min_lr, "patience": ae_patience, "batch_size": ae_batch_size}
        self.siamese_config = {"hiddens": siamese_hiddens, "epochs": siamese_epochs, "lr": siamese_lr, "lr_decay": siamese_lr_decay, "min_lr": siamese_min_lr, "patience": siamese_patience, "n_nbg": siamese_n_nbg, "use_approx": siamese_use_approx, "batch_size": siamese_batch_size}
        self.spectral_config = {"hiddens": spectral_hiddens, "epochs": spectral_epochs, "lr": spectral_lr, "lr_decay": spectral_lr_decay, "min_lr": spectral_min_lr, "patience": spectral_patience, "n_nbg": spectral_n_nbg, "scale_k": spectral_scale_k, "is_local_scale": spectral_is_local_scale, "batch_size": spectral_batch_size}

    def fit(self, X: torch.Tensor, y: torch.Tensor = None):
        # Performs the main training loop for the SpectralNet model. X: Data to train the networks on. y: Labels in case there are any. Defaults to None.
        # AE for dimensionality reduction and feature extraction.
        # Siamese network for learning the similarity between data points.
        # Spectral network for spectral clustering. (Uses the eigenvectors of the Laplacian matrix of the data to perform clustering.)
        if self.should_use_ae:
            self.ae_trainer = AETrainer(config=self.ae_config, device=self.device)
            self.ae_net = self.ae_trainer.train(X)
            X = self.ae_trainer.embed(X)
        if self.should_use_siamese:
            self.siamese_trainer = SiameseTrainer(config=self.siamese_config, device=self.device)
            self.siamese_net = self.siamese_trainer.train(X)
        self.spectral_trainer = SpectralTrainer(config=self.spectral_config, device=self.device, is_sparse=self.is_sparse_graph)
        self.spec_net = self.spectral_trainer.train(X, y, self.siamese_net if self.should_use_siamese else None)

    def predict(self, X: torch.Tensor) -> np.ndarray: # Predicts the cluster assignments for the given data.
        # For inference, don't need the siamese network because it is only used for training to learn the similarity between data points.
        # AE for dimensionality reduction and feature extraction.
        # Spectral network for spectral clustering. (Uses the eigenvectors of the Laplacian matrix of the data to perform clustering.)
        X = X.view(X.size(0), -1).to(self.device)
        with torch.no_grad():
            if self.should_use_ae:
                X = self.ae_net.encode(X)
            embeddings_ = self.spec_net(X, should_update_orth_weights=False).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10).fit(embeddings_)
        return kmeans.predict(embeddings_), embeddings_

    def visualize_embedding(self, X: torch.Tensor, V: np.ndarray, y: torch.Tensor = None, save_path: str = None) -> None: # Visualize the embeddings of the input data using the fitted SpectralNet model.
        # 1. Visualize the sorted Laplacian and the sorted eigenvectors of the batch.
        # 1.1. Get a batch of the input data and encode it.
        permuted_indices = torch.randperm(X.size(0))[:1024]
        X_encoded = self.ae_trainer.embed(X) if self.should_use_ae else X
        X_encoded = self.siamese_net.forward_once(X_encoded) if self.should_use_siamese else X_encoded
        batch_raw = X[permuted_indices]; batch_encoded = X_encoded[permuted_indices]; batch_y = y[permuted_indices]
        # 1.2. Compute the sparse affinity matrix for the batch using the t similarity function and the k nearest neighbors.
        # 1.2.1. Compute the L2 distance matrix between all data points.
        D = torch.cdist(batch_encoded, batch_encoded) # L2 distance matrix between all data points. D.shape: (1024, 1024).
        # 1.2.2. Compute the t similarity function. t_similarity_func = 1 / (1 + distance^2).
        W = torch.pow(1 + torch.pow(D, 2), -1) # t-similarity matrix. W.shape: (1024, 1024).
        # 1.2.3. Compute the k nearest neighbors of each data point and get the mask.
        nbrs = NearestNeighbors(n_neighbors=self.spectral_config["n_nbg"] + 1).fit(batch_encoded.cpu().detach().numpy())
        dis_nbrs, idx_nbrs = nbrs.kneighbors(batch_encoded.cpu().detach().numpy()) # The distances and indices of the k nearest neighbors.
        # dis_nbrs.shape: (1024, n_neighbors + 1), idx_nbrs.shape: (1024, n_neighbors + 1).
        mask = torch.zeros([1024, 1024]).to(device=self.device) # mask.shape: (1024, 1024).
        for i in range(len(idx_nbrs)):
            mask[i, idx_nbrs[i]] = 1 # Set the elements of the mask to 1 for the k nearest neighbors of each data point.
        # 1.2.4. Get the sparse affinity matrix by masking the t-similarity matrix with the indices of the k nearest neighbors of each data point.
        W = W * mask # Sparse affinity matrix. W.shape: (1024, 1024).
        W = (W + W.T) / 2.0 # Symmetric affinity matrix. sym_W.shape: (1024, 1024).
        # 1.3. Get the unnormalized Laplacian matrix of the batch using the affinity matrix W.
        W = W.detach().cpu().numpy()
        D = np.diag(W.sum(axis=1))
        L_batch = D - W # Unnormalized Laplacian matrix. L_batch.shape: (1024, 1024).
        # 1.4. Get the eigenvectors of the Laplacian of the batch.
        _, V_batch = self.predict(batch_raw) # V_batch.shape: (1024, embedding_dim). eigenvectors of the Laplacian of the batch.
        eigenvalues = np.diag(V_batch.T @ L_batch @ V_batch) # eigenvalues.shape: (embedding_dim,).
        indices = np.argsort(eigenvalues) # Indices of the eigenvectors of the Laplacian of the batch in increasing order of the eigenvalues.
        # 1.5. Plot the block diagonal matrix obtained from the sorted Laplacian. (See whether the sorted Laplacian is a block diagonal matrix.)
        plt.figure(figsize=(10, 10))
        # plt.imshow(L_batch[np.argsort(batch_y), :][:, np.argsort(batch_y)], cmap="hot", norm=colors.LogNorm()) 
        # plt.imshow(L_batch[np.argsort(batch_y), :][:, np.argsort(batch_y)], cmap="hot")
        # plt.imshow(L_batch[np.argsort(batch_y), :][:, np.argsort(batch_y)], cmap="hot", norm=colors.PowerNorm(gamma=2))
        # plt.imshow(L_batch[np.argsort(batch_y), :][:, np.argsort(batch_y)], cmap='hot', norm=colors.BoundaryNorm(np.arange(25, 30, 0.1), ncolors=256))
        plt.imshow(L_batch[np.argsort(batch_y), :][:, np.argsort(batch_y)], cmap="flag") 
        for b in (np.where(np.diff(batch_y[np.argsort(batch_y)]))[0] + 1):
            plt.axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            plt.axhline(y=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.title("Sorted Laplacian (Block Diagonal)", fontsize=14)
        plt.xlabel("Sample index (sorted by label)"); plt.ylabel("Sample index (sorted by label)")
        # plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "sorted_laplacian.png")); plt.close()
        # 1.6. Plot the eigenvectors of the Laplacian when the data is sorted in increasing order by the true label.
        plt.figure(figsize=(12, 6))
        for j in range(V_batch.shape[1]):
            plt.plot(range(len(batch_y)), V_batch[np.argsort(batch_y), j], label=f"Eigvec {j+1} (λ={eigenvalues[indices[j]]:.4f})", linewidth=0.8, alpha=0.85)
        for b in (np.where(np.diff(batch_y[np.argsort(batch_y)]))[0] + 1):
            plt.axvline(x=b, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        plt.title("Sorted Eigenvectors (sorted by label)", fontsize=14)
        plt.xlabel("Sample index (sorted by label)"); plt.ylabel("Eigenvector value")
        # plt.legend(loc='upper right', fontsize=8, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "sorted_eigenvectors.png")); plt.close()
        
        # 2. Plot the embeddings of the whole data.
        # Get the two eigenvectors of the Laplacian of the batch that are most informative for clustering. 
        # The first eigenvector is the constant eigenvector, the second and third eigenvectors are the most informative for clustering.
        V = V[:, [indices[1], indices[2]]]
        plt.figure(figsize=(10, 10))
        plt.scatter(V[:, 0], V[:, 1], c=y, cmap="tab10", s=5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "embeddings.png")); plt.close()
        