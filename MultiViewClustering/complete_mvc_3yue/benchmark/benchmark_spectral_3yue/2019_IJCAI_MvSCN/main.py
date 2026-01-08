import os, gzip, yaml, pickle, random, datetime, itertools, urllib.request
import numpy as np, scipy.io as sio
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from utils import clustering, get_data

def contrastive_loss(y_pred, y_true, m_neg=1.0, m_pos=0.05):
    pos = y_true * F.relu(y_pred - m_pos).pow(2)
    neg = (1 - y_true) * F.relu(m_neg - y_pred).pow(2)
    return (pos + neg).mean()

def squared_distance(X, Y=None, W=None):
    if Y is None:
        Y = X
    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    D = (diff * diff).sum(dim=-1)
    if W is not None:
        D_diag = W.sum(dim=1).sqrt().unsqueeze(1)
        X_n = X / (D_diag + 1e-8)
        Y_n = Y / (D_diag + 1e-8)
        diff = X_n.unsqueeze(1) - Y_n.unsqueeze(0)
        D = (diff * diff).sum(dim=-1)
    return D

def pairwise_distance(X, Y):
    return ((X - Y) ** 2).sum()

def knn_affinity(X, n_nbrs, scale=None, scale_nbr=None, local_scale=True):
    n_nbrs = int(n_nbrs)
    Dx = squared_distance(X)
    k = min(n_nbrs, Dx.shape[0])
    vals, nn_idx = torch.topk(-Dx, k, dim=1, sorted=True)
    vals = -vals
    if scale is None:
        if scale_nbr is not None and scale_nbr > 0 and scale_nbr <= k:
            if local_scale:
                scale = vals[:, scale_nbr - 1].unsqueeze(1).clamp(min=1e-8)
                vals = vals / (2 * scale)
            else:
                med = torch.median(vals[:, scale_nbr - 1]).clamp(min=1e-8)
                vals = vals / (2 * med)
        else:
            med = torch.median(vals[:, -1]).clamp(min=1e-8)
            vals = vals / (2 * med)
    else:
        scale = float(scale)
        vals = vals / (2 * scale ** 2)
    aff_vals = torch.exp(vals)
    N = X.shape[0]
    W = torch.zeros(N, N, device=X.device, dtype=X.dtype)
    W.scatter_(1, nn_idx, aff_vals)
    W = (W + W.T) / 2.0
    return W

def get_scale(x, batch_size, n_nbrs):
    n = len(x)
    if n <= 0:
        return 1.0
    sample_size = min(n, batch_size)
    idx = np.random.randint(0, n, size=sample_size)
    sample = x[idx]
    if sample.ndim > 2:
        sample = sample.reshape(sample_size, -1)
    n_nbrs = min(n_nbrs, sample_size - 1)
    if n_nbrs < 1:
        return 1.0
    nbrs = NearestNeighbors(n_neighbors=n_nbrs + 1).fit(sample)
    distances, _ = nbrs.kneighbors(sample)
    return float(np.median(distances[:, n_nbrs]))

class Orthogonal(nn.Module):
    def __init__(self, dim, epsilon=1e-4):
        super().__init__()
        self.dim = dim
        self.epsilon = epsilon
        self.register_buffer("_ortho_buf", None)

    def forward(self, x):
        n, d = x.shape
        if self.training:
            C = x.T @ x + self.epsilon * torch.eye(d, device=x.device, dtype=x.dtype)
            L = torch.linalg.cholesky(C)
            LinvT = torch.inverse(L).T
            out = (x @ LinvT) * (n ** 0.5)
            self._ortho_buf = LinvT.detach()
            return out
        else:
            if self._ortho_buf is None:
                C = x.T @ x + self.epsilon * torch.eye(d, device=x.device, dtype=x.dtype)
                L = torch.linalg.cholesky(C)
                LinvT = torch.inverse(L).T
                self._ortho_buf = LinvT
            out = (x @ self._ortho_buf) * (n ** 0.5)
            return out

def _act(name):
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softmax":
        return nn.Softmax(dim=-1)
    if name == "softplus":
        return nn.Softplus()
    if name == "selu":
        return nn.SELU()
    raise ValueError("Unknown activation '{}'".format(name))

def build_mlp(arch, input_dim, dropout=0.0, l2_reg=None):
    layers = []
    in_dim = input_dim
    for i, a in enumerate(arch):
        size = a["size"]
        layer_type = a["type"]
        layers.append(nn.Linear(in_dim, size))
        if layer_type != "linear":
            layers.append(_act(layer_type))
        if dropout and layer_type != "Flatten":
            layers.append(nn.Dropout(dropout))
        in_dim = size
    return nn.Sequential(*layers)

def build_mlp_with_orthogonal(arch, input_dim, n_clusters, dropout=0.0):
    hidden_arch = arch[:-1]
    layers = []
    in_dim = input_dim
    for a in hidden_arch:
        size = a["size"]
        layer_type = a["type"]
        layers.append(nn.Linear(in_dim, size))
        layers.append(_act(layer_type))
        if dropout:
            layers.append(nn.Dropout(dropout))
        in_dim = size
    layers.append(nn.Linear(in_dim, n_clusters))
    layers.append(nn.Tanh())
    seq = nn.Sequential(*layers)
    ortho = Orthogonal(n_clusters)
    return seq, ortho

class SiameseNet(nn.Module):
    def __init__(self, input_dim, arch, dropout=0.0):
        super().__init__()
        self.mlp = build_mlp(arch, input_dim, dropout=dropout)
        self.input_dim = input_dim

    def embed(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.mlp(x)

    def forward(self, x1, x2):
        e1 = self.embed(x1)
        e2 = self.embed(x2)
        d = (e1 - e2).pow(2).sum(dim=1).clamp(min=1e-8).sqrt()
        return d

class MvSCN(nn.Module):
    def __init__(self, input_dims, arch, n_clusters, view_size, dropout=0.0):
        super().__init__()
        self.view_size = view_size
        self.n_clusters = n_clusters
        self.towers = nn.ModuleList()
        self.orthogonals = nn.ModuleList()
        for i in range(view_size):
            seq, ortho = build_mlp_with_orthogonal(arch, input_dims[i], n_clusters, dropout=dropout)
            self.towers.append(seq)
            self.orthogonals.append(ortho)

    def forward(self, x_list):
        out = []
        for i in range(self.view_size):
            x = x_list[i]
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            h = self.towers[i](x)
            out.append(self.orthogonals[i](h))
        return out

    def predict(self, x_list, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        n = len(x_list[0])
        out_list = [[] for _ in range(self.view_size)]
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_batch = [torch.from_numpy(x_list[i][start:end]).float().to(device) for i in range(self.view_size)]
                pred = self.forward(x_batch)
                for i in range(self.view_size):
                    out_list[i].append(pred[i].cpu().numpy())
        return [np.concatenate(out_list[i], axis=0) for i in range(self.view_size)]

def train_siamese(model, pairs_train, dist_train, lr=0.0001, drop=0.1, patience=15, num_epochs=400, batch_size=128, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler_stage = 0
    best_loss = float("inf")
    wait = 0
    x1 = torch.from_numpy(pairs_train[:, 0]).float()
    x2 = torch.from_numpy(pairs_train[:, 1]).float()
    y = torch.from_numpy(dist_train).float().unsqueeze(1)
    dataset = TensorDataset(x1, x2, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x1_b, x2_b, y_b in loader:
            x1_b = x1_b.to(device)
            x2_b = x2_b.to(device)
            y_b = y_b.squeeze(1).to(device)
            optimizer.zero_grad()
            pred = model(x1_b, x2_b)
            loss = contrastive_loss(pred, y_b, m_neg=1.0, m_pos=0.05)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss <= best_loss:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait > patience:
                scheduler_stage += 1
                wait = 0
                for g in optimizer.param_groups:
                    g["lr"] = lr * (drop ** scheduler_stage)
                if g["lr"] <= 1e-8:
                    print("STOPPING EARLY (lr too small)")
                    break
        print("Epoch: {}, loss={:.4f}".format(epoch, avg_loss))

def train_mvscn(model, siamese_list, x_train_list, n_nbrs, scale_nbr, batch_size, lamb, lr=0.0001, drop=0.1, patience=15, num_epochs=400, batches_per_epoch=100, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for s in siamese_list:
        s.to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler_stage = 0
    best_loss = float("inf")
    wait = 0
    view_size = model.view_size
    n = len(x_train_list[0])
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            bs = min(batch_size, n)
            if bs < 2:
                continue
            batch_ids = np.random.choice(n, size=bs, replace=False)
            x_batch = [torch.from_numpy(x_train_list[i][batch_ids]).float().to(device) for i in range(view_size)]
            with torch.no_grad():
                aff_list = []
                for v in range(view_size):
                    emb = siamese_list[v].embed(x_batch[v])
                    emb_np = emb.detach().cpu().numpy()
                    scale = get_scale(emb_np, bs, scale_nbr)
                    W = knn_affinity(emb, n_nbrs, scale=scale, scale_nbr=scale_nbr, local_scale=True)
                    aff_list.append(W)
            out_list = model(x_batch)
            loss_1 = 0.0
            for v in range(view_size):
                Dy = squared_distance(out_list[v])
                loss_1 = loss_1 + (aff_list[v] * Dy).sum() / (bs ** 2)
            loss_1 = loss_1 / view_size
            loss_2 = 0.0
            for i, j in itertools.combinations(range(view_size), 2):
                loss_2 = loss_2 + pairwise_distance(out_list[i], out_list[j]) / bs
            loss_2 = loss_2 / (view_size ** 2)
            loss = (1 - lamb) * loss_1 + lamb * loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / batches_per_epoch
        if avg_loss <= best_loss:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait > patience:
                scheduler_stage += 1
                wait = 0
                for g in optimizer.param_groups:
                    g["lr"] = lr * (drop ** scheduler_stage)
                if g["lr"] <= 1e-8:
                    print("STOPPING EARLY")
                    break
        print("Epoch: {}, loss={:.4f}".format(epoch, avg_loss))

def load_config(config_name):
    if ".yaml" not in config_name:
        config_name = config_name + ".yaml"
    root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root, "config", config_name)
    if not os.path.exists(config_path):
        config_path = config_name
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["experiment_id"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return cfg

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = load_config("noisymnist")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = get_data(config)
    view_size = config["view_size"]
    n_clusters = config["n_clusters"]
    batch_sizes = {"Embedding": config["batch_size"], "Orthogonal": config["batch_size_orthogonal"]}
    x_train_list = []
    x_test_list = []
    _, y_train, _, y_test = data_list[0]["spectral"]
    siamese_list = []
    input_dims = []
    for i in range(view_size):
        x_train_i, _, x_test_i, _ = data_list[i]["spectral"]
        x_train_list.append(x_train_i)
        x_test_list.append(x_test_i)
        dim = int(np.prod(x_train_i.shape[1:]))
        input_dims.append(dim)
        pairs_train, dist_train = data_list[i]["siamese"]
        siam = SiameseNet(input_dim=dim, arch=config["arch"], dropout=0.0)
        train_siamese(siam, pairs_train, dist_train, lr=config["siam_lr"], drop=config["siam_drop"], patience=config["siam_patience"], num_epochs=config["siam_epoch"], batch_size=config["siam_batch_size"], device=device)
        siamese_list.append(siam)
    mvscn = MvSCN(input_dims=input_dims, arch=config["arch"], n_clusters=n_clusters, view_size=view_size, dropout=0.0)
    train_mvscn(mvscn, siamese_list, x_train_list, n_nbrs=config["n_nbrs"], scale_nbr=config["scale_nbr"], batch_size=batch_sizes["Embedding"], lamb=config["lamb"], lr=config["spectral_lr"], drop=config["spectral_drop"], patience=config["spectral_patience"], num_epochs=config["spectral_epoch"], batches_per_epoch=100, device=device)
    x_test_final_list = mvscn.predict(x_test_list, batch_size=batch_sizes["Embedding"], device=device)
    y_preds, scores = clustering(x_test_final_list, y_test, n_clusters=n_clusters)
    print("Clustering scores:", scores)
