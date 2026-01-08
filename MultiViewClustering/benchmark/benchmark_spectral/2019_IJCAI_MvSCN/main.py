import os, gzip, yaml, pickle, random, datetime, itertools, urllib.request
import numpy as np, scipy.io as sio
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from random import randint
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

def random_index(n_all, n_train, seed):
    random.seed(seed)
    idx = random.sample(range(n_all), n_all)
    train_index = np.array(idx[0:n_train])
    test_index = np.array(idx[n_train:n_all])
    return train_index, test_index

def make_numpy_array(data_xy):
    data_x, data_y = data_xy
    data_x = np.asarray(data_x, dtype=np.float64)
    data_y = np.asarray(data_y, dtype=np.int32)
    return data_x, data_y

def _download_file(url, dest_dir="."):
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.basename(url.split("?")[0])
    path = os.path.join(dest_dir, fname)
    if not os.path.exists(path):
        print(f"Downloading {url} to {path}")
        urllib.request.urlretrieve(url, path)
    return path

def load_data(params, view):
    data_dir = './data'
    if params["dset"] == "noisymnist":
        url = "https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view" + str(view) + ".gz"
        path = _download_file(url, data_dir)
        with gzip.open(path, "rb") as f:
            train_set, valid_set, test_set = pickle.load(f)
        train_set_x, train_set_y = make_numpy_array(train_set)
        valid_set_x, valid_set_y = make_numpy_array(valid_set)
        test_set_x, test_set_y = make_numpy_array(test_set)
        train_set_x = np.concatenate((train_set_x, valid_set_x), axis=0)
        train_set_y = np.concatenate((train_set_y, valid_set_y), axis=0)
        return train_set_x, train_set_y, test_set_x, test_set_y
    if params["dset"] == "Caltech101-20":
        os.makedirs(data_dir, exist_ok=True)
        mat_path = os.path.join(data_dir, params["dset"] + ".mat")
        mat = sio.loadmat(mat_path)
        X = mat["X"][0]
        x = X[view - 1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        y = np.squeeze(mat["Y"])
        data_size = x.shape[0]
        train_index, test_index = random_index(data_size, int(data_size * 0.5), 1)
        test_set_x = x[test_index]
        test_set_y = y[test_index]
        train_set_x = x[train_index]
        train_set_y = y[train_index]
        return train_set_x, train_set_y, test_set_x, test_set_y

def get_choices(arr, num_choices, valid_range=(-1, np.inf), not_arr=None, replace=False):
    if not_arr is None:
        not_arr = []
    if isinstance(valid_range, int):
        valid_range = [0, valid_range]
    if isinstance(arr, tuple):
        if min(arr[1], valid_range[1]) - max(arr[0], valid_range[0]) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        n_arr = arr[1]
        arr0 = arr[0]
        arr = defaultdict(lambda: -1)
        get_arr = lambda x: x
        replace = True
    else:
        arr = np.array(arr, copy=True)
        greater_than = arr > valid_range[0]
        less_than = arr < valid_range[1]
        if np.sum(np.logical_and(greater_than, less_than)) < num_choices:
            raise ValueError("Not enough elements in arr are outside of valid_range!")
        n_arr = len(arr)
        arr0 = 0
        get_arr = lambda x: arr[x]
    not_arr_set = set(not_arr)
    if isinstance(not_arr, int):
        not_arr = list(not_arr)
    def get_choice():
        arr_idx = randint(arr0, n_arr - 1)
        while get_arr(arr_idx) in not_arr_set:
            arr_idx = randint(arr0, n_arr - 1)
        return arr_idx
    choices = []
    for _ in range(num_choices):
        arr_idx = get_choice()
        while get_arr(arr_idx) <= valid_range[0] or get_arr(arr_idx) >= valid_range[1]:
            arr_idx = get_choice()
        choices.append(int(get_arr(arr_idx)))
        if not replace:
            arr[arr_idx], arr[n_arr - 1] = arr[n_arr - 1], arr[arr_idx]
            n_arr -= 1
    return choices

def create_pairs_from_unlabeled_data(x1, k=5, tot_pairs=None, verbose=True):
    n = len(x1)
    pairs_per_pt = max(1, min(k, int(tot_pairs / (n * 2)))) if tot_pairs is not None else max(1, k)
    pairs = []
    labels = []
    if len(x1.shape) > 2:
        x1_flat = x1.reshape(x1.shape[0], np.prod(x1.shape[1:]))[:n]
    else:
        x1_flat = x1[:n]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(x1_flat)
    _, Idx = nbrs.kneighbors(x1_flat)
    new_Idx = np.empty((Idx.shape[0], Idx.shape[1] - 1))
    for i in range(Idx.shape[0]):
        new_Idx[i] = Idx[i, Idx[i] != i][: Idx.shape[1] - 1]
    Idx = new_Idx.astype(np.int32)
    k_max = min(Idx.shape[1], k + 1)
    consecutive_fails = 0
    for i in range(n):
        if consecutive_fails > 5:
            k_max = min(Idx.shape[1], int(k_max * 2))
            consecutive_fails = 0
        try:
            choices = get_choices(Idx[i, :k_max], pairs_per_pt, replace=False)
            consecutive_fails = 0
        except ValueError:
            consecutive_fails += 1
            continue
        assert i not in choices
        new_pos = [[x1[i], x1[c]] for c in choices]
        try:
            choices = get_choices((0, n), pairs_per_pt, not_arr=Idx[i, :k_max], replace=False)
            consecutive_fails = 0
        except ValueError:
            consecutive_fails += 1
            continue
        new_neg = [[x1[i], x1[c]] for c in choices]
        labels += [1] * len(new_pos) + [0] * len(new_neg)
        pairs += new_pos + new_neg
    pairs_arr = np.array(pairs).reshape((len(pairs), 2) + x1.shape[1:])
    labels_arr = np.array(labels, dtype=np.float32)
    return pairs_arr, labels_arr

def get_data(params):
    data_list = []
    if params.get("views") is None:
        params["views"] = list(range(1, params["view_size"] + 1))
    for i in params["views"]:
        ret = {}
        x_train, y_train, x_test, y_test = load_data(params, i)
        print("data size (training, testing)", x_train.shape, x_test.shape)
        ret["spectral"] = (x_train, y_train, x_test, y_test)
        pairs_train, dist_train = create_pairs_from_unlabeled_data(x1=x_train, k=params["siam_k"], tot_pairs=params.get("siamese_tot_pairs"))
        ret["siamese"] = (pairs_train, dist_train)
        data_list.append(ret)
    return data_list

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

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix

def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels

def get_y_preds(y_true, cluster_assignments, n_clusters):
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true = get_cluster_labels_from_indices(indices)
    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true[cluster_assignments]
    return y_pred

def classification_metric(y_true, y_pred, average="macro", decimals=4):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    accuracy = np.round(metrics.accuracy_score(y_true, y_pred), decimals)
    precision = np.round(metrics.precision_score(y_true, y_pred, average=average), decimals)
    recall = np.round(metrics.recall_score(y_true, y_pred, average=average), decimals)
    f_score = np.round(metrics.f1_score(y_true, y_pred, average=average), decimals)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f_measure": f_score}, confusion_matrix

def clustering_metric(y_true, y_pred, n_clusters, verbose=True, decimals=4):
    y_pred_adjusted = get_y_preds(y_true, y_pred, n_clusters)
    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_adjusted, decimals=decimals)
    ami = np.round(metrics.adjusted_mutual_info_score(y_true, y_pred), decimals)
    nmi = np.round(metrics.normalized_mutual_info_score(y_true, y_pred), decimals)
    ari = np.round(metrics.adjusted_rand_score(y_true, y_pred), decimals)
    return (dict({"AMI": ami, "NMI": nmi, "ARI": ari}, **classification_metrics), confusion_matrix)

def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args=None):
    if init_args is None:
        init_args = {}
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except Exception:
                pass
        else:
            return np.zeros((len(x),)), cluster_obj
    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj

def clustering(x_list, y, n_clusters=None):
    if n_clusters is None:
        n_clusters = np.size(np.unique(y))
    x_concat = np.concatenate(x_list, axis=1)
    kmeans_assignments, km = get_cluster_sols(x_concat, ClusterClass=KMeans, n_clusters=n_clusters, init_args={"n_init": 10})
    y_preds = get_y_preds(y, kmeans_assignments, n_clusters)
    y_eval = y
    if np.min(y) == 1:
        y_eval = y - 1
    scores, _ = clustering_metric(y_eval, kmeans_assignments, n_clusters)
    return y_preds, scores

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
                    return
        print("Epoch: {}, loss={:.4f}".format(epoch, avg_loss))
    return

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
                    return
        print("Epoch: {}, loss={:.4f}".format(epoch, avg_loss))
    return

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
