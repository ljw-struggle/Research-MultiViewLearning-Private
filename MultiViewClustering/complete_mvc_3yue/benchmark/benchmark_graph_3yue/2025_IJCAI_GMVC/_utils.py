import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np
from torch.utils.data import DataLoader
import scipy.sparse as sp

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner
    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return acc, nmi, ari, pur

def inference(dataset, loader, model, view, data_size, device):
    model.eval()
    Hs = []
    Zs = []
    soft_vector = []
    pred_vectors = []
    for v in range(view):
        Hs.append([])
        Zs.append([])
        pred_vectors.append([])
    labels_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            _, zs, hs, _ = model.forward(xs)
            q = sum(qs) / view
        for v in range(view):
            hs[v] = hs[v].detach()
            zs[v] = zs[v].detach()
            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Hs[v] = np.array(Hs[v])
        Zs[v] = np.array(Zs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return Hs, Zs, total_pred, pred_vectors, labels_vector

def clustering(model, dataset, view, data_size, cluster_num, device):
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    high_level_vectors, low_level_vectors, total_label, pred_label, label_vector = inference(dataset, test_loader, model, view, data_size, device)
    acc2, nmi1, ari1, pur1 = evaluate(label_vector, total_label)
    print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} pur{}={:.4f}'.format(1, acc2, 1, nmi1, 1, ari1, 1, pur1))
    return acc2, nmi1, ari1, pur1

from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

def normalize_features(x):
    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x

def normalize_adj(x):
    # rowsum = np.array(x.sum(1))
    # colsum = np.array(x.sum(0))
    # r_inv = np.power(rowsum, -0.5).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # c_inv = np.power(colsum, -0.5).flatten()
    # c_inv[np.isinf(c_inv)] = 0.
    # r_inv = r_inv.reshape((x.shape[0], -1))
    # c_inv = c_inv.reshape((-1, x.shape[1]))
    # x = x * r_inv * c_inv

    # rowsum = np.array(x.sum(1))
    # x = x + np.diag(rowsum) - np.eye(x.shape[0])
    # rowsum = np.array(x.sum(1))
    # r_inv = np.power(rowsum, -1.).flatten()
    # r_inv[np.isinf(r_inv)] = 0.
    # r_inv = r_inv.reshape((x.shape[0], -1))
    # x = x * r_inv

    rowsum = np.array(x.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((x.shape[0], -1))
    x = x * r_inv
    return x

class DHA(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path)['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path)['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_view(self, new_data):
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)


class Web(Dataset):
    def __init__(self, path):
        print('web')
        x = scipy.io.loadmat(path)
        data1 = scipy.io.loadmat(path)['X'][0,0].astype(np.float32)
        data2 = scipy.io.loadmat(path)['X'][0,1].astype(np.float32)
        labels = scipy.io.loadmat(path)['gnd']
        unique_numbers = len(np.unique(labels))
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_view(self, new_data):
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)


class NGs(Dataset):
    def __init__(self, path):
        print('ngs')
        scaler = MinMaxScaler()
        data1 = scipy.io.loadmat(path)['data'][0,0].astype(np.float32).transpose()
        data2 = scipy.io.loadmat(path)['data'][0,1].astype(np.float32).transpose()
        data3 = scipy.io.loadmat(path)['data'][0,2].astype(np.float32).transpose()
        labels = scipy.io.loadmat(path)['truelabel'][0,0].transpose()
        unique_numbers = len(np.unique(labels))
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]), torch.from_numpy(
            self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def update_view(self, new_data):
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)
        self.x3 = new_data[2].numpy().astype(np.float32)


class ACM(Dataset):
    def __init__(self, path):
        print('acm')
        scaler = MinMaxScaler()
        data1 = scipy.io.loadmat(path)['feature'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['feature'].astype(np.float32)
        view1 = scipy.io.loadmat(path)['PAP'].astype(np.float32)
        view2 = scipy.io.loadmat(path)['PLP'].astype(np.float32)
        labels = scipy.io.loadmat(path)['label']
        labels = np.argmax(labels, axis=1).reshape(3025,1)
        self.x1 = data1
        self.x2 = data2
        self.view1 = view1
        self.view2 = view2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def update_view(self, new_data):
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)
        # self.x3 = new_data[2].numpy().astype(np.float32)

class IMDB(Dataset):
    def __init__(self, path):
        print('imdb')
        scaler = MinMaxScaler()
        data1 = scipy.io.loadmat(path)['feature'].astype(np.float32)
        data2 = scipy.io.loadmat(path)['feature'].astype(np.float32)
        view1 = scipy.io.loadmat(path)['MAM'].astype(np.float32)
        view2 = scipy.io.loadmat(path)['MDM'].astype(np.float32)
        labels = scipy.io.loadmat(path)['label']
        labels = np.argmax(labels, axis=1).reshape(4780, 1)
        self.x1 = data1
        self.x2 = data2
        self.view1 = view1
        self.view2 = view2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def update_view(self, new_data):
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)

class Texas(Dataset):
    def __init__(self, dataset):
        print('texas')
        scaler = MinMaxScaler()
        path = './data/{}/'.format(dataset)
        f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
        data1 = f.astype(np.float32)
        data2 = f.astype(np.float32)
        l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
        labels = l.reshape(183,1)
        struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
        sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                             shape=(183, 183), dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        sadj = sadj + 1 * sp.eye(sadj.shape[0])
        adjs_labels = sadj.todense()
        view1 = adjs_labels.A.astype(np.float32)
        view2 = adjs_labels.A.astype(np.float32)
        self.x1 = data1
        self.x2 = data2
        self.view1 = view1
        self.view2 = view2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def update_view(self, new_data):
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)


class Chameleon(Dataset):
    def __init__(self, dataset):
        print('chameleon')
        scaler = MinMaxScaler()
        path = './data/{}/'.format(dataset)
        f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
        data1 = f.astype(np.float32)
        data2 = f.astype(np.float32)
        l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
        labels = l.reshape(2277,1)
        struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
        sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                             shape=(2277, 2277), dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        sadj = sadj + 1 * sp.eye(sadj.shape[0])
        adjs_labels = sadj.todense()
        view1 = adjs_labels.A.astype(np.float32)
        view2 = adjs_labels.A.astype(np.float32)
        self.x1 = data1
        self.x2 = data2
        self.view1 = view1
        self.view2 = view2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

    def get_adj(self, idx):
        return [torch.from_numpy(self.view1[idx,:][:,idx]), torch.from_numpy(
            self.view2[idx,:][:,idx])]

    def get_graph(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx])]

    def update_view(self, new_data):
        self.x1 = new_data[0].numpy().astype(np.float32)
        self.x2 = new_data[1].numpy().astype(np.float32)

class Caltech6(Dataset):
    def __init__(self, path):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.view6 = scaler.fit_transform(data['X6'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        return ([torch.from_numpy(self.view1[idx]), torch.from_numpy(
            self.view2[idx]), torch.from_numpy(self.view3[idx]), torch.from_numpy(
            self.view6[idx]), torch.from_numpy(self.view5[idx]),  torch.from_numpy(self.view4[idx])],
                torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long())

    def update_view(self, new_data):
        self.view1 = new_data[0].numpy().astype(np.float32)
        self.view2 = new_data[1].numpy().astype(np.float32)
        self.view3 = new_data[2].numpy().astype(np.float32)
        self.view6 = new_data[3].numpy().astype(np.float32)
        self.view5 = new_data[4].numpy().astype(np.float32)
        self.view4 = new_data[5].numpy().astype(np.float32)

def load_data(dataset):
    if dataset == "DHA":
        dataset = DHA('data/DHA.mat'); dims = [110, 6144]; view = 2; data_size = 483; class_num = 23
    elif dataset == "Web":
        dataset = Web('data/WebKB.mat'); dims = [2949, 334]; view = 2; data_size = 1051; class_num = 2
    elif dataset == "NGs":
        dataset = NGs('data/NGs.mat'); dims = [2000,2000,2000]; view = 3; data_size = 500; class_num = 5
    elif dataset == "acm":
        dataset = ACM('data/acm.mat'); dims = [1870,1870]; view = 2; data_size = 3025; class_num = 3
    elif dataset == 'imdb':
        dataset = IMDB('data/imdb5k.mat'); dims = [1232, 1232]; view = 2; data_size = 4780; class_num = 3
    elif dataset == 'texas':
        dataset = Texas('texas'); dims = [1703, 1703]; view = 2; data_size = 183; class_num = 5
    elif dataset == 'chameleon':
        dataset = Chameleon('chameleon'); dims = [2325, 2325]; view = 2; data_size = 2277; class_num = 5
    elif dataset == 'Caltech6':
        dataset = Caltech6('data/Caltech.mat'); dims = [48, 40, 254, 928, 512, 1984]; view = 6; data_size = 1400; class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

import torch
import numpy as np
import scipy.sparse as sp
import random as random
from torch.nn.functional import normalize

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed:',seed)

"""
Calculation of distance matrix in the formula
"""
def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1) # [B, N, f]
    A = -2 * torch.bmm(y, x) # [B, N, N]
    A += torch.sum(y**2, dim=2, keepdim=True) # [B, N, 1]
    A += torch.sum(x**2, dim=1, keepdim=True) # [B, 1, N]
    return A.squeeze()

"""
KNN adjacency matrix calculation
"""
def create_sparse(I):
    similar = I.reshape(-1).tolist()
    index = np.repeat(range(I.shape[0]), I.shape[1])
    assert len(similar) == len(index)
    indices = torch.tensor([index, similar]).cuda()
    result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)),
                                     [I.shape[0], I.shape[0]])
    return result

"""
AGC module for multi-view data
"""
def node_filter_adaptive_i(dataset, view, K, reg_zero, beta, alpha, verbose=False):
    adj = []
    new_feat = []
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)
    for v in range(view):
        # Calculate matrix A^v through closed form solution
        D = pairwise_distance(xs[v])  # Calculate distance matrix D
        D = normalize(D)
        res = torch.mm(xs[v], torch.transpose(xs[v], 0, 1))  # Calculate H*H.T
        inv = torch.inverse(res + beta * torch.eye(xs[v].shape[0]))  # # Inverse matrix in A^v
        front = res - alpha/2 * D  # The first part of A
        S = torch.mm(front, inv)  # obtain A^v
        S = torch.where(S > 0, torch.sign(S), torch.tensor(0.0))  # Eq.(7) If S > 0, the value is 1
        adj.append(S)
        n, f = xs[v].shape
        adj[v] = adj[v].cpu().numpy()
        # S = D^(-1/2) A D^(-1/2)
        deg = np.array(adj[v].sum(axis=1)).flatten()
        if 0 in deg:
            if verbose:
                print("Added self-loops where degree was 0.")
            idcs = np.argwhere(deg == 0).flatten()
            vec = np.zeros(n)
            vec[idcs] = 1
            deg[idcs] = 1.
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ (adj[v] + sp.spdiags(vec, 0, n, n))
        else:
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ adj[v]
        feats_props = np.empty((K + 1, n, f))  # Initialize T
        feats_props[0, :, :] = xs[v].cpu().numpy()
        for i in range(1, K + 1):
            feats_props[i, :, :] = (adj_norm + 1 * np.eye(n)) @ feats_props[i - 1, :, :]  # Calculate T by Eq.(8)
        coeffs = np.empty((n, K + 1))  # Initialize w
        feats_filtered = np.empty((n, f))  # Initialize E
        feats_props = np.transpose(feats_props, (2, 1, 0))
        reg_vec = np.zeros(K + 1);
        reg_zero = np.sqrt(n) * reg_zero
        reg_vec[0] = np.sqrt(n * reg_zero)
        for node_idx in range(n):
            coeffs[node_idx, :], _, _, _ = np.linalg.lstsq(np.vstack((feats_props[:, node_idx, :], reg_vec[None, :])),
                                                           np.append(feats_props[:, node_idx, 0], np.zeros(1)),
                                                           rcond=None)  # Solve the w by Eq.(9)
            feats_filtered[node_idx, :] = feats_props[:, node_idx, :] @ coeffs[node_idx, :]  # Calculate E by Eq.(10)
            if verbose:
                print("Finished node %i of %i." % (node_idx, n))
        feats_filtered = torch.tensor(feats_filtered, dtype=torch.float32)
        new_feat.append(feats_filtered)
    dataset.update_view(new_feat)
    print('node filter finish')

"""
AGC module for multi-graph data
"""
def node_filter_adaptive_g(dataset, view, K, reg_zero, verbose=False):
    print('graph')
    adj = []
    new_feat = []
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)
    graph = dataset.get_graph(index)
    for v in range(view):
        adj.append(graph[v])
        n, f = xs[v].shape
        adj[v] = adj[v].cpu().numpy()
        # S = D^(-1/2) A D^(-1/2)
        deg = np.array(adj[v].sum(axis=1)).flatten()
        if 0 in deg:
            if verbose:
                print("Added self-loops where degree was 0.")
            idcs = np.argwhere(deg == 0).flatten()
            vec = np.zeros(n)
            vec[idcs] = 1
            deg[idcs] = 1.
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ (adj[v] + sp.spdiags(vec, 0, n, n))
        else:
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ adj[v]
        feats_props = np.empty((K + 1, n, f))  # Initialize T
        feats_props[0, :, :] = xs[v].cpu().numpy()
        for i in range(1, K + 1):
            feats_props[i, :, :] = (adj_norm + 1 * np.eye(n)) @ feats_props[i - 1, :, :]  # Calculate T by Eq.(8)
        coeffs = np.empty((n, K + 1))  # Initialize w
        feats_filtered = np.empty((n, f))  # Initialize E
        feats_props = np.transpose(feats_props, (2, 1, 0))
        reg_vec = np.zeros(K + 1);
        reg_zero = np.sqrt(n) * reg_zero
        reg_vec[0] = np.sqrt(n * reg_zero)
        for node_idx in range(n):
            coeffs[node_idx, :], _, _, _ = np.linalg.lstsq(np.vstack((feats_props[:, node_idx, :], reg_vec[None, :])),
                                                           np.append(feats_props[:, node_idx, 0], np.zeros(1)),
                                                           rcond=None)  # Solve the w by Eq.(9)
            feats_filtered[node_idx, :] = feats_props[:, node_idx, :] @ coeffs[node_idx, :]  # Calculate E by Eq.(10)
            if verbose:
                print("Finished node %i of %i." % (node_idx, n))
        feats_filtered = torch.tensor(feats_filtered, dtype=torch.float32)
        new_feat.append(feats_filtered)
    dataset.update_view(new_feat)
    print('node filter finish')
