import random, numpy as np, scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

class BDGP_Caltech7(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.view1 = MinMaxScaler().fit_transform(data['X1'].T.astype(np.float32).transpose())
        self.view2 = MinMaxScaler().fit_transform(data['X2'].T.astype(np.float32).transpose())
        self.view3 = MinMaxScaler().fit_transform(data['X3'].T.astype(np.float32).transpose())
        self.view4 = MinMaxScaler().fit_transform(data['X4'].T.astype(np.float32).transpose())
        self.view5 = MinMaxScaler().fit_transform(data['X5'].T.astype(np.float32).transpose())
        self.views = [self.view1, self.view2, self.view3, self.view4, self.view5]
        self.labels = sio.loadmat(path)['Y'].T

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(self.view3[idx]) ,torch.from_numpy(self.view4[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

    def get_view(self, idx):
        if idx == -1:
            return self.labels
        else:
            return self.views[idx]

def load_data(dataset):
    if dataset == "Caltech7":
        dataset = BDGP_Caltech7('./data/Caltech-5V-7.mat')
        dims = [40, 254, 1984, 512, 928]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    dataset, ts = Form_Unaligned_Data(dataset, view)
    return dataset, dims, view, data_size, class_num, ts

def Form_Unaligned_Data(dataset, view):
    X = []
    Y = []
    for i in range(view):
        X.append(dataset.get_view(i))
        Y.append(dataset.get_view(-1))
    size = len(Y[0])
    view_num = len(X)
    t = np.linspace(0, size - 1, size, dtype=int)
    random.shuffle(t)
    Xtmp = []
    Ytmp = []
    for i in range(view_num):
        xtmp = np.copy(X[i])
        Xtmp.append(xtmp)
        ytmp = np.copy(Y[i])
        Ytmp.append(ytmp)
    ts = []
    for v in range(view_num):
        random.shuffle(t)
        ts.append(t)
        Xtmp[v][:] = X[v][t]
        Ytmp[v][:] = Y[v][t]
    X = Xtmp
    Y = Ytmp
    result = GN_Dataset(X, Y, view)
    return result, ts

class GN_Dataset(Dataset):
    def __init__(self, dataset, y, view_num):
        self.dataset = dataset
        self.labels = y
        self.view_num = view_num

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):
        X = []; Y = []
        for i in range(self.view_num):
            data_tensor = torch.from_numpy(self.dataset[i][index].astype(np.float32).transpose())
            label = self.labels[i][index]
            label_tensor = torch.tensor(label)
            X.append(data_tensor); Y.append(label_tensor)
        return X, Y, index

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
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
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
    return nmi, ari, acc, pur

def activate_and_normalize(tensor):
    tensor = torch.clamp(tensor, min=0)
    row_sums = tensor.sum(dim=1, keepdim=True)
    row_sums = row_sums + (row_sums == 0).float()
    tensor = tensor / row_sums
    return tensor

def inference(loader, model, device, view, data_size, class_num, max_view):
    Hs = []; Zs = []; labels_vector_multi = []
    for v in range(view):
        Hs.append([]); Zs.append([]); labels_vector_multi.append([])
    labels_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, _, zs, zs_pre, zs_pre_align, hs_align = model.forward(xs, max_view)
        for v in range(view):
            hs[v] = hs[v].detach()
            zs[v] = zs_pre_align[v].detach()
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
            labels_vector_multi[v].extend(y[v].numpy())
        labels_vector.extend(y[max_view].numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    H_avg = np.array(Hs[max_view])
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    total_pred_h = kmeans.fit_predict(H_avg)
    return total_pred_h, labels_vector, H_avg

def valid(model, device, dataset, view, data_size, class_num, max_view, epoch, ts):
    test_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    total_pred_h, labels_vector, H_avg = inference(test_loader, model, device, view, data_size, class_num, max_view)
    print("Clustering results on H: " + str(labels_vector.shape[0]))
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred_h)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur, ari
