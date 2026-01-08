from torch.utils.data import Dataset
import numpy as np
import torch


class MultiviewDataset(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            data.append(torch.tensor(self.data_list[i][idx]))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


class MultiviewDataset2(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            x = torch.tensor(self.data_list[i][idx])
            data.append(x.view(x.size()[0], 28, 28))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


def load_data(name):
    data_path = './data/'
    dataset_names = ['caltech_5m', 'uci', 'rgbd', 'voc', 'mnist_mv']
    if name in dataset_names:
        path = data_path + name + '.npz'
        data = np.load(path)
        num_views = int(data['n_views'])
        data_list = []
        for i in range(num_views):
            x = data[f"view_{i}"]
            if len(x.shape) > 2:
                x = x.reshape([x.shape[0], -1])
            data_list.append(x.astype(np.float32))
        labels = data['labels']
        dims = []
        for i in range(num_views):
            dims.append(data_list[i].shape[1])
        class_num = labels.max() + 1
        data_size = data_list[0].shape[0]
        dataset = MultiviewDataset(num_views, data_list, labels)
        return dataset, dims, num_views, data_size, class_num
    elif name == 'mnist_mv' or 'fmnist':
        path = data_path + name + '.npz'
        data = np.load(path)
        num_views = int(data['n_views'])
        data_list = []
        for i in range(num_views):
            x = data[f"view_{i}"]
            data_list.append(x)
        labels = data['labels']
        dims = []
        for i in range(num_views):
            dims.append(data_list[i].shape[1])
        class_num = labels.max() + 1
        data_size = data_list[0].shape[0]
        dataset = MultiviewDataset2(num_views, data_list, labels)
        return dataset, dims, num_views, data_size, class_num
    else:
        raise NotImplementedError

from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
import numpy as np


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
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur
