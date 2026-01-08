from torch.utils.data import Dataset, Sampler
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
            data.append(torch.tensor(self.data_list[i][idx].astype('float32')))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


def load_data(name):
    """
    :param name: name of dataset
    :return:
    data_list: python list containing all views, where each view is represented as numpy array
    labels: ground_truth labels represented as numpy array
    dims: python list containing dimension of each view
    num_views: number of views
    data_size: size of data
    class_num: number of category
    """
    data_path = "./data/"
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

    return data_list, labels, dims, num_views, data_size, class_num


class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


from sklearn.metrics import v_measure_score, accuracy_score
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
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)

    return acc, nmi, pur


import numpy as np
from numpy.random import randint
import random
import math


def get_mask(view_num, data_size, missing_ratio):
    """
    :param view_num: number of views
    :param data_size: size of data
    :param missing_ratio: missing ratio
    :return: mask matrix
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_size*missing_ratio)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, view_num])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = (rand_v >= v_threshold)
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
                break
        mask[miss_ind[j]] = rand_v

    return mask
