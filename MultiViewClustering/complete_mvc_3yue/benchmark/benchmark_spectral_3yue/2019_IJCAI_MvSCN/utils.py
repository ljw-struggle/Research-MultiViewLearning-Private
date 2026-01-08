import os, gzip, yaml, pickle, random, datetime, itertools, urllib.request
import numpy as np, scipy.io as sio
from collections import defaultdict
from random import randint
from munkres import Munkres
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

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
