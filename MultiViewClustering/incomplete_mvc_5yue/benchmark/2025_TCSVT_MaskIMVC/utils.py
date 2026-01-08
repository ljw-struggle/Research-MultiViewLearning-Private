import numpy as np
import scipy.io as sio
from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def get_mask(view_num, data_len, missing_rate, mask_seed):
    np.random.seed(mask_seed)
    """Randomly generate incomplete multi-view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: e.g., 0.1, 0.3, 0.5, 0.7
        Returns:
          indicator matrix A

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int32)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int32))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int32)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int32)
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix    # indicator matrix A

def get_data(data_name, miss_rate, mask_seed):
    # load the datasets
    data_path = "./datasets/" + data_name + ".mat"
    data = sio.loadmat(data_path)
    if data['fea'].shape[1] < data['fea'].shape[0]:
        data_x = data['fea'][0]
    else:
        data_x = data['fea'][:][0]
    data_y = data['gt'].flatten()
    # get the basic information of the datasets
    view_num = data_x.shape[0]
    sample_num = data_x[0].shape[0]
    cluster_num = len(np.unique(data_y))
    input_dims = [data_x[v].shape[1] for v in range(view_num)]
    # generate missing or observed mask
    random_sequence = np.random.permutation(sample_num)
    mask = get_mask(view_num, sample_num, miss_rate, mask_seed)
    # random permutation the sample orders
    for v in range(view_num):
        data_x[v] = data_x[v][random_sequence]
    data_y = data_y[random_sequence]
    mask = mask[random_sequence]
    # normalize the datasets
    for v in range(view_num):
        pipeline = MinMaxScaler()
        data_x[v] = pipeline.fit_transform(data_x[v])
    print(
        f"Data: {data_name},"
        f" number of data: {sample_num},"
        f" views: {view_num},"
        f" clusters: {cluster_num},"
        f" dims of each view: {input_dims}")

    return mask, data_x, data_y, view_num, sample_num, cluster_num, input_dims

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score
import warnings
from scipy import sparse as sp
from scipy.special import comb

warnings.filterwarnings("ignore")


def bestMap(y_pred, y_true):
    from scipy.optimize import linear_sum_assignment
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    np.asarray(ind)
    ind = np.transpose(ind)
    label = np.zeros(y_pred.size)
    for i in range(y_pred.size):
        label[i] = ind[y_pred[i]][1]
    return label.astype(np.int64)


def similarity_function(points):
    """

    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    indx_list = []
    for i in range(len(ind[0])):
        indx_list.append((ind[0][i], ind[1][i]))
    return sum([w[i1, j1] for (i1, j1) in indx_list]) * 1.0 / y_pred.size


def cluster_f(y_true, y_pred):
    N = len(y_true)
    numT = 0
    numH = 0
    numI = 0
    for n in range(0, N):
        C1 = [y_true[n] for x in range(1, N - n)]
        C1 = np.array(C1)
        C2 = y_true[n + 1:]
        C2 = np.array(C2)
        Tn = (C1 == C2) * 1

        C3 = [y_pred[n] for x in range(1, N - n)]
        C3 = np.array(C3)
        C4 = y_pred[n + 1:]
        C4 = np.array(C4)
        Hn = (C3 == C4) * 1

        numT = numT + np.sum(Tn)
        numH = numH + np.sum(Hn)
        numI = numI + np.sum(np.multiply(Tn, Hn))
    if numH > 0:
        p = numI / numH
    if numT > 0:
        r = numI / numT
    if (p + r) == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f, p, r


def clustering_purity(labels_true, labels_pred):
    """
    :param y_true:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]
    :param y_pred:
        data type: numpy.ndarray
        shape: (n_samples,)
        sample: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    :return: Purity
    """
    y_true = labels_true.copy()
    y_pred = labels_pred.copy()
    if y_true.shape[1] != 1:
        y_true = y_true.T
    if y_pred.shape[1] != 1:
        y_pred = y_pred.T

    n_samples = len(y_true)

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    y_true_temp = np.zeros((n_samples, 1))
    if n_true_classes != max(y_true):
        for i in range(n_true_classes):
            y_true_temp[np.where(y_true == u_y_true[i])] = i + 1
        y_true = y_true_temp

    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)
    y_pred_temp = np.zeros((n_samples, 1))
    if n_pred_classes != max(y_pred):
        for i in range(n_pred_classes):
            y_pred_temp[np.where(y_pred == u_y_pred[i])] = i + 1
        y_pred = y_pred_temp

    u_y_true = np.unique(y_true)
    n_true_classes = len(u_y_true)
    u_y_pred = np.unique(y_pred)
    n_pred_classes = len(u_y_pred)

    n_correct = 0
    for i in range(n_pred_classes):
        incluster = y_true[np.where(y_pred == u_y_pred[i])]

        inclunub = np.histogram(incluster, bins=range(1, int(max(incluster)) + 1))[0]
        if len(inclunub) != 0:
            n_correct = n_correct + max(inclunub)

    Purity = n_correct / len(y_pred)

    return Purity


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_pred = bestMap(labels_pred, labels_true)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return f_score, precision, recall


# Evaluation metrics of clustering performance
def clusteringMetrics(trueLabel, predictiveLabel):
    y_pred = bestMap(predictiveLabel, trueLabel)
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, y_pred)
    # NMI
    NMI = normalized_mutual_info_score(trueLabel, y_pred)
    # Purity
    Purity = clustering_purity(trueLabel.reshape((-1, 1)), y_pred.reshape(-1, 1))
    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, y_pred)
    # Fscore, Precision, Recall = cluster_f(trueLabel, y_pred)
    Fscore, Precision, Recall = b3_precision_recall_fscore(trueLabel, y_pred)

    return ACC, NMI, Purity, ARI, Fscore, Precision, Recall


def get_clustering_performance(features, y_label, cluster_num, random_numbers_for_kmeans):
    # Initialize lists to store performance metrics
    metrics = {
        "ACC": [], "NMI": [], "Purity": [],
        "ARI": [], "Fscore": [], "Precision": [], "Recall": []
    }
    for random_state in random_numbers_for_kmeans:
        # Create and fit KMeans model
        kmeans = KMeans(n_clusters=cluster_num, n_init=10, random_state=random_state)
        y_predict = kmeans.fit_predict(features)
        # Calculate clustering metrics
        ACC, NMI, Purity, ARI, Fscore, Precision, Recall = clusteringMetrics(y_label, y_predict)
        # Append metrics to respective lists
        metrics["ACC"].append(ACC)
        metrics["NMI"].append(NMI)
        metrics["Purity"].append(Purity)
        metrics["ARI"].append(ARI)
        metrics["Fscore"].append(Fscore)
        metrics["Precision"].append(Precision)
        metrics["Recall"].append(Recall)
    # Calculate average performance metrics
    average_metrics = {key: np.mean(values) * 100 for key, values in metrics.items()}
    std_metrics = {key: np.std(values) * 100 for key, values in metrics.items()}

    return average_metrics

import os
import pandas as pd

class BaseLogger:
    def __init__(self, log_save_dir, log_name, params=None, decimal_places=2):
        self.log_save_dir = log_save_dir
        self.log_name = log_name
        self.params = params
        self.log_file_loc = os.path.join(log_save_dir, log_name)
        self.decimal_places = decimal_places
        if os.path.isfile(self.log_file_loc):
            self.log_data = pd.read_csv(self.log_file_loc)
        else:
            self.log_data = pd.DataFrame(columns=['PARAM', 'Epoch', 'ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall'])
            self._save_to_csv()

        os.makedirs(log_save_dir, exist_ok=True)

    def close_logger(self):
        self._save_to_csv()
        print(f"Log saved to {self.log_file_loc}")

    def write_parameters(self, parameters):
        print("\nThe parameters: %.3f" % parameters)  # 控制参数的小数位数，若需要可调整
        self.params = parameters
        # param_row = pd.DataFrame({'PARAM': [ self.params], 'Epoch': [None], 'ACC': [None], 'NMI': [None], 'Purity': [None],
        #                           'ARI': [None], 'Fscore': [None], 'Precision': [None], 'Recall': [None]})
        # self.log_data = pd.concat([self.log_data, param_row], ignore_index=True)
        # self._save_to_csv()

    def write_val(self, epoch, loss_tr, scores):
        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.6f\t Acc = %.4f" % (epoch, loss_tr, scores[0]))
        new_row = pd.DataFrame({
            'PARAM': [ self.params],
            'Epoch': [epoch],
            'ACC': [round(scores[0], self.decimal_places)],
            'NMI': [round(scores[1], self.decimal_places)],
            'Purity': [round(scores[2], self.decimal_places)],
            'ARI': [round(scores[3], self.decimal_places)],
            'Fscore': [round(scores[4], self.decimal_places)],
            'Precision': [round(scores[5], self.decimal_places)],
            'Recall': [round(scores[6], self.decimal_places)]
        })
        self.log_data = pd.concat([self.log_data, new_row], ignore_index=True)
        self._save_to_csv()

    def _save_to_csv(self):
        self.log_data.to_csv(self.log_file_loc, index=False)
        