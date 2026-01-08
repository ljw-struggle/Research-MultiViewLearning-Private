import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score, silhouette_score

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = clustering_acc(label, pred)
    pur = purity_score(label, pred)
    # cls = clustering_classification_metrics(label, pred, average="macro")  # accuracy, precision, recall, f1
    # asw = silhouette_score(embedding, pred) # silhouette score
    return nmi, ari, acc, pur  # for F1 etc. use clustering_classification_metrics(label, pred)

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(D)] for i in range(D)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, ind[0] is the row indices (predicted clusters), ind[1] is the column indices (true clusters)
    # matched_dict = {i: j for i, j in zip(ind[0], ind[1])}
    # y_pred_aligned = np.array([matched_dict[i] for i in y_pred])
    # acc = accuracy_score(y_true, y_pred_aligned)
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] # accuracy

def purity_score(y_true, y_pred):
    contingency_matrix_result = contingency_matrix(y_true, y_pred) # shape: (num_true_clusters, num_pred_clusters)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result) 

def load_data(dataset):
    match dataset:
        case "Caltech":
            dataset = Caltech('./data/Caltech.mat'); dims = [1984, 512]; view = 2; data_size = 2386; class_num = 20
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

class Caltech(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (2386, 1984)
        self.x_2 = data['X2'].astype(np.float32) # shape: (2386, 512)
        self.y = data['Y'].reshape(-1).astype(np.int64) # shape: (2386,)
    
    def __len__(self):
        return self.y.shape[0] # 2386
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
