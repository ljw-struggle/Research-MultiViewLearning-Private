import os, random, numpy as np, scipy.io as sio
import torch
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score, silhouette_score

def get_aligned_pred(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    D = max(np.max(y_true), np.max(y_pred)) + 1
    M = confusion_matrix(y_true, y_pred, labels=np.arange(D))  # shape: (num_true_clusters, num_pred_clusters) 
    cost_matrix = M.max() - M # shape: [num_true_clusters, num_pred_clusters]
    ind = linear_sum_assignment(cost_matrix) # align clusters using the Hungarian algorithm, ind[0] is the true clusters, ind[1] is the predicted clusters
    matched_dict = {i: j for i, j in zip(ind[1], ind[0])} # shape: {pred_cluster: true_cluster}
    matched_matrix = np.zeros((D, D)); matched_matrix[ind[1], ind[0]] = 1 # shape: [num_pred_clusters, num_true_clusters]
    y_pred_aligned = np.array([matched_dict[i] for i in y_pred])
    return y_pred_aligned, matched_matrix # y_pred_aligned: (num_samples,), matched_matrix: (num_pred_clusters, num_true_clusters)

def evaluate(label, pred):
    pred_aligned, _ = get_aligned_pred(label, pred)
    nmi = normalized_mutual_info_score(label, pred_aligned)
    ari = adjusted_rand_score(label, pred_aligned)
    acc = accuracy_score(label, pred_aligned)
    pur = np.sum(np.amax(contingency_matrix(label, pred_aligned), axis=0)) / len(label)
    # f1_weighted = f1_score(label, pred_aligned, average="weighted")
    # f1_macro = f1_score(label, pred_aligned, average="macro")
    # f1_micro = f1_score(label, pred_aligned, average="micro")
    # asw = silhouette_score(embedding, pred_aligned) # silhouette score
    return nmi, ari, acc, pur

def load_data(dataset):
    match dataset:
        case "Caltech101-20":
            dataset = Caltech('./data/Caltech101-20.mat'); dims = [1984, 512]; view = 2; data_size = 2386; class_num = 20
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

def normalize(x):
    x = (x-np.min(x)) / (np.max(x)-np.min(x))
    return x

class Caltech(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = normalize(data['X1']).astype(np.float32) # shape: (2386, 1984)
        self.x_2 = normalize(data['X2']).astype(np.float32) # shape: (2386, 512)
        self.y = data['Y'].reshape(-1).astype(np.int64) # shape: (2386,)
        self.y = self.y - np.min(self.y)
    
    def __len__(self):
        return self.y.shape[0] # 2386
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
