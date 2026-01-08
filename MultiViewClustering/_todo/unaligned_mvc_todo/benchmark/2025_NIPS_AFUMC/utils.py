import random, numpy as np, scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
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

class Caltech(Dataset):
    def __init__(self, path, view):
        data = sio.loadmat(path)
        self.x_1 = MinMaxScaler().fit_transform(data['X1'].astype(np.float32)) # shape: (1400, 40)
        self.x_2 = MinMaxScaler().fit_transform(data['X2'].astype(np.float32)) # shape: (1400, 254)
        self.x_3 = MinMaxScaler().fit_transform(data['X3'].astype(np.float32)) # shape: (1400, 1984)
        self.x_4 = MinMaxScaler().fit_transform(data['X4'].astype(np.float32)) # shape: (1400, 512)
        self.x_5 = MinMaxScaler().fit_transform(data['X5'].astype(np.float32)) # shape: (1400, 928)
        self.y = data['Y'].reshape(1400).astype(np.int64) # shape: (1400,)
        self.view = view

    def __len__(self):
        return self.y.shape[0] # 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx) 
        if self.view == 3:
            return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_5[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
        if self.view == 4:
            return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_5[idx]), torch.from_numpy(self.x_4[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
        if self.view == 5:
            return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_5[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

def load_data(dataset):
    match dataset:
        case "Caltech7":
            dataset = Caltech('./data/Caltech-5V.mat', 5); dims = [40, 254, 1984, 512, 928]; view = 5; data_size = 1400; class_num = 7
        case _:
            raise NotImplementedError
    return dataset, dims, view, data_size, class_num
