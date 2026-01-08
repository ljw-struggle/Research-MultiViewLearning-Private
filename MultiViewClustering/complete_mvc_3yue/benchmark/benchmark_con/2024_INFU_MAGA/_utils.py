import os, torch, numpy as np, scipy.io as sio
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler # normalize the data to [0, 1] at column for default
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score, silhouette_score, v_measure_score

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    # nmi = v_measure_score(label, pred) # This score is identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
    ari = adjusted_rand_score(label, pred)
    acc = clustering_acc(label, pred)
    pur = purity_score(label, pred)
    # asw = silhouette_score(embedding, pred) # silhouette score
    return nmi, ari, acc, pur #, asw

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(D)] for i in range(D)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, ind[0] is the row indices (predicted clusters), ind[1] is the column indices (true clusters)
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] # accuracy

def purity_score(y_true, y_pred):
    contingency_matrix_result = contingency_matrix(y_true, y_pred) # shape: (num_true_clusters, num_pred_clusters)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result) 

def load_data(dataset):
    match dataset:
        case "Fashion":
            dataset = Fashion('./data/Fashion.mat'); dims = [784, 784, 784]; view = 3; data_size = 10000; class_num = 10
        case "LabelMe":
            dataset = LabelMe('./data/LabelMe.mat'); dims = [512, 245]; view = 2; data_size = 2688; class_num = 8
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

class Fashion(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].reshape(10000, 784).astype(np.float32) # shape: (10000, 784)
        self.x_2 = data['X2'].reshape(10000, 784).astype(np.float32) # shape: (10000, 784)
        self.x_3 = data['X3'].reshape(10000, 784).astype(np.float32) # shape: (10000, 784)
        self.y = data['Y'].reshape(10000).astype(np.int64) # shape: (10000,)

    def __len__(self):
        return self.y.shape[0] # 10000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class LabelMe(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (2688, 512)
        self.x_2 = data['X2'].astype(np.float32) # shape: (2688, 245)
        self.y = data['Y'].reshape(2688).astype(np.int64) # shape: (2688,)
        
    def __len__(self):
        return self.y.shape[0] # 2688    
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
