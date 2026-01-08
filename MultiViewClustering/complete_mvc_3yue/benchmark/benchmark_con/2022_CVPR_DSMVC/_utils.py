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
        case "BDGP":
            dataset = BDGP('./data/BDGP.mat'); dims = [1750, 79]; view = 2; data_size = 2500; class_num = 5 
        case "caltech_5m":
            dataset = Caltech5M('./data/caltech_5m.npz'); dims = [40, 254, 928, 512, 1984]; view = 5; data_size = 1400; class_num = 7
        case "uci":
            dataset = UCI('./data/uci.npz'); dims = [240, 76, 216, 47, 64, 6]; view = 6; data_size = 2000; class_num = 10
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

class BDGP(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (2500, 1750)
        self.x_2 = data['X2'].astype(np.float32) # shape: (2500, 79)
        self.y = data['Y'].reshape(2500).astype(np.int64) # shape: (2500,)
        
    def __len__(self):
        return self.y.shape[0] # 2500
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    
class Caltech5M(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.x_1 = data['view_0'].astype(np.float32) # shape: (1400, 40)
        self.x_2 = data['view_1'].astype(np.float32) # shape: (1400, 254)
        self.x_3 = data['view_2'].astype(np.float32) # shape: (1400, 928)
        self.x_4 = data['view_3'].astype(np.float32) # shape: (1400, 512)
        self.x_5 = data['view_4'].astype(np.float32) # shape: (1400, 1984)
        self.y = data['labels'].astype(np.int64) # shape: (1400,)
        
    def __len__(self):
        return self.y.shape[0] # 1400
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class UCI(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.x_1 = data['view_0'].astype(np.float32) # shape: (2000, 240)
        self.x_2 = data['view_1'].astype(np.float32) # shape: (2000, 76)
        self.x_3 = data['view_2'].astype(np.float32) # shape: (2000, 216)
        self.x_4 = data['view_3'].astype(np.float32) # shape: (2000, 47)
        self.x_5 = data['view_4'].astype(np.float32) # shape: (2000, 64)
        self.x_6 = data['view_5'].astype(np.float32) # shape: (2000, 6)
        self.y = data['labels'].astype(np.int64) # shape: (2000,)
        
    def __len__(self):
        return self.y.shape[0] # 2000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx]), torch.from_numpy(self.x_6[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
