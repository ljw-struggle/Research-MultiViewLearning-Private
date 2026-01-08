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
        case "blobs_overlap_5":
            dataset = BlobsOverlap5('./data/blobs_overlap_5.npz'); dims = [2, 2]; view = 2; data_size = 2500; class_num = 5
        case "blobs_overlap":
            dataset = BlobsOverlap('./data/blobs_overlap.npz'); dims = [2, 2]; view = 2; data_size = 3000; class_num = 3
        case "rgbd":
            dataset = RGBD('./data/rgbd.npz'); dims = [2048, 300]; view = 2; data_size = 1449; class_num = 13
        case "voc":
            dataset = VOC('./data/voc.npz'); dims = [512, 399]; view = 2; data_size = 5649; class_num = 20
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

class BlobsOverlap5(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.x_1 = data['view_0'].astype(np.float32) # shape: (2500, 2)
        self.x_2 = data['view_1'].astype(np.float32) # shape: (2500, 2)
        self.y = data['labels'].astype(np.int64) # shape: (2500,)
        
    def __len__(self):
        return self.y.shape[0] # 2500
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class BlobsOverlap(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.x_1 = data['view_0'].astype(np.float32) # shape: (3000, 2)
        self.x_2 = data['view_1'].astype(np.float32) # shape: (3000, 2)
        self.y = data['labels'].astype(np.int64) # shape: (3000,)
        
    def __len__(self):
        return self.y.shape[0] # 3000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class RGBD(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.x_1 = data['view_0'].astype(np.float32) # shape: (1449, 2048)
        self.x_2 = data['view_1'].astype(np.float32) # shape: (1449, 300)
        self.y = data['labels'].astype(np.int64) # shape: (1449,)
        
    def __len__(self):
        return self.y.shape[0] # 1449
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class VOC(Dataset):
    def __init__(self, path):
        data = np.load(path)
        self.x_1 = data['view_0'].astype(np.float32) # shape: (5649, 512)
        self.x_2 = data['view_1'].astype(np.float32) # shape: (5649, 399)
        self.y = data['labels'].astype(np.int64) # shape: (5649,)
        
    def __len__(self):
        return self.y.shape[0] # 5649
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
