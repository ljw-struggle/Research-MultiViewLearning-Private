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
        case "CCV":
            dataset = CCV('./data/CCV/'); dims = [5000, 5000, 4000]; view = 3; data_size = 6773; class_num = 20
        case "Caltech":
            dataset = Caltech_6V('data/Caltech.mat'); dims = [48, 40, 254, 1984, 512, 928]; view = 6; data_size = 1400; class_num = 7
        case "NUSWIDE":
            dataset = NUSWIDE('data/NUSWIDE.mat'); dims = [65, 226, 145, 74, 129]; view = 5; data_size = 5000; class_num = 5
        case "DHA":
            dataset = DHA('data/DHA.mat'); dims = [110, 6144]; view = 2; data_size = 483; class_num = 23
        case "YoutubeVideo":
            dataset = YoutubeVideo("./data/Video-3V.mat"); dims = [512, 647, 838]; view = 3; data_size = 101499; class_num = 31
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

class CCV(Dataset):
    def __init__(self, path):
        self.x_1 = MinMaxScaler().fit_transform(np.load(os.path.join(path, 'STIP.npy')).astype(np.float32)) # shape: (6773, 5000)
        self.x_2 = np.load(os.path.join(path, 'SIFT.npy')).astype(np.float32) # shape: (6773, 5000)
        self.x_3 = np.load(os.path.join(path, 'MFCC.npy')).astype(np.float32) # shape: (6773, 4000)
        self.y = np.load(os.path.join(path, 'label.npy')).reshape(6773).astype(np.int64) # shape: (6773,)

    def __len__(self):
        return self.y.shape[0] # 6773

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class Caltech_6V(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = MinMaxScaler().fit_transform(data['X1'].astype(np.float32)) # shape: (1400, 48)
        self.x_2 = MinMaxScaler().fit_transform(data['X2'].astype(np.float32)) # shape: (1400, 40)
        self.x_3 = MinMaxScaler().fit_transform(data['X3'].astype(np.float32)) # shape: (1400, 254)
        self.x_4 = MinMaxScaler().fit_transform(data['X4'].astype(np.float32)) # shape: (1400, 1984)
        self.x_5 = MinMaxScaler().fit_transform(data['X5'].astype(np.float32)) # shape: (1400, 512)
        self.x_6 = MinMaxScaler().fit_transform(data['X6'].astype(np.float32)) # shape: (1400, 928)
        self.y = data['Y'].reshape(1400).astype(np.int64) # shape: (1400,)
        
    def __len__(self):
        return self.y.shape[0] # 1400
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx]), torch.from_numpy(self.x_6[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class NUSWIDE(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (5000, 65)
        self.x_2 = data['X2'].astype(np.float32) # shape: (5000, 226)
        self.x_3 = data['X3'].astype(np.float32) # shape: (5000, 145)
        self.x_4 = data['X4'].astype(np.float32) # shape: (5000, 74)
        self.x_5 = data['X5'].astype(np.float32) # shape: (5000, 129)
        self.y = data['Y'].reshape(5000).astype(np.int64) # shape: (5000,)
        
    def __len__(self):
        return self.y.shape[0] # 5000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class DHA(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (483, 110)
        self.x_2 = data['X2'].astype(np.float32) # shape: (483, 6144)
        self.y = data['Y'].reshape(483).astype(np.int64) # shape: (483,)
        
    def __len__(self):
        return self.y.shape[0] # 483
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class YoutubeVideo(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (101499, 512)
        self.x_2 = data['X2'].astype(np.float32) # shape: (101499, 647)
        self.x_3 = data['X3'].astype(np.float32) # shape: (101499, 838)
        self.y = data['Y'].reshape(101499).astype(np.int64) # shape: (101499,)
        
    def __len__(self):
        return self.y.shape[0] # 101499
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
