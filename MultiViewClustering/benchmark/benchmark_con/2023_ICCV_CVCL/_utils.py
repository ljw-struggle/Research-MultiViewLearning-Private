import os, torch, numpy as np, scipy.io as sio
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler # normalize the data to [0, 1] at column for default
from sklearn.metrics.cluster import contingency_matrix, normalized_mutual_info_score, adjusted_rand_score, silhouette_score

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
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
        case "MNIST-USPS":
            dataset = MNIST_USPS('./data/MNIST_USPS.mat'); dims = [784, 784]; view = 2; data_size = 5000; class_num = 10
        case "Fashion":
            dataset = Fashion('./data/Fashion.mat'); dims = [784, 784, 784]; view = 3; data_size = 10000; class_num = 10
        case "MSRCv1":
            dataset = MSRCv1('./data/MSRCv1.mat'); dims = [24, 576, 512, 256, 254]; view = 5; data_size = 210; class_num = 7
        case "COIL20":
            dataset = COIL20('./data/COIL20.mat'); dims = [1024, 944, 4096]; view = 3; data_size = 1440; class_num = 20
        case "Handwritten":
            dataset = Handwritten('./data/handwritten.mat'); dims = [76, 216, 64, 240, 47, 6]; view = 6; data_size = 2000; class_num = 10
        case "Scene15":
            dataset = Scene15('./data/Scene15.mat'); dims = [20, 59, 40]; view = 3; data_size = 4485; class_num = 15
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
    

class MNIST_USPS(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].reshape(5000, 784).astype(np.float32) # shape: (5000, 784)
        self.x_2 = data['X2'].reshape(5000, 784).astype(np.float32) # shape: (5000, 784)
        self.y = data['Y'].reshape(5000).astype(np.int64) # shape: (5000,)

    def __len__(self):
        return self.y.shape[0] # 5000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)


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


class MSRCv1(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0].astype(np.float32)) # shape: (210, 24)
        self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1].astype(np.float32)) # shape: (210, 576)
        self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2].astype(np.float32)) # shape: (210, 512)
        self.x_4 = MinMaxScaler().fit_transform(data['X'][0][3].astype(np.float32)) # shape: (210, 256)
        self.x_5 = MinMaxScaler().fit_transform(data['X'][0][4].astype(np.float32)) # shape: (210, 254)
        self.y = data['Y'].reshape(210).astype(np.int64) # shape: (210, 1)

    def __len__(self):
        return self.y.shape[0] # 210

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    

class COIL20(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0].astype(np.float32)) # shape: (1440, 1024)
        self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1].astype(np.float32)) # shape: (1440, 944)
        self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2].astype(np.float32)) # shape: (1440, 4096)
        self.y = data['Y'].reshape(1440).astype(np.int64) # shape: (1440,)
        
    def __len__(self):
        return self.y.shape[0] # 1440

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class Handwritten(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0].astype(np.float32)) # shape: (2000, 76)
        self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1].astype(np.float32)) # shape: (2000, 216)
        self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2].astype(np.float32)) # shape: (2000, 64)
        self.x_4 = MinMaxScaler().fit_transform(data['X'][0][3].astype(np.float32)) # shape: (2000, 240)
        self.x_5 = MinMaxScaler().fit_transform(data['X'][0][4].astype(np.float32)) # shape: (2000, 47)
        self.x_6 = MinMaxScaler().fit_transform(data['X'][0][5].astype(np.float32)) # shape: (2000, 6)
        self.y = data['Y'].reshape(2000).astype(np.int64) # shape: (2000,)
        
    def __len__(self):
        return self.y.shape[0] # 2000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx]), torch.from_numpy(self.x_6[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)


class Scene15(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0].astype(np.float32)) # shape: (4485, 20)
        self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1].astype(np.float32)) # shape: (4485, 59)
        self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2].astype(np.float32)) # shape: (4485, 40)
        self.y = data['Y'].reshape(4485).astype(np.int64) # shape: (4485,)
        
    def __len__(self):
        return self.y.shape[0] # 4485
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

