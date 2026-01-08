import os, scipy.io as sio, numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics.cluster import contingency_matrix, v_measure_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment

def evaluate(label, pred):
    # nmi = normalized_mutual_info_score(label, pred)
    nmi = v_measure_score(label, pred) # This score is identical to normalized_mutual_info_score with the 'arithmetic' option for averaging.
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
        case "CCV":
            dataset = CCV('./data/CCV/'); dims = [5000, 5000, 4000]; view = 3; data_size = 6773; class_num = 20
        case "Fashion":
            dataset = Fashion('./data/Fashion.mat'); dims = [784, 784, 784]; view = 3; data_size = 10000; class_num = 10
        case "Caltech-2V":
            dataset = Caltech('./data/Caltech-5V.mat', view=2); dims = [40, 254]; view = 2; data_size = 1400; class_num = 7
        case "Caltech-3V":
            dataset = Caltech('./data/Caltech-5V.mat', view=3); dims = [40, 254, 928]; view = 3; data_size = 1400; class_num = 7
        case "Caltech-4V":
            dataset = Caltech('./data/Caltech-5V.mat', view=4); dims = [40, 254, 928, 512]; view = 4; data_size = 1400; class_num = 7
        case "Caltech-5V":
            dataset = Caltech('./data/Caltech-5V.mat', view=5); dims = [40, 254, 928, 512, 1984]; view = 5; data_size = 1400; class_num = 7
        case "CIFAR10-v5":
            dataset = CIFAR10('./data/CIFAR10-view5.mat'); dims = [768, 576, 512, 640, 944]; view = 5; data_size = 60000; class_num = 10
        case "Out-Scene":
            dataset = OutScene('./data/Out-Scene.mat'); dims = [512, 432, 256, 48]; view = 4; data_size = 2688; class_num = 8
        case "RSSCN7":
            dataset = RSSCN7('./data/RSSCN7.mat'); dims = [768, 540, 885, 512, 800]; view = 5; data_size = 2800; class_num = 7
        case "MirFlickr":
            dataset = MirFlickr('./data/MirFlickr.mat'); dims = [100, 100]; view = 2; data_size = 12154; class_num = 7
        case "STL10_deep":
            dataset = STL10_deep('./data/STL10_deep.mat'); dims = [1024, 512, 2048]; view = 3; data_size = 13000; class_num = 10
        case "CIFAR10_deep":
            dataset = CIFAR10_deep('./data/CIFAR10_deep.mat'); dims = [512, 2048, 1024]; view = 3; data_size = 50000; class_num = 10
        case "Hdigit":
            dataset = Hdigit('./data/Hdigit.mat'); dims = [784, 256]; view = 2; data_size = 10000; class_num = 10
        case "LabelMe":
            dataset = LabelMe('./data/LabelMe.mat'); dims = [512, 245]; view = 2; data_size = 2688; class_num = 10
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

class OutScene(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (2688, 512)
        self.x_2 = data['X2'].astype(np.float32) # shape: (2688, 432)
        self.x_3 = data['X3'].astype(np.float32) # shape: (2688, 256)
        self.x_4 = data['X4'].astype(np.float32) # shape: (2688, 48)
        self.y = data['Y'].reshape(2688).astype(np.int64) # shape: (2688,)
        
    def __len__(self):
        return self.y.shape[0] # 2688
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(self.view3[idx]), torch.from_numpy(self.view4[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class RSSCN7(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (2800, 768)
        self.x_2 = data['X2'].astype(np.float32) # shape: (2800, 540)
        self.x_3 = data['X3'].astype(np.float32) # shape: (2800, 885)
        self.x_4 = data['X4'].astype(np.float32) # shape: (2800, 512)
        self.x_5 = data['X5'].astype(np.float32) # shape: (2800, 800)
        self.y = data['Y'].reshape(2800).astype(np.int64) # shape: (2800,)
        
    def __len__(self):
        return self.y.shape[0] # 2800
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class MirFlickr(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (12154, 100)
        self.x_2 = data['X2'].astype(np.float32) # shape: (12154, 100)
        self.y = data['Y'].reshape(12154).astype(np.int64) # shape: (12154,)
        
    def __len__(self):
        return self.y.shape[0] # 12154
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class STL10_deep(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (13000, 1024)
        self.x_2 = data['X2'].astype(np.float32) # shape: (13000, 512)
        self.x_3 = data['X3'].astype(np.float32) # shape: (13000, 2048)
        self.y = data['Y'].reshape(13000).astype(np.int64) # shape: (13000,)
        
    def __len__(self):
        return 13000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class CIFAR10_deep(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (50000, 512)
        self.x_2 = data['X2'].astype(np.float32) # shape: (50000, 2048)
        self.x_3 = data['X3'].astype(np.float32) # shape: (50000, 1024)
        self.y = data['Y'].reshape(50000).astype(np.int64) # shape: (50000,)
        
    def __len__(self):
        return self.y.shape[0] # 50000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(self.view3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class Hdigit(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (10000, 784)
        self.x_2 = data['X2'].astype(np.float32) # shape: (10000, 256)
        self.y = data['Y'].reshape(10000).astype(np.int64) # shape: (10000,)
        
    def __len__(self):
        return self.y.shape[0] # 10000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

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

class CIFAR10(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (60000, 768)
        self.x_2 = data['X2'].astype(np.float32) # shape: (60000, 576)
        self.x_3 = data['X3'].astype(np.float32) # shape: (60000, 512)
        self.x_4 = data['X4'].astype(np.float32) # shape: (60000, 640)
        self.x_5 = data['X5'].astype(np.float32) # shape: (60000, 944)
        self.y = data['Y'].reshape(60000).astype(np.int64) # shape: (60000,)
        
    def __len__(self):
        return self.y.shape[0] # 60000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
