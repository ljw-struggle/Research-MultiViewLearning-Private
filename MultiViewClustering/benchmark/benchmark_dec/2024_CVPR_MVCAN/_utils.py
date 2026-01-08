import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, contingency_matrix
from scipy.optimize import linear_sum_assignment

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
        case "DIGIT": # [1, 32, 32] -> [1024]
            dataset_obj = DIGIT('./data/DIGIT2V_N.mat'); dims = [1024, 1024]; view = 2; data_size = 5000; class_num = 10
        case "NoisyDIGIT": # [1, 32, 32] -> [1024]
            dataset_obj = NoisyDIGIT('./data/DIGIT2V_N.mat'); dims = [1024, 1024, 1024]; view = 3; data_size = 5000; class_num = 10
        case "Amazon": # [3, 32, 32] -> [3072]
            dataset_obj = Amazon('./data/Amazon3V_N.mat'); dims = [3072, 3072, 3072]; view = 3; data_size = 4790; class_num = 10
        case "NoisyAmazon": # [3, 32, 32] -> [3072]
            dataset_obj = NoisyAmazon('./data/Amazon3V_N.mat'); dims = [3072, 3072, 3072, 3072]; view = 4; data_size = 4790; class_num = 10
        case "COIL": # [1, 32, 32] -> [1024]
            dataset_obj = COIL('./data/COIL3V_N.mat'); dims = [1024, 1024, 1024]; view = 3; data_size = 720; class_num = 10
        case "NoisyCOIL": # [1, 32, 32] -> [1024]
            dataset_obj = NoisyCOIL('./data/COIL3V_N.mat'); dims = [1024, 1024, 1024, 1024]; view = 4; data_size = 720; class_num = 10
        case "BDGP":
            dataset_obj = BDGP('./data/BDGP2V_N.mat'); dims = [1750, 79]; view = 2; data_size = 2500; class_num = 5
        case "NoisyBDGP":
            dataset_obj = NoisyBDGP('./data/BDGP2V_N.mat'); dims = [1750, 79, 10]; view = 3; data_size = 2500; class_num = 5
        case "DHA":
            dataset_obj = DHA('./data/DHA.mat'); dims = [110, 6144]; view = 2; data_size = 483; class_num = 23
        case "Caltech-6V":
            dataset_obj = Caltech6V('./data/Caltech.mat'); dims = [48, 40, 254, 1984, 512, 928]; view = 6; data_size = 1400; class_num = 7
        case "RGB-D":
            dataset_obj = RGBD('./data/RGB-D.mat'); dims = [2048, 300]; view = 2; data_size = 1449; class_num = 13
        case "YoutubeVideo":
            dataset_obj = YoutubeVideo('./data/Video-3V.mat'); dims = [512, 647, 838]; view = 3; data_size = 101499; class_num = 31
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset_obj, dims, view, data_size, class_num

class DIGIT(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        # Flatten 图像数据: (1, 32, 32) -> (1024,)
        x1_flat = self.x_1[idx].flatten()
        x2_flat = self.x_2[idx].flatten()
        return [torch.from_numpy(x1_flat), torch.from_numpy(x2_flat)], torch.tensor(self.y[idx]), torch.tensor(idx)
        # return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class NoisyDIGIT(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = data['X3'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        # Flatten 图像数据: (1, 32, 32) -> (1024,)
        x1_flat = self.x_1[idx].flatten()
        x2_flat = self.x_2[idx].flatten()
        x3_flat = self.x_3[idx].flatten()
        return [torch.from_numpy(x1_flat), torch.from_numpy(x2_flat), torch.from_numpy(x3_flat)], torch.tensor(self.y[idx]), torch.tensor(idx)
        # return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class Amazon(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = data['X3'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        # Flatten 图像数据: (3, 32, 32) -> (3072,)
        x1_flat = self.x_1[idx].flatten()
        x2_flat = self.x_2[idx].flatten()
        x3_flat = self.x_3[idx].flatten()
        return [torch.from_numpy(x1_flat), torch.from_numpy(x2_flat), torch.from_numpy(x3_flat)], torch.tensor(self.y[idx]), torch.tensor(idx)
        # return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class NoisyAmazon(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = data['X3'].astype(np.float32)
        self.x_4 = data['X4'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        # Flatten 图像数据: (3, 32, 32) -> (3072,)
        x1_flat = self.x_1[idx].flatten()
        x2_flat = self.x_2[idx].flatten()
        x3_flat = self.x_3[idx].flatten()
        x4_flat = self.x_4[idx].flatten()
        return [torch.from_numpy(x1_flat), torch.from_numpy(x2_flat), torch.from_numpy(x3_flat), torch.from_numpy(x4_flat)], torch.tensor(self.y[idx]), torch.tensor(idx)
        # return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class COIL(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = data['X3'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        # Flatten 图像数据: (1, 32, 32) -> (1024,)
        x1_flat = self.x_1[idx].flatten()
        x2_flat = self.x_2[idx].flatten()
        x3_flat = self.x_3[idx].flatten()
        return [torch.from_numpy(x1_flat), torch.from_numpy(x2_flat), torch.from_numpy(x3_flat)], torch.tensor(self.y[idx]), torch.tensor(idx)
        # return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class NoisyCOIL(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = data['X3'].astype(np.float32)
        self.x_4 = data['X4'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        # Flatten 图像数据: (1, 32, 32) -> (1024,)
        x1_flat = self.x_1[idx].flatten()
        x2_flat = self.x_2[idx].flatten()
        x3_flat = self.x_3[idx].flatten()
        x4_flat = self.x_4[idx].flatten()
        return [torch.from_numpy(x1_flat), torch.from_numpy(x2_flat), torch.from_numpy(x3_flat), torch.from_numpy(x4_flat)], torch.tensor(self.y[idx]), torch.tensor(idx)
        # return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class BDGP(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class NoisyBDGP(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = data['X3'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class DHA(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = MinMaxScaler().fit_transform(data['X1'].astype(np.float32))
        self.x_2 = MinMaxScaler().fit_transform(data['X2'].astype(np.float32))
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class Caltech6V(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = MinMaxScaler().fit_transform(data['X3'].astype(np.float32))
        self.x_4 = data['X4'].astype(np.float32)
        self.x_5 = data['X5'].astype(np.float32)
        self.x_6 = data['X6'].astype(np.float32)
        y = np.squeeze(data['Y']).astype(np.int64)
        y[y == 95] = 5  # cleaning labels to [0, 1, 2 ... K-1] for visualization
        self.y = y
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), 
                torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx]), torch.from_numpy(self.x_6[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class RGBD(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.y = np.squeeze(data['Y']).astype(np.int64)
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class YoutubeVideo(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32)
        self.x_2 = data['X2'].astype(np.float32)
        self.x_3 = data['X3'].astype(np.float32)
        y = np.squeeze(data['Y']).astype(np.int64)
        self.y = y - 1  # cleaning labels to [0, 1, 2 ... K-1] for visualization
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
