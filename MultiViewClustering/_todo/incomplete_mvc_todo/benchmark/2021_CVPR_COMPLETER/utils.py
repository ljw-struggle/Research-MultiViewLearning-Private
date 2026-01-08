import random, numpy as np, scipy.io as sio, torch
from scipy import sparse
from torch.utils.data import Dataset

def load_data(dataset):
    match dataset:
        case "Scene_15":
            dataset = Scene15('./data/Scene-15.mat'); dims = [20, 59]; view = 2; data_size = 4485; class_num = 15
        case "LandUse_21":
            dataset = LandUse21('./data/LandUse-21.mat'); dims = [59, 40]; view = 2; data_size = 2100; class_num = 21
        case "NoisyMNIST":
            dataset = NoisyMNIST('./data/NoisyMNIST.mat'); dims = [784, 784]; view = 2; data_size = 10000; class_num = 10
        case "Caltech101-20":
            dataset = Caltech10120('./data/Caltech101-20.mat'); dims = [1984, 512]; view = 2; data_size = 2386; class_num = 20
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

class Scene15(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X'][0][0].astype(np.float32) # shape: (4485, 20)
        self.x_2 = data['X'][0][1].astype(np.float32) # shape: (4485, 59)
        self.y = np.squeeze(data['Y']).astype(np.int64) # shape: (4485,)
        
    def __len__(self):
        return self.y.shape[0] # 4485
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class LandUse21(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        index = random.sample(range(data['X'][0][0].shape[0]), 2100)
        self.x_1 = sparse.csr_matrix(data['X'][0][0]).A[index].astype(np.float32) # shape: (2100, 59)
        self.x_2 = sparse.csr_matrix(data['X'][0][1]).A[index].astype(np.float32) # shape: (2100, 40)
        self.y = np.squeeze(data['Y']).astype(np.int64)[index] # shape: (2100,)
        
    def __len__(self):
        return self.y.shape[0] # 2100
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class NoisyMNIST(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = np.concatenate([data['XV1'], data['XTe1']], axis=0).astype(np.float32) # shape: (10000, 784)
        self.x_2 = np.concatenate([data['XV2'], data['XTe2']], axis=0).astype(np.float32) # shape: (10000, 784)
        self.y = np.concatenate([np.squeeze(data['tuneLabel'][:, 0]), np.squeeze(data['testLabel'][:, 0])]).astype(np.int64) # shape: (10000,)
        
    def __len__(self):
        return self.y.shape[0] # 10000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class Caltech10120(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = self.normalize(data['X'][0][3]).astype(np.float32) # shape: (2386, 1984)
        self.x_2 = self.normalize(data['X'][0][4]).astype(np.float32) # shape: (2386, 512)
        self.y = np.squeeze(data['Y']).astype(np.int64) # shape: (2386,)
        
    def __len__(self):
        return self.y.shape[0] # 2386
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

    @staticmethod
    def normalize(x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) # normalize to [0, 1]
        return x