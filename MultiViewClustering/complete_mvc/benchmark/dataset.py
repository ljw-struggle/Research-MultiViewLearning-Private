import os, torch, numpy as np, scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler # normalize the data to [0, 1] at column for default

_BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BENCHMARK_DIR)
_DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

def _get_data_path(relative_path): # convert relative path to absolute path
    return os.path.join(_DATA_DIR, os.path.basename(relative_path))

def load_data(dataset):
    match dataset:
        case "BDGP":
            dataset = BDGP(_get_data_path('BDGP.mat')); dims = [1750, 79]; view = 2; data_size = 2500; class_num = 5
        case "CCV":
            dataset = CCV(_get_data_path('CCV')); dims = [5000, 5000, 4000]; view = 3; data_size = 6773; class_num = 20
        case "Fashion":
            dataset = Fashion(_get_data_path('Fashion.mat')); dims = [784, 784, 784]; view = 3; data_size = 10000; class_num = 10
        case "Caltech-2V":
            dataset = Caltech(_get_data_path('Caltech-5V.mat'), view=2); dims = [40, 254]; view = 2; data_size = 1400; class_num = 7
        case "Caltech-3V":
            dataset = Caltech(_get_data_path('assets/Caltech-5V.mat'), view=3); dims = [40, 254, 928]; view = 3; data_size = 1400; class_num = 7
        case "Caltech-4V":
            dataset = Caltech(_get_data_path('assets/Caltech-5V.mat'), view=4); dims = [40, 254, 928, 512]; view = 4; data_size = 1400; class_num = 7
        case "Caltech-5V":
            dataset = Caltech(_get_data_path('assets/Caltech-5V.mat'), view=5); dims = [40, 254, 928, 512, 1984]; view = 5; data_size = 1400; class_num = 7
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
