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
    # For each cluster, we take the number of samples belonging to the most frequent ground-truth class; 
    # the purity score is then obtained by summing these values over all clusters and normalizing by the total number of samples.
    contingency_matrix_result = contingency_matrix(y_true, y_pred) # shape: (num_true_clusters, num_pred_clusters)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result) 

def load_data(dataset):
    match dataset:
        case "BDGP":
            dataset = BDGP('./data/BDGP.mat'); dims = [1750, 79]; view = 2; data_size = 2500; class_num = 5 
        case "MNIST-USPS": # aka. DIGIT
            dataset = MNIST_USPS('./data/MNIST_USPS.mat'); dims = [784, 784]; view = 2; data_size = 5000; class_num = 10
        case "Fashion":
            dataset = Fashion('./data/Fashion.mat'); dims = [784, 784, 784]; view = 3; data_size = 10000; class_num = 10
        case "DHA":
            dataset = DHA('./data/DHA.mat'); dims = [110, 6144]; view = 2; data_size = 483; class_num = 23
        case "WebKB":
            dataset = WebKB('./data/WebKB.mat'); dims = [2949, 334]; view = 2; data_size = 1051; class_num = 2
        case "NGs":
            dataset = NGs('./data/NGs.mat'); dims = [2000, 2000 , 2000]; view = 3; data_size = 500; class_num = 5
        case "VOC":
            dataset = VOC('./data/VOC.mat'); dims = [512, 399]; view = 2; data_size = 5649; class_num = 20
        case "Fc_COIL_20": # aka. COIL-20
            dataset = Fc_COIL_20('./data/Fc_COIL_20.mat'); dims = [1024, 1024, 1024]; view = 3; data_size = 1440; class_num = 20
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

# MinMaxScaler: normalize the data to [0, 1] at column for default
# scale_normalize_matrix: scale the data to [min_value, max_value] for the whole matrix
def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min(); max_val = input_matrix.max()
    scaled_matrix = (input_matrix - min_val) / (max_val - min_val) * (max_value - min_value) + min_value
    return scaled_matrix

class BDGP(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (2500, 1750)
        self.x_2 = data['X2'].astype(np.float32) # shape: (2500, 79)
        self.y = data['Y'].reshape(2500).astype(np.int64) # shape: (2500,)
        self.x_1 = scale_normalize_matrix(self.x_1)
        self.x_2 = scale_normalize_matrix(self.x_2)

    def __len__(self):
        return self.y.shape[0]

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


class DHA(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (483, 110)
        self.x_2 = data['X2'].astype(np.float32) # shape: (483, 6144)
        self.y = data['Y'].reshape(483).astype(np.int64) # shape: (483,)
        
    def __len__(self):
        return self.y.shape[0] # 483

    def __getitem__(self, idx):
        return [torch.from_numpy(scale_normalize_matrix(self.x_1[idx])), torch.from_numpy(scale_normalize_matrix(self.x_2[idx]))], torch.tensor(self.y[idx]), torch.tensor(idx)

class WebKB(Dataset):
    def __init__(self,path):
        data = sio.loadmat(path)
        self.x_1 = data['X'][0][0].astype(np.float32) # shape: (1051, 2949)
        self.x_2 = data['X'][0][1].astype(np.float32) # shape: (1051, 334)
        self.y = data['gnd'].reshape(1051).astype(np.int64) # shape: (1051,)
        
    def __len__(self):
        return self.y.shape[0] # 1051
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    
class NGs(Dataset):
    def __init__(self,path):
        data = sio.loadmat(path)
        self.x_1 = scale_normalize_matrix(data['data'][0][0].T.astype(np.float32)) # shape: (500, 2000)
        self.x_2 = scale_normalize_matrix(data['data'][0][1].T.astype(np.float32)) # shape: (500, 2000)
        self.x_3 = scale_normalize_matrix(data['data'][0][2].T.astype(np.float32)) # shape: (500, 2000)
        self.y = data['truelabel'][0][0].reshape(500).astype(np.int64) # shape: (500,)

    def __len__(self):
        return self.y.shape[0] # 500
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class VOC(Dataset):
    def __init__(self,path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (5649, 512)
        self.x_2 = data['X2'].astype(np.float32) # shape: (5649, 399)
        self.y = data['Y'].reshape(5649).astype(np.int64) # shape: (5649,)
        
    def __len__(self):
        return self.y.shape[0] # 5649
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    
class Fc_COIL_20(Dataset):
    def __init__(self,path):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (1440, 1024)
        self.x_2 = data['X2'].astype(np.float32) # shape: (1440, 944)
        self.x_3 = data['X3'].astype(np.float32) # shape: (1440, 4096)
        self.y = data['Y'].reshape(1440).astype(np.int64) # shape: (1440,)
        
    def __len__(self):
        return self.y.shape[0] # 1440
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    