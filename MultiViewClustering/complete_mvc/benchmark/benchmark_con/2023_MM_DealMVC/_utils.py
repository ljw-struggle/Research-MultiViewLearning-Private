from re import L
import h5py, numpy as np, scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler # normalize the data to [0, 1] at column for default
from scipy.optimize import linear_sum_assignment
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
        case "BBCSport":
            dataset = BBCSport('./data/BBCSport.mat'); dims = [3183, 3203]; view = 2; data_size = 544; class_num = 5
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
    
class BBCSport(Dataset):
    def __init__(self, path):
        # data = sio.loadmat(path)
        data = h5py.File(path, 'r')
        np.random.seed(1); index = list(range(544)); np.random.shuffle(index)
        self.x_1 = MinMaxScaler().fit_transform(np.array(data[data['X'][0][0]]).T.astype(np.float32))[index] # shape: (544, 3183)
        self.x_2 = MinMaxScaler().fit_transform(np.array(data[data['X'][0][1]]).T.astype(np.float32))[index] # shape: (544, 3203)
        self.y = np.array(data['Y']).reshape(544).astype(np.int64)[index] # shape: (544,)
        
    def __len__(self):
        return self.y.shape[0] # 544
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)


class GetData(Dataset):
    def __init__(self, name):
        data_path = './data/{}.mat'.format(name[1])
        np.random.seed(1)
        index = [i for i in range(name['N'])]
        np.random.shuffle(index)

        data = h5py.File(data_path)
        Final_data = []
        for i in range(name['V']):
            diff_view = data[data['X'][0, i]]
            diff_view = np.array(diff_view, dtype=np.float32).T
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y']).T
        LABELS = label[index]
        self.name = name
        self.data = Final_data
        self.y = LABELS