import random, numpy as np, scipy.io as sio
import torch
from torch.utils.data import Dataset
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, contingency_matrix
from sklearn.preprocessing import MinMaxScaler
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

def load_data(dataset, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
    match dataset:
        case "BDGP":
            dataset = BDGP('./data/BDGP.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [1750, 79]; view = 2; data_size = 2500; class_num = 5 
        case "DHA":
            dataset = DHA('./data/DHA.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [110, 6144]; view = 2; data_size = 483; class_num = 23
        case "WebKB":
            dataset = WebKB('./data/WebKB.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [2949, 334]; view = 2; data_size = 1051; class_num = 2
        case "NGs":
            dataset = NGs('./data/NGs.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [2000, 2000 , 2000]; view = 3; data_size = 500; class_num = 5
        case "VOC":
            dataset = VOC('./data/VOC.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [512, 399]; view = 2; data_size = 5649; class_num = 20
        case "Cora":
            dataset = Cora('./data/Cora.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [2708, 1433]; view = 2; data_size = 2708; class_num = 7
        case "YoutubeVideo":
            dataset = YoutubeVideo('./data/Video-3V.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [512, 647, 838]; view = 3; data_size = 101499; class_num = 31
        case "Prokaryotic":
            dataset = Prokaryotic('./data/Prokaryotic.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [393, 3, 438]; view = 3; data_size = 551; class_num = 4
        case "Cifar100":
            dataset = Cifar100('./data/cifar100.mat', type=type, trainset_rate=trainset_rate, noise_rate=noise_rate, seed=seed); dims = [512, 2048, 1024]; view = 3; data_size = 50000; class_num = 100
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
    def __init__(self, path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (2500, 1750)
        self.x_2 = data['X2'].astype(np.float32) # shape: (2500, 79)
        self.y = data['Y'].reshape(2500).astype(np.int64) # shape: (2500,)
        self.x_1 = scale_normalize_matrix(self.x_1)
        self.x_2 = scale_normalize_matrix(self.x_2)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.y = self.y[index]
        self.y = noisify(dataset='BDGP', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class DHA(Dataset):
    def __init__(self, path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (483, 110)
        self.x_2 = data['X2'].astype(np.float32) # shape: (483, 6144)
        self.y = data['Y'].reshape(483).astype(np.int64) # shape: (483,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.y = self.y[index]
        self.y = noisify(dataset='DHA', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 483

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class WebKB(Dataset):
    def __init__(self,path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['X'][0][0].astype(np.float32) # shape: (1051, 2949)
        self.x_2 = data['X'][0][1].astype(np.float32) # shape: (1051, 334)
        self.y = data['gnd'].reshape(1051).astype(np.int64) # shape: (1051,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.y = self.y[index]
        self.y = noisify(dataset='WebKB', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 1051
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    
class NGs(Dataset):
    def __init__(self,path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = scale_normalize_matrix(data['data'][0][0].T.astype(np.float32)) # shape: (500, 2000)
        self.x_2 = scale_normalize_matrix(data['data'][0][1].T.astype(np.float32)) # shape: (500, 2000)
        self.x_3 = scale_normalize_matrix(data['data'][0][2].T.astype(np.float32)) # shape: (500, 2000)
        self.y = data['truelabel'][0][0].reshape(500).astype(np.int64) # shape: (500,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.x_3 = self.x_3[index]; self.y = self.y[index]
        self.y = noisify(dataset='NGs', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 500
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
            
class VOC(Dataset):
    def __init__(self,path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (5649, 512)
        self.x_2 = data['X2'].astype(np.float32) # shape: (5649, 399)
        self.y = data['Y'].reshape(5649).astype(np.int64) # shape: (5649,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.y = self.y[index]
        self.y = noisify(dataset='VOC', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 5649
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    
class Cora(Dataset):
    def __init__(self, path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['coracites'].astype(np.float32) # shape: (2708, 2708)
        self.x_2 = data['coracontent'].astype(np.float32) # shape: (2708, 1433)
        # self.x_3 = data['corainbound'].astype(np.float32) # shape: (2708, 128)
        # self.x_4 = data['coraoutbound'].astype(np.float32) # shape: (2708, 128)
        self.y = data['y'].reshape(2708).astype(np.int64)-1 # shape: (2708,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.y = self.y[index]
        self.y = noisify(dataset='Cora', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 2708

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    

class YoutubeVideo(Dataset):
    def __init__(self, path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['X1'].astype(np.float32) # shape: (101499, 512)
        self.x_2 = data['X2'].astype(np.float32) # shape: (101499, 647)
        self.x_3 = data['X3'].astype(np.float32) # shape: (101499, 838)
        self.y = data['Y'].reshape(-1).astype(np.int64)-1 # shape: (101499,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.x_3 = self.x_3[index]; self.y = self.y[index]
        self.y = noisify(dataset='YoutubeVideo', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 101499
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    
class Prokaryotic(Dataset):
    def __init__(self, path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['gene_repert'].astype(np.float32) # shape: (551, 393)
        self.x_2 = MinMaxScaler().fit_transform(data['proteome_comp'].astype(np.float32)) # shape: (551, 3)
        self.x_3 = MinMaxScaler().fit_transform(data['text'].astype(np.float32)) # shape: (551, 438)
        self.y = data['Y'].reshape(-1).astype(np.int64)-1 # shape: (551,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.x_3 = self.x_3[index]; self.y = self.y[index]
        self.y = noisify(dataset='Prokaryotic', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 551
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)
    
class Cifar100(Dataset):
    def __init__(self, path, type='train', trainset_rate=0.7, noise_rate=0.5, seed=0):
        data = sio.loadmat(path)
        self.x_1 = data['data'][0][0].T.astype(np.float32) # shape: (50000, 512)
        self.x_2 = data['data'][1][0].T.astype(np.float32) # shape: (50000, 2048)
        self.x_3 = data['data'][2][0].T.astype(np.float32) # shape: (50000, 1024)
        self.y = data['truelabel'][0][0].T[0].astype(np.int64) # shape: (50000,)
        # Data split and label noise addition
        index = list(range(0, len(self.y))); random.seed(seed); random.shuffle(index)
        if type == 'train': index = index[:int(len(index) * trainset_rate)]
        elif type == 'test': index = index[int(len(index) * trainset_rate):]
        self.x_1 = self.x_1[index]; self.x_2 = self.x_2[index]; self.x_3 = self.x_3[index]; self.y = self.y[index]
        self.y = noisify(dataset='Cifar100', class_num=np.unique(self.y).shape[0], label=self.y, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
        
    def __len__(self):
        return self.y.shape[0] # 50000
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)


## Noise addition functions
def multiclass_noisify(label, probability_matrix, random_state=0): # Flip classes according to transition probability matrix T. It expects a number between 0 and the number of classes - 1.
    noisy_label = label.copy(); 
    random_state = np.random.RandomState(random_state)
    for idx in range(0, label.shape[0]):
        # For example: 
        # multinomial(1, [0.7, 0.2, 0.1], 1) -> 1 time sampling, 1 time trial -> return shape:(1, 3) -> [[1,0,0]]
        # multinomial(1, [0.7, 0.2, 0.1], 5) -> 1 time sampling, 5 time trial -> return shape:(5, 3) -> [[1,0,0], [0,1,0], ...]
        # multinomial(3, [0.7, 0.2, 0.1], 1) -> 3 time sampling, 1 time trial -> return shape:(1, 3) -> [[2,1,0]] (sum=3)
        # multinomial(3, [0.7, 0.2, 0.1], 4) -> 3 time sampling, 4 time trial -> return shape:(4, 3) -> [[2,0,1], [2,1,0], ...]
        sampled_label = random_state.multinomial(1, probability_matrix[label[idx], :], 1)[0]
        noisy_label[idx] = np.where(sampled_label == 1)[0]
    return noisy_label
    
def noisify(dataset='mnist', class_num=10, label=None, noise_type=None, noise_rate=0, random_state=0):
    assert noise_rate >= 0.0 and noise_rate <= 1.0, 'Noise rate must be between 0 and 1'
    if noise_type == 'pairflip': # noise type: pairflip, flip in the pair
        # Flip pairs: class i is flipped to i+1 with probability noise (last class flips to 0)
        # For example: 0 -> 1, 1 -> 2, ..., 9 -> 0
        P = np.eye(class_num)
        for i in range(0, class_num):
            P[i, i], P[i, (i + 1) % class_num] = 1.0 - noise_rate, 1.0 * noise_rate
        return multiclass_noisify(label, probability_matrix=P, random_state=random_state)
    if noise_type == 'symmetric': # noise type: symmetric, flip in the symmetric way
        # Symmetric flip: each class is flipped to a random other class with probability noise
        # For example: 0 -> random (other than 0), 1 -> random (other than 1), ..., 9 -> random (other than 9)
        P = (noise_rate / (class_num - 1)) * np.ones((class_num, class_num))
        for i in range(0, class_num):
            P[i, i] = 1.0 - noise_rate
        return multiclass_noisify(label, probability_matrix=P, random_state=random_state)
    if noise_type == 'asymmetric': # noise type: asymmetric, flip in the asymmetric way
        # Asymmetric flip: each class is flipped to a specific other class with probability noise
        # For example: 7 -> 1 with probability noise, 2 -> 7 with probability noise, 5 <-> 6 with probability noise, 6 <-> 5 with probability noise, 3 -> 8 with probability noise
        if dataset == 'mnist':
            P = np.eye(10)
            P[7, 7], P[7, 1] = 1.0 - noise_rate, 1.0 * noise_rate # 1 <- 7
            P[2, 2], P[2, 7] = 1.0 - noise_rate, 1.0 * noise_rate # 2 -> 7
            P[5, 5], P[5, 6] = 1.0 - noise_rate, 1.0 * noise_rate # 5 <-> 6
            P[6, 6], P[6, 5] = 1.0 - noise_rate, 1.0 * noise_rate # 5 <-> 6
            P[3, 3], P[3, 8] = 1.0 - noise_rate, 1.0 * noise_rate # 3 -> 8
            return multiclass_noisify(label, probability_matrix=P, random_state=random_state)
        if dataset == 'cifar10':
            P = np.eye(10)
            P[9, 9], P[9, 1] = 1.0 - noise_rate, 1.0 * noise_rate # automobile <- truck
            P[2, 2], P[2, 0] = 1.0 - noise_rate, 1.0 * noise_rate # bird -> airplane
            P[3, 3], P[3, 5] = 1.0 - noise_rate, 1.0 * noise_rate # cat <-> dog
            P[5, 5], P[5, 3] = 1.0 - noise_rate, 1.0 * noise_rate # cat <-> dog  
            P[4, 4], P[4, 7] = 1.0 - noise_rate, 1.0 * noise_rate # automobile -> truck
            return multiclass_noisify(label, probability_matrix=P, random_state=random_state)
        if dataset == 'cifar100':
            P = np.eye(100)
            for i in range(0, 20):
                P_temp = np.eye(5)
                for j in range(0, 5):
                    P_temp[j, j], P_temp[j, (j+1) % 5] = 1.0 - noise_rate, 1.0 * noise_rate
                P[i * 5 : (i+1) * 5, i * 5 : (i+1) * 5] = P_temp
            return multiclass_noisify(label, probability_matrix=P, random_state=random_state)
