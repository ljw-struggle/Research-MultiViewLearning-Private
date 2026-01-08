import os, csv, logging, random, scipy.io as sio, numpy as np
import matplotlib.pyplot as plt
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
        case "MSRCV1":
            dataset = MSRCV1('./data/MSRCV1.mat'); dims = [1302, 48, 512, 100, 256, 210]; view = 6; data_size = 210; class_num = 7
        case "Out-Scene":
            dataset = OutScene('./data/Out-Scene.mat'); dims = [512, 432, 256, 48]; view = 4; data_size = 2688; class_num = 8
        case "ALOI-100":
            dataset = ALOI100('./data/slow_ALOI-100.mat'); dims = [77, 13, 64, 125]; view = 4; data_size = 10800; class_num = 100
        case "Synthetic3d":
            dataset = Synthetic3d('./data/synthetic3d.mat'); dims = [3, 3, 3]; view = 3; data_size = 600; class_num = 3
        case _:
            raise NotImplementedError(f"Dataset '{dataset}' is not supported")
    return dataset, dims, view, data_size, class_num

class MSRCV1(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        # self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0].astype(np.float32)) # shape: (210, 24)
        # self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1].astype(np.float32)) # shape: (210, 576)
        # self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2].astype(np.float32)) # shape: (210, 512)
        # self.x_4 = MinMaxScaler().fit_transform(data['X'][0][3].astype(np.float32)) # shape: (210, 256)
        # self.x_5 = MinMaxScaler().fit_transform(data['X'][0][4].astype(np.float32)) # shape: (210, 254)
        # self.x_6 = MinMaxScaler().fit_transform(data['X'][0][5].astype(np.float32)) # shape: (210, 210)
        # NOTE: SparseMVC original code
        self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0]).astype(np.float32) # shape: (210, 24)
        self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1]).astype(np.float32) # shape: (210, 576)
        self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2]).astype(np.float32) # shape: (210, 512)
        self.x_4 = MinMaxScaler().fit_transform(data['X'][0][3]).astype(np.float32) # shape: (210, 256)
        self.x_5 = MinMaxScaler().fit_transform(data['X'][0][4]).astype(np.float32) # shape: (210, 254)
        self.x_6 = MinMaxScaler().fit_transform(data['X'][0][5]).astype(np.float32) # shape: (210, 210)
        self.y = data['Y'].reshape(210).astype(np.int64) # shape: (210, 1)

    def __len__(self):
        return self.y.shape[0] # 210

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx]), torch.from_numpy(self.x_5[idx]), torch.from_numpy(self.x_6[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class OutScene(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        # self.x_1 = MinMaxScaler().fit_transform(data['X1'].astype(np.float32)) # shape: (2688, 512)
        # self.x_2 = MinMaxScaler().fit_transform(data['X2'].astype(np.float32)) # shape: (2688, 432)
        # self.x_3 = MinMaxScaler().fit_transform(data['X3'].astype(np.float32)) # shape: (2688, 256)
        # self.x_4 = MinMaxScaler().fit_transform(data['X4'].astype(np.float32)) # shape: (2688, 48)
        # NOTE: SparseMVC original code
        self.x_1 = MinMaxScaler().fit_transform(data['X1']).astype(np.float32) # shape: (2688, 512)
        self.x_2 = MinMaxScaler().fit_transform(data['X2']).astype(np.float32) # shape: (2688, 432)
        self.x_3 = MinMaxScaler().fit_transform(data['X3']).astype(np.float32) # shape: (2688, 256)
        self.x_4 = MinMaxScaler().fit_transform(data['X4']).astype(np.float32) # shape: (2688, 48)
        self.y = data['Y'].reshape(2688).astype(np.int64) # shape: (2688,)
        
    def __len__(self):
        return self.y.shape[0] # 2688
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(self.view3[idx]), torch.from_numpy(self.view4[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class Synthetic3d(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        # self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0].astype(np.float32)) # shape: (600, 3)
        # self.x_2 = MinMaxScaler().fit_transform(data['X'][1][0].astype(np.float32)) # shape: (600, 3)
        # self.x_3 = MinMaxScaler().fit_transform(data['X'][2][0].astype(np.float32)) # shape: (600, 3)
        # NOTE: SparseMVC original code
        self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0]).astype(np.float32) # shape: (600, 3)
        self.x_2 = MinMaxScaler().fit_transform(data['X'][1][0]).astype(np.float32) # shape: (600, 3)
        self.x_3 = MinMaxScaler().fit_transform(data['X'][2][0]).astype(np.float32) # shape: (600, 3)
        self.y = data['Y'].astype(np.int64).reshape(600,) # shape: (600,)
        
    def __len__(self):
        return self.y.shape[0] # 600
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

class ALOI100(Dataset):
    def __init__(self, path):
        data = sio.loadmat(path)
        # self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0].astype(np.float32)) # shape: (10800, 77)
        # self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1].astype(np.float32)) # shape: (10800, 13)
        # self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2].astype(np.float32)) # shape: (10800, 64)
        # self.x_4 = MinMaxScaler().fit_transform(data['X'][0][3].astype(np.float32)) # shape: (10800, 125)
        # NOTE: SparseMVC original code
        self.x_1 = MinMaxScaler().fit_transform(data['X'][0][0]).astype(np.float32) # shape: (10800, 77)
        self.x_2 = MinMaxScaler().fit_transform(data['X'][0][1]).astype(np.float32) # shape: (10800, 13)
        self.x_3 = MinMaxScaler().fit_transform(data['X'][0][2]).astype(np.float32) # shape: (10800, 64)
        self.x_4 = MinMaxScaler().fit_transform(data['X'][0][3]).astype(np.float32) # shape: (10800, 125)
        self.y = data['Y'].astype(np.int64).reshape(10800,) # shape: (10800,)
        
    def __len__(self):
        return self.y.shape[0] # 10800
    
    def __getitem__(self, idx):
        return [torch.from_numpy(self.x_1[idx]), torch.from_numpy(self.x_2[idx]), torch.from_numpy(self.x_3[idx]), torch.from_numpy(self.x_4[idx])], torch.tensor(self.y[idx]), torch.tensor(idx)

def plot_feature_kline(data, dataset_name, plot_name):
    save_dir = 'result/{}'.format(dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    num_features = data.shape[1]
    q_0 = np.min(data, axis=0); q_1 = np.percentile(data, 25, axis=0); q_2 = np.median(data, axis=0); 
    q_3 = np.percentile(data, 75, axis=0); q_4 = np.max(data, axis=0); candle_lengths = q_3 - q_1
    norm = plt.Normalize(vmin=candle_lengths.min(), vmax=candle_lengths.max())
    cmap = plt.colormaps['coolwarm']
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.subplots_adjust(right=0.85)
    for i in range(num_features):
        color = cmap(norm(candle_lengths[i]))
        ax.plot([i, i], [q_3[i], q_4[i]], color=color, lw=1.2)
        ax.plot([i, i], [q_1[i], q_0[i]], color=color, lw=1.2)
        ax.add_patch(plt.Rectangle(xy=(i - 0.2, q_1[i]), width=0.4, height=q_3[i] - q_1[i], fill=False, edgecolor=color, lw=1.2))
    xticks = list(range(0, num_features, num_features // 10))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(i + 1) for i in xticks], ha='center')
    ax.set_xlabel('Feature Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Feature Separation ({})'.format(plot_name))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cax = fig.add_axes([0.86, 0.15, 0.01, 0.7]) # Add a new axes for the color bar
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Candle Length')
    plt.savefig(os.path.join(save_dir, plot_name, '.png'), format='png', dpi=150, bbox_inches='tight')

def generate_dataset_with_feature_in_different_distributions(sample_num, feature_num, seed=42):
    np.random.seed(seed)
    dataset = np.zeros((sample_num, feature_num))
    for i in range(feature_num):
        mean = np.random.uniform(-3, 3)  # Mean randomly selected in range [-3, 3]
        std = np.random.uniform(0.5, 2)  # Standard deviation randomly selected in range [0.5, 2]
        dataset[:, i] = np.random.randn(sample_num) * std + mean
    return dataset

if __name__ == '__main__':
    example_dataset = generate_dataset_with_feature_in_different_distributions(sample_num=1000, feature_num=128, seed=42)
    plot_feature_kline(example_dataset, dataset_name='demo', plot_name='test')
    