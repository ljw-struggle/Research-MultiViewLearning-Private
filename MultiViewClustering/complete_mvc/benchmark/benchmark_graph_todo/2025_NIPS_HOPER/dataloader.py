from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# path 替换成本地的数据集位置

def load_data(dataset):
    if dataset == "BBCSport":
        path = "/root/0316/实验1/dataset/BBCSport.mat"          # 2 views
        data = loadmat(path)
        X,y = data['X'], data['y']

        labels = y.squeeze(1)  # size (544,)
        num_views = X.shape[0]
        num_samples = labels.size
        num_clusters = np.unique(labels).size

        data_views = list()  # 原始的输入数据的列表
        input_sizes = list() # 原始的输入特征的维度列表

        for idx in range(num_views):
            data_view = torch.from_numpy(X[idx][0].astype(np.float32)).to(device)
            data_views.append(data_view)
            input_sizes.append(X[idx][0].shape[1])
    elif dataset == "WebKB":
        path = "/root/0316/实验1/dataset/WebKB.mat"          # 2 views
        data = loadmat(path)
        X,y = data['X'], data['y']
        labels = y.squeeze(1)  
        num_views = X.shape[0]
        num_samples = labels.size
        num_clusters = np.unique(labels).size

        data_views = list()  # 原始的输入数据的列表
        input_sizes = list() # 原始的输入特征的维度列表

        for idx in range(num_views):
            data_view = torch.from_numpy(X[idx][0].astype(np.float32)).to(device)
            data_views.append(data_view)
            input_sizes.append(X[idx][0].shape[1])
    elif dataset =="synthetic3d":
        path = "/root/0316/实验1/dataset/synthetic3d.mat"     # 3 views
        data = loadmat(path)
        X,y = data['X'], data['Y']
        labels = y.squeeze(1)  
        num_views = X.shape[0]
        num_samples = labels.size
        num_clusters = np.unique(labels).size

        data_views = list()  # 原始的输入数据的列表
        input_sizes = list() # 原始的输入特征的维度列表

        for idx in range(num_views):
            data_view = torch.from_numpy(X[idx][0].astype(np.float32)).to(device)
            data_views.append(data_view)
            input_sizes.append(X[idx][0].shape[1])
    elif dataset =="Hdigit":
        path = "/root/0316/实验1/dataset/Hdigit.mat"
        data = loadmat(path)
        X,y = data['data'].T, data['truelabel'][0][0].T
        labels = y.squeeze(1)  
        num_views = X.shape[0]
        num_samples = labels.size
        num_clusters = np.unique(labels).size
        for idx in range(num_views):
            X[idx][0] = X[idx][0].T.copy()

        data_views = list()  # 原始的输入数据的列表
        input_sizes = list() # 原始的输入特征的维度列表

        for idx in range(num_views):
            tmp = MinMaxScaler().fit_transform(X[idx][0].astype(np.float32))
            data_view = torch.from_numpy(tmp).to(device)
            data_views.append(data_view)
            input_sizes.append(X[idx][0].shape[1])
    elif dataset == "coil20":
        path = "/root/0316/实验1/dataset/COIL20.mat"  # 80mb
        data = loadmat(path)
        X,y = data['X'].T, data['Y']
        labels = y.squeeze(1)  
        num_views = X.shape[0]
        num_samples = labels.size
        num_clusters = np.unique(labels).size
        for idx in range(num_views):
            X[idx][0] = X[idx][0].copy()

        data_views = list()  # 原始的输入数据的列表
        input_sizes = list() # 原始的输入特征的维度列表

        for idx in range(num_views):
            tmp = StandardScaler().fit_transform(X[idx][0].astype(np.float32))
            data_view = torch.from_numpy(tmp).to(device)
            data_views.append(data_view)
            input_sizes.append(X[idx][0].shape[1])
    elif dataset == "Handwritten":     
        path = "/root/0316/实验1/dataset/Mfeat.mat"        # 26 views
        data = loadmat(path)
        X,y = data['data'].T, data['truelabel']
        labels = y[0][0].squeeze(1)  
        num_views = X.shape[0]
        num_samples = labels.size
        num_clusters = np.unique(labels).size

        data_views = list()  # 原始的输入数据的列表
        input_sizes = list() # 原始的输入特征的维度列表

        for idx in range(num_views):
            tmp = StandardScaler().fit_transform(X[idx][0].T.astype(np.float32))
            data_view = torch.from_numpy(tmp).to(device)
            data_views.append(data_view)
            input_sizes.append(X[idx][0].T.shape[1])
    else:
        raise NotImplementedError
    return data_views,labels,input_sizes,num_views,num_samples,num_clusters

