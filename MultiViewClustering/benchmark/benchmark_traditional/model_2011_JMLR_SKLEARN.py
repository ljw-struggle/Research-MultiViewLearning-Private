import torch, random, numpy as np
from sklearn.cluster import (KMeans, MiniBatchKMeans, SpectralClustering, AgglomerativeClustering, Birch)
from sklearn.mixture import GaussianMixture
from ..dataset import load_data
from ..metric import evaluate

def benchmark_2011_JMLR_SKLEARN_KMeans(dataset_name='BDGP', use_view=-1, init='k-means++', n_init=10, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    model = KMeans(n_clusters=class_num, init=init, n_init=n_init, random_state=random_state)
    y_pred = model.fit_predict(data)
    nmi, ari, acc, pur = evaluate(label, y_pred)
    return nmi, ari, acc, pur

def benchmark_2011_JMLR_SKLEARN_MiniBatchKMeans(dataset_name='BDGP', use_view=-1, init='k-means++', n_init=10, batch_size=1024, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    model = MiniBatchKMeans(n_clusters=class_num, init=init, n_init=n_init, batch_size=batch_size, random_state=random_state)
    y_pred = model.fit_predict(data)
    nmi, ari, acc, pur = evaluate(label, y_pred)
    return nmi, ari, acc, pur

def benchmark_2011_JMLR_SKLEARN_SpectralClustering(dataset_name='BDGP', use_view=-1, n_neighbors=10, affinity='rbf', random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    model = SpectralClustering(n_clusters=class_num, n_neighbors=n_neighbors, affinity=affinity, random_state=random_state)
    y_pred = model.fit_predict(data)
    nmi, ari, acc, pur = evaluate(label, y_pred)
    return nmi, ari, acc, pur

def benchmark_2011_JMLR_SKLEARN_AgglomerativeClustering(dataset_name='BDGP', use_view=-1, linkage='ward', metric='euclidean', random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    model = AgglomerativeClustering(n_clusters=class_num, linkage=linkage, metric=metric)
    y_pred = model.fit_predict(data)
    nmi, ari, acc, pur = evaluate(label, y_pred)
    return nmi, ari, acc, pur

def benchmark_2011_JMLR_SKLEARN_Birch(dataset_name='BDGP', use_view=-1, threshold=0.5, branching_factor=50, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    model = Birch(n_clusters=class_num, threshold=threshold, branching_factor=branching_factor)
    y_pred = model.fit_predict(data)
    nmi, ari, acc, pur = evaluate(label, y_pred)
    return nmi, ari, acc, pur

def benchmark_2011_JMLR_SKLEARN_GaussianMixture(dataset_name='BDGP', use_view=-1, covariance_type='full', max_iter=100, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]
    
    ## 3. Run the clustering.
    data = data.cpu().numpy()
    label = label.cpu().numpy()
    model = GaussianMixture(n_components=class_num, covariance_type=covariance_type, max_iter=max_iter, random_state=random_state)
    y_pred = model.fit_predict(data)
    nmi, ari, acc, pur = evaluate(label, y_pred)
    return nmi, ari, acc, pur
