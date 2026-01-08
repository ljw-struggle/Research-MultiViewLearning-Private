import math, os, h5py, argparse
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment
# import warnings
# warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        x = x + self.sigma * torch.randn_like(x) if self.training else x
        return x

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
    
activation_dict = {'relu': nn.ReLU(), 'selu': nn.SELU(), 'sigmoid': nn.Sigmoid(), 'elu': nn.ELU(), 'mean_act': MeanAct(), 'disp_act': DispAct()}
    
############################################################################################################################################################################

def read_dataset(adata, transpose=False, test_split=False, copy=False):
    assert adata.X.size < 50e6, 'The dataset is too large. Please check if the dataset is too large.'
    assert np.all(adata.X.astype(int) == adata.X), 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    if copy: adata = adata.copy()
    if transpose: adata = adata.transpose()
    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        obs = pd.Series(['train' if i in train_idx else 'test' for i in range(adata.n_obs)])
        adata.obs['data_split'] = obs.values
    else:
        adata.obs['data_split'] = ['train' for i in range(adata.n_obs)]
    adata.obs['data_split'] = adata.obs['data_split'].astype('category')
    print('Successfully loaded data with {} genes and {} cells'.format(adata.n_vars, adata.n_obs))
    return adata

def clr_normalize_each_cell(adata):
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts) # because this is calculated on the normalized data, so the size factor will be 1
    seurat_clr = lambda x: np.log1p(x / np.exp(np.sum(np.log1p(x[x > 0])) / len(x)))
    adata.X = np.apply_along_axis(seurat_clr, 1, adata.X)
    return adata
    
def normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True):
    sc.pp.filter_genes(adata, min_counts=1) # filter genes detected with less than 1 count
    sc.pp.filter_cells(adata, min_counts=1) # filter cells detected with less than 1 count
    adata.raw = adata.copy()
    if size_factors:
        sc.pp.normalize_total(adata, target_sum=1e4) # normalize total UMI counts to 1e4
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts) # n_counts is the total counts per cell before normalization
    else:
        adata.obs['size_factors'] = 1.0
    sc.pp.log1p(adata) if logtrans_input else None # log1p transformation.
    sc.pp.scale(adata) if normalize_input else None # z-score normalization for each var.
    return adata
   
def get_cluster_number(X, resolution, n_neighbors=30):
    adata=sc.AnnData(X)
    if adata.shape[0] > 200000: # if the number of cells is larger than 200000, randomly select 200000 cells
       np.random.seed(adata.shape[0]) # set seed 
       adata=adata[np.random.choice(adata.shape[0], 200000, replace=False)] # randomly select 200000 cells
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.louvain(adata, resolution=resolution) # clustering
    clustering_result = adata.obs['louvain'] 
    clustering_result = np.asarray(clustering_result, dtype=int)
    assert np.unique(clustering_result).shape[0] > 1, "Error: There is only a cluster detected. The resolution:" + str(resolution) + "is too large, please choose a smaller resolution!!"
    print("Estimated n_clusters is: ", len(np.unique(clustering_result)))
    return len(np.unique(clustering_result))
    
def best_map(L1, L2): # L1 should be the groundtruth labels and L2 should be the clustering labels we got, using *Hungarian algorithm* to find the best match
    L1 = L1.astype(np.int64); L2 = L2.astype(np.int64)
    label_1 = np.unique(L1); n_class_1 = len(label_1); label_2 = np.unique(L2); n_class_2 = len(label_2); n_class = max(n_class_1, n_class_2)
    G = np.zeros((n_class, n_class)) # G is the confusion matrix 
    for i in range(n_class_1):
        index_class_1 = (L1 == label_1[i]) 
        index_class_1 = index_class_1.astype(float)
        for j in range(n_class_2):
            index_class_2 = (L2 == label_2[j])
            index_class_2 = index_class_2.astype(float)
            G[i, j] = np.sum(index_class_2 * index_class_1)
    m = Munkres() # Hungarian algorithm
    indexes = m.compute(-G.T) # find the best match with the minimum cost (row is the cluster, column is the groundtruth)
    indexes = np.array(indexes)
    best_match_L2 = np.zeros(len(L2))
    for i in range(n_class_2):
        best_match_L2[L2 == label_2[indexes[i][0]]] = label_1[indexes[i][1]] # the best match of L2 is the corresponding label in L1
    return best_match_L2 # return the best match of L2
 
def gene_selection(data, n=None, decay=1.5, xoffset=5, yoffset=0.02, markers=None, genes=None, plot=True):
    threshold = 0
    data = data.toarray() if sp.sparse.issparse(data) else data
    low_detection_genes = np.sum(data > threshold, axis=0) < 10 # genes detected in less than 10 cells
    gene_zero_rate = 1 - np.mean(data > threshold, axis=0); gene_zero_rate[low_detection_genes] = np.nan 
    data_log2_nonzero = data.copy(); data_log2_nonzero[data_log2_nonzero <= threshold] = np.nan; data_log2_nonzero = np.log2(data_log2_nonzero)
    gene_mean_expr = np.nanmean(data_log2_nonzero, axis=0); gene_mean_expr[low_detection_genes] = np.nan
    # 1\ select genes based on the zero rate and mean expression
    if n is None:
        selected = np.zeros_like(gene_zero_rate).astype(bool)
        selected[~np.isnan(gene_zero_rate)] = gene_zero_rate[~np.isnan(gene_zero_rate)] > np.exp(-decay*(gene_mean_expr[~np.isnan(gene_zero_rate)] - xoffset)) + yoffset
    else:
        up = 10; low = 0
        for _ in range(100):
            nonan = ~np.isnan(gene_zero_rate)
            selected = np.zeros_like(gene_zero_rate).astype(bool)
            selected[nonan] = gene_zero_rate[nonan] > np.exp(-decay*(gene_mean_expr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n: break
            if np.sum(selected) < n: up = xoffset; xoffset = (xoffset + low)/2
            if np.sum(selected) > n: low = xoffset; xoffset = (xoffset + up)/2
    # 2\ plot the selection curve
    if plot:
        plt.figure(figsize=(6, 3.5))
        x = np.arange(0, np.ceil(np.nanmax(gene_mean_expr)) + 0.1, 0.1) 
        y = np.exp(-decay*(x - xoffset)) + yoffset
        selected_area = np.concatenate((np.concatenate((x[:,None], y[:,None]),axis=1), np.array([[np.ceil(np.nanmax(gene_mean_expr)), 1]])), axis=0)
        plt.gca().add_patch(plt.matplotlib.patches.Polygon(selected_area, color=sns.color_palette()[1], alpha=0.4)) # plot the selected area
        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        plt.scatter(gene_mean_expr, gene_zero_rate, s=1, alpha=1, rasterized=True)
        plt.xlim([0, np.ceil(np.nanmax(gene_mean_expr))])
        plt.ylim([0, 1])
        plt.xlabel('Mean log2 nonzero expression') if threshold == 0 else plt.xlabel('Mean log2 nonzero expression')
        plt.ylabel('Frequency of zero expression') if threshold == 0 else plt.ylabel('Frequency of near-zero expression')
        plt.text(0.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected), decay, xoffset, yoffset), color='k', fontsize=10, transform=plt.gca().transAxes)
        plt.tight_layout()
        if markers is not None and genes is not None:
            for marker in markers:
                i = np.where(genes==marker)[0]
                if ~np.isnan(gene_zero_rate[i]): # plot the markers that are not NaN
                    plt.scatter(gene_mean_expr[i], gene_zero_rate[i], s=10, color='k')
                    plt.text(gene_mean_expr[i]+0.12, gene_zero_rate[i]+0.02, marker, color='k', fontsize=10)
        plt.savefig('gene_selection.png', dpi=300)
    return selected

def cluster_acc(y_true, y_pred): # calculate clustering accuracy based on the best match between true labels and predicted labels
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64)
    n_class = (max(y_pred.max(), y_true.max()) + 1).astype(int)
    w = np.zeros((n_class, n_class), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w) # the Hungarian algorithm
    indexes = np.array(list(zip(row_ind, col_ind)))
    return sum([w[i, j] for i, j in indexes]) * 1.0 / w.sum()

############################################################################################################################################################################

class NBLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super(NBLoss, self).__init__()
        self.eps = eps

    def forward(self, x, mean, r, scale_factor): # x: true value, mean: predicted value, r: dispersion parameter
        # term_1 = torch.lgamma(r + x + self.eps) - torch.lgamma(x + 1.0) - torch.lgamma(r + self.eps)
        # term_2 = r * torch.log((r + self.eps)/(r + mean + self.eps)) + x * torch.log((mean + self.eps)/(r + mean + self.eps)) 
        # return -torch.mean(term_1 + term_2) # equivalent to the following
        mean = mean * scale_factor[:, None]
        term_1 = torch.lgamma(r + self.eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + r + self.eps)
        term_2 = (r + x) * torch.log(1.0 + (mean/(r + self.eps))) + (x * (torch.log(r + self.eps) - torch.log(mean + self.eps)))
        return torch.mean(term_1 + term_2)

class ZINBLoss(nn.Module):
    def __init__(self, eps=1e-10):
        super(ZINBLoss, self).__init__()
        self.eps = eps

    def forward(self, x, mean, r, pi, scale_factor, ridge_lambda=0.0): # x: true value, mean: predicted value, r: dispersion parameter, pi: zero inflation parameter, ridge_lambda: l2 regularization parameter, like ridge regression
        mean = mean * scale_factor[:, None]
        term_1 = torch.lgamma(r + self.eps) + torch.lgamma(x + 1.0) - torch.lgamma(x+r + self.eps)
        term_2 = (r + x) * torch.log(1.0 + (mean/(r + self.eps))) + (x * (torch.log(r + self.eps) - torch.log(mean + self.eps)))
        nb_loss = term_1 + term_2
        nb_case = nb_loss - torch.log(1.0 - pi + self.eps) # the paper implementation
        # nb_case = - torch.log(1 - pi + self.eps) - nb_loss # the right implementation
        zero_nb_loss = torch.pow(r/(r + mean + self.eps), r)
        zero_nb_case = -torch.log(pi + (1.0 - pi) * zero_nb_loss + self.eps)
        result = torch.where(torch.le(x, 1e-8), zero_nb_case, nb_case)
        result = (result + ridge_lambda * torch.square(pi)) if ridge_lambda > 0 else result
        return torch.mean(result)

############################################################################################################################################################################

def build_block(layer_dim_list, activation='relu'):
    block = []
    for i in range(1, len(layer_dim_list)):
        block.append(nn.Linear(layer_dim_list[i-1], layer_dim_list[i]))
        block.append(nn.BatchNorm1d(layer_dim_list[i], affine=True))
        block.append(activation_dict[activation])
    block = nn.Sequential(*block)
    return block

class scMultiCluster(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, encode_layer=[], decode_layer_1=[], decode_layer_2=[], 
                 activation='elu', tau=1.0, sigma_1=2.5, sigma_2=0.1, alpha=1.0, gamma=1.0, phi_1=0.0001, phi_2=0.0001, cutoff=0.5, save_dir='./result'):
        super(scMultiCluster, self).__init__()
        self.input_dim_1 = input_dim_1; self.input_dim_2 = input_dim_2; self.tau = tau; self.activation = activation; self.z_dim = encode_layer[-1]; self.zinb_loss = ZINBLoss(); 
        self.sigma_1 = sigma_1; self.sigma_2 = sigma_2; self.alpha = alpha; self.gamma = gamma; self.phi_1 = phi_1; self.phi_2 = phi_2; self.tau=tau; self.cutoff = cutoff; self.save_dir = save_dir
        self.encoder = build_block([input_dim_1 + input_dim_2] + encode_layer, activation)
        self.decoder_1 = build_block(decode_layer_1, activation)
        self.decoder_mean_1 = nn.Sequential(nn.Linear(decode_layer_1[-1], input_dim_1), activation_dict['mean_act'])
        self.decoder_disp_1 = nn.Sequential(nn.Linear(decode_layer_1[-1], input_dim_1), activation_dict['disp_act'])
        self.decoder_pi_1 = nn.Sequential(nn.Linear(decode_layer_1[-1], input_dim_1), activation_dict['sigmoid'])
        self.decoder_2 = build_block(decode_layer_2, activation)
        self.decoder_mean_2 = nn.Sequential(nn.Linear(decode_layer_2[-1], input_dim_2), activation_dict['mean_act'])
        self.decoder_disp_2 = nn.Sequential(nn.Linear(decode_layer_2[-1], input_dim_2), activation_dict['disp_act'])
        self.decoder_pi_2 = nn.Sequential(nn.Linear(decode_layer_2[-1], input_dim_2), activation_dict['sigmoid'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); self.to(self.device)
        
    def forward(self, x1, x2):
        x_noise = torch.cat([x1 + torch.randn_like(x1) * self.sigma_1, x2 + torch.randn_like(x2) * self.sigma_2], dim=-1) # add gaussian noise, shape: (batch_size, input_dim_1 + input_dim_2)
        h_noise = self.encoder(x_noise)
        h_1 = self.decoder_1(h_noise); mean_1 = self.decoder_mean_1(h_1); disp_1 = self.decoder_disp_1(h_1); pi_1 = self.decoder_pi_1(h_1)
        h_2 = self.decoder_2(h_noise); mean_2 = self.decoder_mean_2(h_2); disp_2 = self.decoder_disp_2(h_2); pi_2 = self.decoder_pi_2(h_2)
        x = torch.cat([x1, x2], dim=-1)
        z = self.encoder(x) # latent representation, shape: (batch_size, z_dim)
        z_square = torch.sum(torch.square(z), dim=1, keepdim=True) # shape: (batch_size, 1)
        euclidean_distance = z_square + z_square.t() - 2.0 * torch.matmul(z, z.t()) # shape: (batch_size, batch_size)
        matrix = euclidean_distance / self.alpha # shape: (batch_size, batch_size)
        matrix = torch.pow(1.0 + matrix, -(self.alpha + 1.0) / 2.0) # shape: (batch_size, batch_size)
        zerodiag_matrix = matrix - torch.diag(torch.diag(matrix)) # shape: (batch_size, batch_size), remove diagonal elements
        latent_q = zerodiag_matrix / torch.sum(zerodiag_matrix, dim=1, keepdim=True) # the probability of each pair of cells, shape: (batch_size, batch_size)
        return z, matrix, latent_q, mean_1, mean_2, disp_1, disp_2, pi_1, pi_2
    
    def encode(self, X1, X2, batch_size=256):
        self.eval()
        encoded = []
        num_sample = X1.shape[0]; num_batch = int(math.ceil(1.0 * X1.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            inputs_1 = X1[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
            inputs_2 = X2[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
            z, _, _, _, _, _, _, _, _ = self.forward(inputs_1, inputs_2)
            encoded.append(z.data)
        return torch.cat(encoded, dim=0)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict}) 
        self.load_state_dict(model_dict)
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0, keepdim=True) # shape: (batch_size, batch_size)
        return p / p.sum(1, keepdim=True) # shape: (batch_size, batch_size)

    def kld_loss(self, p, q): # KL divergence between two distributions
        return torch.mean(torch.sum(p * torch.log(p/q), dim=-1)) # torch.mean(torch.sum(p * torch.log(p), dim=-1) - torch.sum(p * torch.log(q), dim=-1))

    def kmeans_loss(self, z): # z: latent representation, shape: (batch_size, z_dim), self.mu: cluster centers, shape: (n_clusters, z_dim)
        dist_1 = self.tau * torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2) # shape: (batch_size, n_clusters)
        temp_dist_1 = dist_1 - torch.mean(dist_1, dim=1, keepdim=True) # shape: (batch_size, n_clusters), remove mean
        q = torch.exp(- temp_dist_1) # shape: (batch_size, n_clusters) 
        q = q / torch.sum(q, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        q = torch.pow(q, 2); # shape: (batch_size, n_clusters)
        q = q / torch.sum(q, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        dist_2 = dist_1 * q # shape: (batch_size, n_clusters)
        return dist_1, torch.mean(torch.sum(dist_2, dim=1)) # shape: (batch_size, n_clusters), shape: (1)
    
    def pretrain_autoencoder(self, X_1, X_1_raw, X_1_size_factor, X_2, X_2_raw, X_2_size_factor, batch_size=256, lr=0.001, epoch_num=400):
        print('Pretraining stage for scMultiCluster.')
        dataset = TensorDataset(torch.Tensor(X_1), torch.Tensor(X_1_raw), torch.Tensor(X_1_size_factor), torch.Tensor(X_2), torch.Tensor(X_2_raw), torch.Tensor(X_2_size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        num = X_1.shape[0] # number of cells
        self.train()
        for epoch in range(epoch_num):
            loss_val = 0; loss_val_recon_1 = 0; loss_val_recon_2 = 0; loss_val_kld = 0
            for batch_idx, (x_1_batch, x_1_raw_batch, x_1_size_factor_batch, x_2_batch, x_2_raw_batch, x_2_size_factor_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                x_1 = x_1_batch.to(self.device); x_1_raw = x_1_raw_batch.to(self.device); x_1_size_factor = x_1_size_factor_batch.to(self.device)
                x_2 = x_2_batch.to(self.device); x_2_raw = x_2_raw_batch.to(self.device); x_2_size_factor = x_2_size_factor_batch.to(self.device)
                z_batch, z_batch_matrix, z_batch_latent_q, mean_1_batch, mean_2_batch, disp_1_batch, disp_2_batch, pi_1_batch, pi_2_batch = self.forward(x_1, x_2)
                # print(torch.isnan(z_batch).sum(), torch.isnan(z_batch_matrix).sum(), torch.isnan(z_batch_latent_q).sum(), torch.isnan(mean_1_batch).sum(), torch.isnan(mean_2_batch).sum(), torch.isnan(disp_1_batch).sum(), torch.isnan(disp_2_batch).sum(), torch.isnan(pi_1_batch).sum(), torch.isnan(pi_2_batch).sum())
                recon_loss_1 = self.zinb_loss(x=x_1_raw, mean=mean_1_batch, r=disp_1_batch, pi=pi_1_batch, scale_factor=x_1_size_factor)
                recon_loss_2 = self.zinb_loss(x=x_2_raw, mean=mean_2_batch, r=disp_2_batch, pi=pi_2_batch, scale_factor=x_2_size_factor)
                latent_p_batch = self.target_distribution(z_batch_latent_q); latent_p_batch = latent_p_batch + torch.diag(torch.diag(z_batch_matrix)); latent_q_batch = z_batch_latent_q + torch.diag(torch.diag(z_batch_matrix))
                kld_loss = self.kld_loss(latent_p_batch, latent_q_batch)
                # print(torch.isnan(recon_loss_1).sum(), torch.isnan(recon_loss_2).sum(), torch.isnan(kld_loss).sum())
                loss = recon_loss_1 + recon_loss_2 + kld_loss * self.phi_1 if epoch + 1 >= epoch_num * self.cutoff else recon_loss_1 + recon_loss_2
                loss.backward()
                # print(torch.isnan(self.encoder[0].weight.grad).sum(), torch.isnan(self.decoder_1[0].weight.grad).sum(), torch.isnan(self.decoder_mean_1[0].weight.grad).sum(), torch.isnan(self.decoder_disp_1[0].weight.grad).sum(), torch.isnan(self.decoder_pi_1[0].weight.grad).sum())
                optimizer.step()
                loss_val += loss.item() * len(x_1_batch); loss_val_recon_1 += recon_loss_1.item() * len(x_1_batch); loss_val_recon_2 += recon_loss_2.item() * len(x_2_batch)
                loss_val_kld += kld_loss.item() * len(x_1_batch) if epoch + 1 >= epoch_num * self.cutoff else 0
            loss_val = loss_val / num; loss_val_recon_1 = loss_val_recon_1 / num; loss_val_recon_2 = loss_val_recon_2 / num; loss_val_kld = loss_val_kld / num
            print('Pretrain epoch {}: Total loss={:.6f}, ZINB_loss_1={:.6f}, ZINB_loss_2={:.6f}, KLD_loss={:.6f}'.format(epoch + 1, loss_val, loss_val_recon_1, loss_val_recon_2, loss_val_kld)) if epoch % 10 == 0 else None
        torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(self.save_dir, 'pretrain_checkpoint_{:d}.pth'.format(epoch+1)))

    def fit(self, X_1, X_1_raw, X_1_size_factor, X_2, X_2_raw, X_2_size_factor, y=None, lr=1.0, n_clusters = 4, batch_size=256, epoch_num=10, update_interval=1, tol=1e-3):
        print("Clustering stage for scMultiCluster.")
        print("Initializing cluster centers with kmeans.")
        X_1 = torch.tensor(X_1, dtype=torch.float32).to(self.device); X_1_raw = torch.tensor(X_1_raw, dtype=torch.float32).to(self.device); X_1_size_factor = torch.tensor(X_1_size_factor, dtype=torch.float32).to(self.device)
        X_2 = torch.tensor(X_2, dtype=torch.float32).to(self.device); X_2_raw = torch.tensor(X_2_raw, dtype=torch.float32).to(self.device); X_2_size_factor = torch.tensor(X_2_size_factor, dtype=torch.float32).to(self.device)
        Z = self.encode(X_1, X_2, batch_size=batch_size)
        kmeans = KMeans(n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(Z.data.cpu().numpy()); y_pred_last = y_pred
        self.mu = nn.Parameter(torch.Tensor(n_clusters, self.z_dim), requires_grad=True).to(self.device) # add cluster centers as parameters
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_)) # initialize cluster centers
        if y is not None:
            ami = np.round(adjusted_mutual_info_score(y, y_pred), 5); nmi = np.round(normalized_mutual_info_score(y, y_pred), 5); ari = np.round(adjusted_rand_score(y, y_pred), 5)
            print('Initializing k-means: AMI={:.4f}, NMI={:.4f}, ARI={:.4f}'.format(ami, nmi, ari))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=0.95)
        self.train()
        num_sample = X_1.shape[0]; num_batch = int(math.ceil(1.0*X_1.shape[0]/batch_size))
        for epoch in range(epoch_num):
            if epoch % update_interval == 0:
                Z = self.encode(X_1, X_2, batch_size=batch_size)
                dist, _ = self.kmeans_loss(Z)
                y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5); 
                    ami = np.round(adjusted_mutual_info_score(y, y_pred), 5); nmi = np.round(normalized_mutual_info_score(y, y_pred), 5); ari = np.round(adjusted_rand_score(y, y_pred), 5)
                    print('Clustering {:d}: ACC={:.4f}, AMI={:.4f}, NMI={:.4f}, ARI={:.4f}'.format(epoch, acc, ami, nmi, ari))
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num_sample # check convergence
                y_pred_last = y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label:', delta_label, '< tol:', tol, '. Reached tolerance threshold. Stopping training.')
                    # torch.save({'epoch': epoch+1, 'state_dict': self.state_dict(), 'mu': self.mu, 'y_pred': y_pred, 'y_pred_last': y_pred_last, 'y': y }, os.path.join(self.save_dir, 'finetune_checkpoint_{:d}.pth'.format(epoch+1)))
                    break
            loss_val = 0; loss_val_recon_1 = 0; loss_val_recon_2 = 0; loss_val_kld = 0; loss_val_cluster = 0
            for batch_idx in range(num_batch):
                x_1 = X_1[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]; x_2 = X_2[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
                x_1_raw = X_1_raw[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]; x_2_raw = X_2_raw[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
                x_1_size_factor = X_1_size_factor[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]; x_2_size_factor = X_2_size_factor[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
                z_batch, z_batch_matrix, z_batch_latent_q, mean_1_batch, mean_2_batch, disp_1_batch, disp_2_batch, pi_1_batch, pi_2_batch = self.forward(x_1, x_2)
                _, cluster_loss = self.kmeans_loss(z_batch)
                recon_loss_1 = self.zinb_loss(x=x_1_raw, mean=mean_1_batch, r=disp_1_batch, pi=pi_1_batch, scale_factor=x_1_size_factor)
                recon_loss_2 = self.zinb_loss(x=x_2_raw, mean=mean_2_batch, r=disp_2_batch, pi=pi_2_batch, scale_factor=x_2_size_factor)
                latent_p_batch = self.target_distribution(z_batch_latent_q); latent_p_batch = latent_p_batch + torch.diag(torch.diag(z_batch_matrix)); latent_q_batch = z_batch_latent_q + torch.diag(torch.diag(z_batch_matrix))
                kld_loss = self.kld_loss(latent_p_batch, latent_q_batch) # latent_p_batch is the target distribution, latent_q_batch is the predicted distribution
                loss = recon_loss_1 + recon_loss_2 + kld_loss * self.phi_2 + cluster_loss * self.gamma
                optimizer.zero_grad()
                loss.backward(); # torch.nn.utils.clip_grad_norm_(self.mu, 1) # clip gradient
                optimizer.step()
                loss_val += loss.data * len(x_1); loss_val_cluster += cluster_loss.data * len(x_1); loss_val_recon_1 += recon_loss_1.data * len(x_1); loss_val_recon_2 += recon_loss_2.data * len(x_2); loss_val_kld += kld_loss.data * len(x_1);
            loss_val = loss_val / num_sample; loss_val_cluster = loss_val_cluster / num_sample; loss_val_recon_1 = loss_val_recon_1 / num_sample; loss_val_recon_2 = loss_val_recon_2 / num_sample; loss_val_kld = loss_val_kld / num_sample
            print('Epoch {:d}: total_loss={:.6f}, clustering_loss={:.6f}, ZINB_loss_1={:.6f}, ZINB_loss_2={:.6f}, KLD_loss={:.6f}'.format(epoch+1, loss_val, loss_val_cluster, loss_val_recon_1, loss_val_recon_2, loss_val_kld)) if epoch % 10 == 0 else None
        return y_pred

############################################################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='./data/CITESeq_GSE128639_BMNC_anno.h5')
    parser.add_argument('--pretrain_weight_file', default='None', type=str)
    parser.add_argument('--save_dir', default='./result/')
    parser.add_argument('--run', default=1, type=int) # the number of runs
    parser.add_argument('-el','--encode_layer', nargs='+', default=[256,64,32,16])
    parser.add_argument('-dl1','--decode_layer_1', nargs='+', default=[16,64,256])
    parser.add_argument('-dl2','--decode_layer_2', nargs='+', default=[16,20])
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--finetune_epochs', default=5000, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--gamma', default=.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float, help='fuzziness of clustering loss')          
    parser.add_argument('--sigma_1', default=2.5, type=float)
    parser.add_argument('--sigma_2', default=1.5, type=float)          
    parser.add_argument('--phi_1', default=0.001, type=float, help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi_2', default=0.001, type=float, help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_clusters', default=27, type=int)
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--no_labels', action='store_true', default=False)
    parser.add_argument('--filter_1', action='store_true', default=False, help='Do mRNA selection')
    parser.add_argument('--filter_2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--filter_1_num', default=1000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--filter_2_num', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1\ preprocessing scRNA-seq read counts matrix
    with h5py.File(args.data_file, 'r') as f:
        x1 = np.array(f['X1']); x2 = np.array(f['X2']); y = np.array(f['Y']) if not args.no_labels else None
    x1 = x1[:, gene_selection(x1, n=args.filter_1_num, plot=False)] if args.filter_1 else x1
    x2 = x2[:, gene_selection(x2, n=args.filter_2_num, plot=False)] if args.filter_2 else x2
    adata1 = sc.AnnData(x1); # adata1.obs['Group'] = y
    adata1 = read_dataset(adata1, transpose=False, test_split=False, copy=True)
    adata1 = normalize(adata1, size_factors=True, normalize_input=True, logtrans_input=True)
    adata2 = sc.AnnData(x2); # adata2.obs['Group'] = y
    adata2 = read_dataset(adata2, transpose=False, test_split=False, copy=True)
    adata2 = normalize(adata2, size_factors=True, normalize_input=True, logtrans_input=True)
    
    # 2\ Pretrain scMultiCluster
    model = scMultiCluster(input_dim_1=adata1.n_vars, input_dim_2=adata2.n_vars, encode_layer=args.encode_layer, decode_layer_1=args.decode_layer_1, decode_layer_2=args.decode_layer_2,
                           activation='elu', tau=args.tau, sigma_1=args.sigma_1, sigma_2=args.sigma_2, gamma=args.gamma, phi_1=args.phi_1, phi_2=args.phi_2, cutoff = args.cutoff, save_dir=args.save_dir)
    if not os.path.exists(args.pretrain_weight_file):
        model.pretrain_autoencoder(X_1=adata1.X, X_1_raw=adata1.raw.X, X_1_size_factor=adata1.obs.size_factors.to_numpy(), X_2=adata2.X, X_2_raw=adata2.raw.X, X_2_size_factor=adata2.obs.size_factors.to_numpy(), 
                                   batch_size=args.batch_size, epoch_num=args.pretrain_epochs)
    else:
        print("==> loading checkpoint '{}'".format(args.ptrain_weight_file))
        model.load_state_dict(torch.load(args.pretrain_weight_file)['state_dict'])
    
    # 3\ Fine-tune scMultiCluster according to the clustering loss that makes the latent representation cluster well (refine the representation and cluster results)
    latent = model.encode(torch.tensor(adata1.X, dtype=torch.float32).to(model.device), torch.tensor(adata2.X, dtype=torch.float32).to(model.device)).cpu().numpy()
    n_clusters = get_cluster_number(latent, res=args.resolution, n=args.n_neighbors) if args.n_clusters == -1 else args.n_clusters
    y_pred = model.fit(X_1=adata1.X, X_1_raw=adata1.raw.X, X_1_size_factor=adata1.obs.size_factors.to_numpy(), X_2=adata2.X, X_2_raw=adata2.raw.X, X_2_size_factor=adata2.obs.size_factors.to_numpy(), 
                       y=y, n_clusters=n_clusters, batch_size=args.batch_size, epoch_num=args.finetune_epochs, update_interval=args.update_interval, tol=args.tol, lr=args.lr)
    y_pred_best_map = best_map(y, y_pred) if not args.no_labels else y_pred.astype(int)
    np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred_best_map, delimiter=",")
    final_latent = model.encode(torch.tensor(adata1.X, dtype=torch.float32).to(model.device), torch.tensor(adata2.X, dtype=torch.float32).to(model.device)).cpu().numpy()
    np.savetxt(args.save_dir + "/" + str(args.run) + "_embedding.csv", final_latent, delimiter=",")
    if not args.no_labels:
        ami = np.round(adjusted_mutual_info_score(y, y_pred), 5); nmi = np.round(normalized_mutual_info_score(y, y_pred), 5); ari = np.round(adjusted_rand_score(y, y_pred), 5)
        print('Final: AMI= {:.4f}, NMI= {:.4f}, ARI= {:.4f}'.format(ami, nmi, ari))
      