import argparse, random, numpy as np, scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from utils import load_data, evaluate

class MDSCN(nn.Module):
    def __init__(self, n_samples, views=2, channel_dims=[3, 1]):
        super(MDSCN, self).__init__()
        self.n_samples = n_samples; self.views = views; self.channel_dims = channel_dims
        self.encoders_comm = []; self.encoders_dive = []; self.decoders_comm = []; self.decoders_dive = []; self.self_express_dive = []
        for v in range(views):
            self.encoders_comm.append(nn.Sequential(nn.Conv2d(in_channels=channel_dims[v], out_channels=64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), nn.ReLU()))
            self.encoders_dive.append(nn.Sequential(nn.Conv2d(in_channels=channel_dims[v], out_channels=64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1), nn.ReLU()))
            self.decoders_comm.append(nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                                    nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                                    nn.ConvTranspose2d(in_channels=64, out_channels=channel_dims[v], kernel_size=3, stride=2, output_padding=1, padding=1)))
            self.decoders_dive.append(nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                                    nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                                    nn.ConvTranspose2d(in_channels=64, out_channels=channel_dims[v], kernel_size=3, stride=2, padding=1, output_padding=1)))
            self.self_express_dive.append(nn.Parameter(1.0e-8 * torch.ones(self.n_samples, self.n_samples, dtype=torch.float32), requires_grad=True))
        self.encoders_comm = nn.ModuleList(self.encoders_comm)
        self.encoders_dive = nn.ModuleList(self.encoders_dive)
        self.decoders_comm = nn.ModuleList(self.decoders_comm)
        self.decoders_dive = nn.ModuleList(self.decoders_dive)
        self.self_express_dive = nn.ParameterList(self.self_express_dive)
        self.self_express_comm = torch.nn.Parameter(1.0e-8 * torch.ones(self.n_samples, self.n_samples, dtype=torch.float32), requires_grad=True)
        
    def forward_pretrain(self, all_views_data):
        recon_input_common_list = []; recon_input_diversity_list = []
        for v in range(self.views):
            latent_common = self.encoders_comm[v](all_views_data[v])
            latent_diversity = self.encoders_dive[v](all_views_data[v])
            recon_common = self.decoders_comm[v](latent_common)
            recon_diversity = self.decoders_dive[v](latent_diversity)
            recon_input_common_list.append(recon_common); recon_input_diversity_list.append(recon_diversity)
        return recon_input_common_list, recon_input_diversity_list

    def forward_finetune(self, all_views_data):
        recon_input_common_list = []; recon_input_diversity_list = []
        latent_common_list = []; latent_diversity_list = []
        recon_latent_common_list = []; recon_latent_diversity_list = []
        coef_diversity_offset_list = []; coef_common_offset = self.self_express_comm - torch.diag(torch.diag(self.self_express_comm))
        for v in range(self.views):
            latent_common = self.encoders_comm[v](all_views_data[v])
            latent_diversity = self.encoders_dive[v](all_views_data[v])
            recon_latent_common = torch.einsum('ij,jchw->ichw', coef_common_offset, latent_common)
            recon_latent_diversity = torch.einsum('ij,jchw->ichw', self.self_express_dive[v] - torch.diag(torch.diag(self.self_express_dive[v])), latent_diversity)
            recon_common = self.decoders_comm[v](recon_latent_common)
            recon_diversity = self.decoders_dive[v](recon_latent_diversity)
            latent_common_list.append(latent_common); latent_diversity_list.append(latent_diversity)
            recon_latent_common_list.append(recon_latent_common); recon_latent_diversity_list.append(recon_latent_diversity)
            recon_input_common_list.append(recon_common); recon_input_diversity_list.append(recon_diversity)
            coef_diversity_offset_list.append(self.self_express_dive[v])
        return recon_input_common_list, recon_input_diversity_list, coef_diversity_offset_list, coef_common_offset, latent_common_list, latent_diversity_list, recon_latent_common_list, recon_latent_diversity_list

def HSIC(c_v, c_w, device): # HSIC loss for two coefficient matrices to enforce the diversity of the latent representations
    # c_v: coefficient matrix of view v, c_w: coefficient matrix of view w, device: device to use
    # c_v.shape: (data_size, data_size), c_w.shape: (data_size, data_size)
    # formula: HSIC(c_v, c_w) = trace(K_1 * H * K_2 * H)
    # trace: the sum of the diagonal elements of the matrix
    N = c_v.shape[0]
    H = torch.ones((N, N)) * ((1 / N) * (-1)) + torch.eye(N) # shape: (data_size, data_size)
    H = H.to(device)
    K_1 = torch.matmul(c_v, c_v.t()).to(device) # shape: (data_size, data_size)
    K_2 = torch.matmul(c_w, c_w.t()).to(device) # shape: (data_size, data_size)
    rst = torch.matmul(K_1, H).to(device) # shape: (data_size, data_size)
    rst = torch.matmul(rst, K_2).to(device) # shape: (data_size, data_size)
    rst = torch.matmul(rst, H).to(device) # shape: (data_size, data_size)
    rst = torch.trace(rst).to(device) # shape: (1,)
    return rst

def subspace_clustering(C, num_clusters, dim_subspace, alpha, sparse_ratio):
    # This function is used to perform subspace clustering on the coefficient matrix C.
    # C: coefficient matrix, num_clusters: number of clusters, dim_subspace: dimension of each subspace, alpha: exponent for the Laplacian matrix, sparse_ratio: threshold for the coefficient matrix
    
    # 1. Threshold the coefficient matrix C to a sparse matrix.
    # Only keep the top abs(C) elements that sum up to ro * 100% of the L1 norm of the coefficients for each column.
    if sparse_ratio < 1:
        C_abs_descending_indices = np.argsort(-np.abs(C), axis=0)
        C_abs_descending_sorted = np.take_along_axis(np.abs(C), C_abs_descending_indices, axis=0)
        C_abs_descending_sorted_cumsum = np.cumsum(C_abs_descending_sorted, axis=0)
        cutoffs_per_column = np.argmax(C_abs_descending_sorted_cumsum > sparse_ratio * C_abs_descending_sorted_cumsum[-1], axis=0) + 1 # shape: (data_size,)
        C_sparse = np.zeros(C.shape) # shape: (data_size, data_size)
        for i in range(C.shape[1]):
            indices_to_keep = C_abs_descending_indices[:cutoffs_per_column[i], i] # shape: (num_samples,)
            C_sparse[indices_to_keep, i] = C[indices_to_keep, i] # shape: (num_samples, data_size)
        C = C_sparse
    
    # 2. Post-process the coefficient matrix C to a clustering-friendly matrix.
    # 2.1. Symmetrize the matrix.
    C = 0.5 * (C + C.T) # symmetrize the matrix
    # 2.2. Perform singular value decomposition.
    r = dim_subspace * num_clusters + 1 # number of singular values to keep
    U, S, _ = svds(A=C, k=min(r, C.shape[0] - 1), v0=np.ones(C.shape[0])) # U: (data_size, r), S: (r,), _: (data_size, data_size)
    # 2.3. Sort the singular values and singular vectors.
    U = U[:, ::-1] # reverse the order of the singular vectors
    S = np.sqrt(S[::-1]) # shape: (r,)
    S = np.diag(S) # shape: (r, r)
    U = U.dot(S) # shape: (data_size, r)
    # 2.4. Normalize the matrix of the singular vectors.
    U = normalize(U, norm='l2', axis=1) # normalize the rows of the matrix to have unit length
    # 2.5. Compute the affinity matrix L.
    Z = U.dot(U.T) # shape: (data_size, data_size)
    Z = Z * (Z > 0) # set the negative values to 0
    L = np.abs(Z ** alpha) # shape: (data_size, data_size)
    L = L / L.max() # normalize the matrix
    L = 0.5 * (L + L.T) # symmetrize the matrix
    # 2.6. Perform spectral clustering.
    spectral = SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize', random_state=66)
    spectral.fit(L) # fit the spectral clustering model
    y_pred = spectral.fit_predict(L) # get the cluster labels
    return y_pred, L

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='RGBDMTV')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_epochs', type=int, default=1000) # 10000 for original code
    parser.add_argument('--finetune_epochs', type=int, default=100) # 100 for original code
    parser.add_argument('--seed', type=int, default=41)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## 1. Set seed for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data.
    dataset, dims, view, data_size, class_num = load_data(args.dataset_name)
    x_list = [torch.from_numpy(dataset.x_1).to(device), torch.from_numpy(dataset.x_2).to(device)] # shape: [(data_size, 3, 64, 64), (data_size, 1, 64, 64)]
    y = torch.from_numpy(dataset.y).to(device) # shape: (data_size,)
    
    # 3. Train the model.
    model = MDSCN(n_samples=data_size, views=view, channel_dims=[3, 1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    criterion_mse = torch.nn.MSELoss(reduction='sum')
    model.train()
    for epoch in range(args.pretrain_epochs):
        recon_input_common_list, recon_input_diversity_list = model.forward_pretrain(x_list)
        loss = sum([0.5 * criterion_mse(recon_input_common_list[v], x_list[v]) for v in range(view)]) + sum([0.5 * criterion_mse(recon_input_diversity_list[v], x_list[v]) for v in range(view)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item():.4f}") if (epoch + 1) % 100 == 0 else None
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    model.train()
    for epoch in range(args.finetune_epochs):
        recon_input_common_list, recon_input_diversity_list, coef_diversity_offset_list, coef_common_offset, latent_common_list, latent_diversity_list, recon_latent_common_list, recon_latent_diversity_list = model.forward_finetune(x_list)
        loss_recon_input = sum([criterion_mse(recon_input_common_list[v], x_list[v]) for v in range(view)]) + sum([criterion_mse(recon_input_diversity_list[v], x_list[v]) for v in range(view)])
        loss_recon_latent = sum([criterion_mse(recon_latent_common_list[v], latent_common_list[v]) for v in range(view)]) + sum([criterion_mse(recon_latent_diversity_list[v], latent_diversity_list[v]) for v in range(view)])
        reg_coef_loss = sum([torch.sum(torch.pow(coef_diversity_offset_list[v], 2.0)) for v in range(view)]) + torch.sum(torch.pow(coef_common_offset, 2.0))
        coef_unify_loss = sum([torch.sum(torch.abs(coef_common_offset - coef_diversity_offset_list[v])) for v in range(view)])
        hsic_loss = HSIC(coef_diversity_offset_list[0], coef_diversity_offset_list[1], device)
        loss = loss_recon_input + loss_recon_latent + 0.1 * reg_coef_loss + 0.1 * coef_unify_loss + 0.1 * hsic_loss # total loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item():.4f}") if (epoch + 1) % 10 == 0 else None
        if (epoch + 1) % 10 == 0:
            ratio = max(0.4 - (class_num - 1) / 10 * 0.1, 0.1)
            y_pred, L = subspace_clustering(coef_common_offset.detach().cpu().numpy(), num_clusters=class_num, dim_subspace=3, alpha=1, sparse_ratio=ratio)
            nmi, ari, acc, pur = evaluate(y.detach().cpu().numpy(), y_pred)
            print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, PUR: {pur:.4f}")
        
    # 4. Plot the coefficient matrix and affinity matrix.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    im1 = ax1.imshow(coef_common_offset.detach().cpu().numpy(), cmap='viridis', aspect='equal')
    fig.colorbar(im1, ax=ax1, extend='both', shrink=0.8).set_label('Intensity', size=16)
    ax1.set_title('Coefficient Matrix C', fontsize=18)
    ax1.set_xlabel('Sample Index', fontsize=14)
    ax1.set_ylabel('Sample Index', fontsize=14)
    im2 = ax2.imshow(L, cmap='viridis', aspect='equal')
    fig.colorbar(im2, ax=ax2, extend='both', shrink=0.8).set_label('Affinity', size=16)
    ax2.set_title('Affinity Matrix L', fontsize=18)
    ax2.set_xlabel('Sample Index', fontsize=14)
    ax2.set_ylabel('Sample Index', fontsize=14)
    fig.tight_layout()
    fig.savefig('result/MDSCN_coef_and_affinity.png', dpi=150, bbox_inches='tight')