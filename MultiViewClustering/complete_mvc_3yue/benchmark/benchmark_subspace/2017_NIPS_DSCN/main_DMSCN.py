import argparse, random, numpy as np, scipy.io as sio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from utils import load_data, evaluate
    
class ConvAEwithSR(nn.Module):
    def __init__(self, samples, views):
        super(ConvAEwithSR, self).__init__()
        self.view_number = views
        self.batch_size = samples
        self.encoders = []; self.decoders = []
        for v in range(views):
            self.encoders.append(nn.Sequential(nn.Conv2d(1, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                                               nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),nn.ReLU(),
                                               nn.Conv2d(20, 30, kernel_size=3, stride=2, padding=1),nn.ReLU()))
            self.decoders.append(nn.Sequential(nn.ConvTranspose2d(30, 20, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                               nn.ConvTranspose2d(20, 10, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
                                               nn.ConvTranspose2d(10, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU()))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.coef = nn.Parameter(torch.ones(samples, samples, dtype=torch.float32) * 1e-8, requires_grad=True)
        
    def forward_pretrain(self, multi_view_data):
        recon_list = []; latent_list = []
        for v in range(self.view_number):
            latent = self.encoders[v](multi_view_data[v]); recon = self.decoders[v](latent)
            recon_list.append(recon); latent_list.append(latent)
        return recon_list, latent_list

    def forward_finetune(self, multi_view_data):
        latent_list = []; latent_coef_list = []; recon_list = []
        # coef_offset = self.coef # no offset
        coef_offset = self.coef - torch.diag(torch.diag(self.coef)) # offset the diagonal elements of the common expression matrix to 0
        for v in range(self.view_number):
            latent = self.encoders[v](multi_view_data[v])
            latent_coef = torch.einsum('ij,jchw->ichw', coef_offset, latent)
            recon = self.decoders[v](latent_coef)
            latent_list.append(latent); latent_coef_list.append(latent_coef); recon_list.append(recon)
        return latent_list, latent_coef_list, recon_list, coef_offset

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
    spectral = SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize', random_state=None)
    spectral.fit(L) # fit the spectral clustering model
    y_pred = spectral.fit_predict(L) # get the cluster labels
    return y_pred, L

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='EYB_FC')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--pretrain_epochs', type=int, default=10000) # 10000 for original code
    parser.add_argument('--finetune_epochs', type=int, default=2000) # 2000 for original code
    parser.add_argument('--seed', type=int, default=0)
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
    x_list = [torch.from_numpy(dataset.x_1).to(device), torch.from_numpy(dataset.x_2).to(device), torch.from_numpy(dataset.x_3).to(device), torch.from_numpy(dataset.x_4).to(device), torch.from_numpy(dataset.x_5).to(device)] # shape: view * (data_size, 1, 32, 32).
    y = torch.from_numpy(dataset.y).to(device) # shape: (data_size,).
    
    # 2. Pretrain AE
    model = ConvAEwithSR(samples=data_size, views=view).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    criterion_mse = torch.nn.MSELoss()
    for epoch in range(args.pretrain_epochs):
        model.train()
        optimizer.zero_grad()
        recon_list, latent_list = model.forward_pretrain(x_list)
        loss = sum([criterion_mse(recon_list[v], x_list[v]) for v in range(view)])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item():.4f}") if (epoch + 1) % 100 == 0 else None

    # 3. Train AE with Self-Expression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.)
    criterion_mse = torch.nn.MSELoss()
    lambda_1 = 1.0 / 2
    lambda_2 = 1.0 * 10 ** (class_num / 10.0 - 3.0) / 2
    for epoch in range(args.finetune_epochs):
        model.train()
        latent_list, latent_coef_list, recon_list, coef_offset = model.forward_finetune(x_list)
        rec_data_loss = sum([criterion_mse(recon_list[v], x_list[v]) for v in range(view)])
        rec_latent_loss = sum([criterion_mse(latent_coef_list[v], latent_list[v]) for v in range(view)])
        reg_coef_loss = torch.sum(torch.pow(coef_offset, 2.0))
        loss = lambda_1 * rec_latent_loss + lambda_2 * rec_data_loss + reg_coef_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss: {loss.item():.4f}") if (epoch + 1) % 100 == 0 else None
        if (epoch + 1) % 100 == 0:
            ratio = max(0.4 - (class_num - 1) / 10 * 0.1, 0.1)
            y_pred, L = subspace_clustering(coef_offset.detach().cpu().numpy(), num_clusters=class_num, dim_subspace=6, alpha=8, sparse_ratio=ratio)
            nmi, ari, acc, pur = evaluate(y.detach().cpu().numpy(), y_pred)
            print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, PUR: {pur:.4f}")
    
    # 4. Plot the coefficient matrix and affinity matrix.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    im1 = ax1.imshow(coef_offset.detach().cpu().numpy(), cmap='viridis', aspect='equal')
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
    fig.savefig('result/DMSCN_coef_and_affinity.png', dpi=150, bbox_inches='tight')
    