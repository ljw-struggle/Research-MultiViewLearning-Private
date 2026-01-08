import random, argparse, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from _utils import load_data, evaluate

class MLP(nn.Module):
    def __init__(self, dim_list):
        super().__init__()
        layers = nn.ModuleList([])
        for i in range(len(dim_list)-2):
            layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
            layers.append(nn.BatchNorm1d(dim_list[i+1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(dim_list[-2], dim_list[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DeepGCCA(nn.Module):
    def __init__(self, dim_list_list):
        """
        dim_list_list: list of dimension lists, one for each view
        e.g., for 3 views: [[dims[0], 128, 64, 32, outdim], [dims[1], 128, 64, 32, outdim], [dims[2], 128, 64, 32, outdim]]
        """
        super().__init__()
        self.model_list = nn.ModuleList([MLP(dim_list) for dim_list in dim_list_list])
        
    def forward(self, x_list):
        """
        x_list: list of tensors, each x[i].shape: (batch_size, n_features_i)
        Returns: list of encoded tensors, each z[i].shape: (batch_size, outdim_size)
        """
        output_list = []
        for x, model in zip(x_list, self.model_list):
            output_list.append(model(x))
        return output_list
    
class DGCCALoss(nn.Module):
    def __init__(self, outdim_size, r=1e-4, eps=1e-8, use_all_singular_values=False):
        super(DGCCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.r = r
        self.eps = eps
        self.use_all_singular_values = use_all_singular_values

    def forward(self, H_list): # H_list: list of tensors, each H[i].shape: (n_samples, n_features)
        """
        Compute GCCA loss for multiple views
        """
        top_k = self.outdim_size
        AT_list = []
        
        for H in H_list:
            # Center the data
            Hbar = H - H.mean(dim=0, keepdim=True)  # Hbar.shape: (n_samples, n_features)
            
            # SVD: Hbar = A @ S @ B^T
            A, S, B = torch.linalg.svd(Hbar, full_matrices=False)
            
            # Take top_k components
            top_k_actual = min(top_k, S.shape[0])
            A = A[:, :top_k_actual]  # A.shape: (n_samples, top_k_actual)
            S_thin = S[:top_k_actual]  # S_thin.shape: (top_k_actual,)
            
            # Compute T matrix
            S2_inv = 1.0 / (S_thin ** 2 + self.eps)
            T2 = S_thin * S2_inv * S_thin
            T2 = torch.where(T2 > self.eps, T2, torch.ones_like(T2) * self.eps)
            T = torch.diag(torch.sqrt(T2))
            
            # AT = A @ T
            AT = torch.matmul(A, T)  # AT.shape: (n_samples, top_k_actual)
            AT_list.append(AT)
        
        # Concatenate all AT matrices
        M_tilde = torch.cat(AT_list, dim=1)  # M_tilde.shape: (n_samples, sum of top_k_actual for all views)
        
        # Compute QR decomposition of M_tilde
        Q, R = torch.linalg.qr(M_tilde)
        
        # Compute SVD of R
        _, S, _ = torch.linalg.svd(R, full_matrices=False)
        
        if not self.use_all_singular_values:
            S = S[:top_k]
        
        # Sum of singular values (correlation)
        corr = torch.sum(S)
        
        # Return negative correlation as loss (to maximize correlation)
        return -corr

def validation(model, dataset, view, data_size, class_num, device):
    model.eval()
    x_list, y, idx = next(iter(DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)))
    x_list = [x_list[v].to(device) for v in range(view)]
    y = y.numpy()
    with torch.no_grad():
        Z_list = model(x_list)
        Z_list = [Z.detach().cpu().numpy() for Z in Z_list]
    # Fuse multiple views by averaging
    Z_fused = np.mean(Z_list, axis=0)  # shape: (num_samples, outdim_size)
    kmeans = KMeans(n_clusters=class_num, random_state=42, n_init=10)
    pred = kmeans.fit_predict(Z_fused)
    nmi, ari, acc, pur = evaluate(y, pred)
    print('Clustering results: ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
    return Z_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCCA')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--outdim_size', default=10, type=int)
    parser.add_argument('--seed', default=2, type=int)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## 1. Set seed for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Create dimension lists for each view
    dim_list_list = [[dims[v], 128, 64, 32, args.outdim_size] for v in range(view)]
    model = DeepGCCA(dim_list_list).to(device)
    criterion = DGCCALoss(outdim_size=args.outdim_size, use_all_singular_values=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    ## 3. Train the model.
    model.train()
    for epoch in range(args.num_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            z_list = model(x_list)
            loss = criterion(z_list)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Training Epoch {}'.format(epoch+1), 'Loss:{:.6f}'.format(np.mean(losses)))
    
    ## 4. Evaluate the model.
    validation(model, dataset, view, data_size, class_num, device)
