import os, sys, random, argparse, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Aos parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

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

class DeepCCAE(nn.Module):
    def __init__(self, dim_list_multi_view):
        super().__init__()
        self.encoder_list = nn.ModuleList([MLP(dim_list) for dim_list in dim_list_multi_view])
        self.decoder_list = nn.ModuleList([MLP(dim_list[::-1]) for dim_list in dim_list_multi_view])
        
    def forward(self, x_list):
        encoded_list = []; reconstructed_list = []
        for x, encoder, decoder in zip(x_list, self.encoder_list, self.decoder_list):
            z = encoder(x); r = decoder(z)
            encoded_list.append(z); reconstructed_list.append(r)
        return encoded_list, reconstructed_list

class DCCALoss(nn.Module):
    def __init__(self, outdim_size, r1=1e-3, r2=1e-3, eps=1e-9, use_all_singular_values=False):
        super(DCCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.r1 = r1; self.r2 = r2; self.eps = eps; self.use_all_singular_values = use_all_singular_values

    def forward(self, H1, H2): # H1.shape: (n_samples, n_features); H2.shape: (n_samples, n_features)
        # Get the number of features and samples of two views
        n_samples = H1.shape[0]; n_features_1 = H1.shape[1]; n_features_2 = H2.shape[1]
        # Calculate the centered data
        H1bar = H1 - H1.mean(dim=0, keepdim=True) # H1bar.shape: (n_samples, n_features_1)
        H2bar = H2 - H2.mean(dim=0, keepdim=True) # H2bar.shape: (n_samples, n_features_2)
        SigmaHat12 = (1.0 / (n_samples - 1)) * torch.matmul(H1bar.t(), H2bar) # SigmaHat12.shape: (n_features_1, n_features_2)
        SigmaHat11 = (1.0 / (n_samples - 1)) * torch.matmul(H1bar.t(), H1bar) + self.r1 * torch.eye(n_features_1, device=H1.device) # SigmaHat11.shape: (n_features_1, n_features_1)
        SigmaHat22 = (1.0 / (n_samples - 1)) * torch.matmul(H2bar.t(), H2bar) + self.r2 * torch.eye(n_features_2, device=H1.device) # SigmaHat22.shape: (n_features_2, n_features_2)
        # Calculating the root inverse of covariance matrices by using eigen decomposition
        D1, V1 = torch.linalg.eigh(SigmaHat11) # D1.shape: (n_features,); V1.shape: (n_features, n_features)
        D2, V2 = torch.linalg.eigh(SigmaHat22) # D2.shape: (n_features,); V2.shape: (n_features, n_features)
        # Added to increase stability (filter out the small eigenvalues and eigenvectors; for more stability)
        posInd1 = torch.gt(D1, self.eps).nonzero(as_tuple=False)[:, 0] # posInd1.shape: (len(posInd1),)
        D1 = D1[posInd1] # D1.shape: (len(posInd1),)
        V1 = V1[:, posInd1] # V1.shape: (n_features, len(posInd1))
        posInd2 = torch.gt(D2, self.eps).nonzero(as_tuple=False)[:, 0] # posInd2.shape: (len(posInd2),)
        D2 = D2[posInd2] # D2.shape: (len(posInd2),)
        V2 = V2[:, posInd2] # V2.shape: (n_features, len(posInd2))
        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t()) # SigmaHat11RootInv.shape: (n_features, n_features)
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t()) # SigmaHat22RootInv.shape: (n_features, n_features)
        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv) # Tval.shape: (n_features, n_features)
        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp)) # trace: sum of the diagonal elements
        else:
            # just the top outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval) # trace_TT.shape: (n_features, n_features)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*self.r1).to(H1.device)) # regularization for more stability
            U, V = torch.linalg.eigh(trace_TT) # U.shape: (n_features,); V.shape: (n_features, n_features); sorted in ascending order
            U = torch.where(U>self.eps, U, (torch.ones(U.shape)*self.eps).to(H1.device)) # U.shape: (n_features,); set the small eigenvalues to eps
            U = U.topk(self.outdim_size)[0] # U.shape: (outdim_size,); get the top outdim_size eigenvalues
            corr = torch.sum(torch.sqrt(U)) # corr.shape: (1,); sum of the top outdim_size eigenvalues
        return - corr # return the negative correlation
    
class DGCCALoss(nn.Module):
    def __init__(self, outdim_size, r=1e-3, eps=1e-9, use_all_singular_values=False):
        super(DGCCALoss, self).__init__()
        self.outdim_size = outdim_size
        self.r = r
        self.eps = eps
        self.use_all_singular_values = use_all_singular_values

    def forward(self, H_list): # H_list: list of tensors, each H[i].shape: (n_samples, n_features)
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
        Z_list, _ = model(x_list)  # Get encoded representations
        Z_list = [Z.detach().cpu().numpy() for Z in Z_list] # shape: (num_samples, outdim_size) * view
    # Fuse multiple views by averaging
    Z_fused = np.mean(Z_list, axis=0)  # shape: (num_samples, outdim_size)
    kmeans = KMeans(n_clusters=class_num, random_state=42, n_init=10)
    pred = kmeans.fit_predict(Z_fused)
    nmi, ari, acc, pur = evaluate(y, pred)
    return nmi, ari, acc, pur

def benchmark_2015_ICML_DCCAE(dataset_name="BDGP", 
                              outdim_size=10, 
                              regularization=1e-3, 
                              recon_weight=0.001, 
                              batch_size=256, 
                              learning_rate=1e-3, 
                              weight_decay=1e-4, 
                              pretrain_epochs=20, 
                              num_epochs=20, 
                              seed=42, 
                              verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    assert view > 1, "DCCAE(DGCCAE) only supports two views or more, but got {view} views"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dim_list_multi_view = [[dims[v], 128, 64, 32, outdim_size] for v in range(view)]
    model = DeepCCAE(dim_list_multi_view).to(device)
    if view == 2:
        criterion = DCCALoss(outdim_size=outdim_size, r1=regularization, r2=regularization, use_all_singular_values=False)
    else:
        criterion = DGCCALoss(outdim_size=outdim_size, r=regularization, use_all_singular_values=False)
    criterion_recon = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    ## 3. Pretrain the model (reconstruction only).
    model.train()
    for epoch in range(pretrain_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            _, reconstructed_list = model(x_list)
            loss = sum([criterion_recon(reconstructed_list[v], x_list[v]) for v in range(view)])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Pretraining Epoch {}'.format(epoch+1), 'Loss:{:.6f}'.format(np.mean(losses)))
    
    ## 4. Joint training (GCCA + reconstruction).
    model.train()
    for epoch in range(num_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            z_list, reconstructed_list = model(x_list)
            loss_gcca = criterion(z_list[0], z_list[1]) if view == 2 else criterion(z_list)
            loss_recon = sum([criterion_recon(reconstructed_list[v], x_list[v]) for v in range(view)])
            loss = loss_gcca + recon_weight * loss_recon
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Joint Training Epoch {}'.format(epoch+1), 'Loss:{:.6f}'.format(np.mean(losses))) if verbose else None
    
    ## 5. Evaluate the model.
    nmi, ari, acc, pur = validation(model, dataset, view, data_size, class_num, device)
    print('Clustering results: ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCCAE')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--outdim_size', default=10, type=int)
    parser.add_argument('--regularization', default=1e-3, type=float)
    parser.add_argument('--recon_weight', default=0.001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--pretrain_epochs', default=20, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_2015_ICML_DCCAE(
        dataset_name=args.dataset,
        outdim_size=args.outdim_size,
        regularization=args.regularization,
        recon_weight=args.recon_weight,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrain_epochs=args.pretrain_epochs,
        num_epochs=args.num_epochs,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
    
