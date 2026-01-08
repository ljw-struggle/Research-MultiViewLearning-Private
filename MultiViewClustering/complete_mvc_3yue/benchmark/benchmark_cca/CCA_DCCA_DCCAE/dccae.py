import random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class DeepCCAE(nn.Module):
    def __init__(self, dim_list_1, dim_list_2):
        super().__init__()
        self.encoder_1 = MLP(dim_list_1)
        self.decoder_1 = MLP(dim_list_1[::-1])
        self.encoder_2 = MLP(dim_list_2)
        self.decoder_2 = MLP(dim_list_2[::-1])
        
    def forward(self, x_1, x_2):
        z_1 = self.encoder_1(x_1)
        z_2 = self.encoder_2(x_2)
        r_1 = self.decoder_1(z_1)
        r_2 = self.decoder_2(z_2)
        return z_1, z_2, r_1, r_2

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

def validation(model, dataset, view, data_size, class_num, device):
    model.eval()
    x_list, y, idx = next(iter(DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)))
    x_list = [x_list[v].to(device) for v in range(view)]
    y = y.numpy()
    with torch.no_grad():
        Z1, Z2, _, _ = model(x_list[0], x_list[1])
        Z1 = Z1.detach().cpu().numpy()
        Z2 = Z2.detach().cpu().numpy()
    Z_fused = (Z1 + Z2) / 2
    kmeans = KMeans(n_clusters=class_num, random_state=42, n_init=10)
    pred = kmeans.fit_predict(Z_fused)
    nmi, ari, acc, pur = evaluate(y, pred)
    print('Clustering results: ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
    return Z1, Z2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DCCAE')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--pretrain_epochs', default=20, type=int, help='Number of epochs for pretraining (reconstruction only)')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs for joint training (CCA + reconstruction)')
    parser.add_argument('--outdim_size', default=10, type=int)
    parser.add_argument('--recon_weight', default=0.001, type=float, help='Weight for reconstruction loss in joint training')
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
    model = DeepCCAE([dims[0], 128, 64, 32, args.outdim_size], [dims[1], 128, 64, 32, args.outdim_size]).to(device)
    criterion_cca = DCCALoss(outdim_size=args.outdim_size, use_all_singular_values=False)
    criterion_recon = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    ## 3. Pretrain the model (reconstruction only).
    print("=" * 50)
    print("Stage 1: Pretraining (Reconstruction Only)")
    print("=" * 50)
    model.train()
    for epoch in range(args.pretrain_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            z1, z2, r1, r2 = model(x_list[0], x_list[1])
            loss = criterion_recon(r1, x_list[0]) + criterion_recon(r2, x_list[1])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Pretraining Epoch {}'.format(epoch+1), 'Loss:{:.6f}'.format(np.mean(losses)))
    
    ## 4. Joint training (CCA + reconstruction).
    print("=" * 50)
    print("Stage 2: Joint Training (CCA + Reconstruction)")
    print("=" * 50)
    model.train()
    for epoch in range(args.num_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            z1, z2, r1, r2 = model(x_list[0], x_list[1])
            loss_cca = criterion_cca(z1, z2)
            loss_recon = criterion_recon(r1, x_list[0]) + criterion_recon(r2, x_list[1])
            loss = loss_cca + args.recon_weight * loss_recon
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Joint Training Epoch {}'.format(epoch+1), 'Loss:{:.6f}'.format(np.mean(losses)))
    
    ## 5. Evaluate the model.
    validation(model, dataset, view, data_size, class_num, device)
