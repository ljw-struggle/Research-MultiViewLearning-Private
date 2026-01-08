import os, sys, random, argparse, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class ReconstructionNet(nn.Module):
    def __init__(self, h_dim, feature_dim):
        super(ReconstructionNet, self).__init__()
        self.linears = nn.Sequential(nn.Linear(h_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, feature_dim))

    def forward(self, h):
        return self.linears(h)

class UncertaintyNet(nn.Module):
    def __init__(self, h_dim, feature_dim):
        super(UncertaintyNet, self).__init__()
        self.linears = nn.Sequential(nn.Linear(h_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, feature_dim))

    def forward(self, h):
        return self.linears(h)

def validation(H, dataset, data_size, class_num, verbose=False):
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    _, y, _ = next(iter(test_loader))
    y = y.numpy()
    H_numpy = H.detach().cpu().numpy() if isinstance(H, torch.Tensor) else H
    H_normalized = MinMaxScaler().fit_transform(H_numpy)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    print("Clustering results on latent space H:") if verbose else None
    p = kmeans.fit_predict(H_normalized)
    nmi, ari, acc, pur = evaluate(y, p)
    print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

def benchmark_2021_AAAI_DUANET_HDUA(dataset_name="BDGP",
                                    latent_dim=20,
                                    batch_size=2000,
                                    epoch_num_pretrain=200,
                                    epoch_num_finetune=100,
                                    learning_rate_pretrain=5e-3,
                                    learning_rate_finetune=1e-3,
                                    seed=2,
                                    verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, feature_dim_list, view_num, data_size, class_num = load_data(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    H = torch.normal(mean=torch.zeros([data_size, latent_dim]), std=0.01).to(device).detach()
    H.requires_grad_(True)
    RNet = nn.ModuleList([ReconstructionNet(latent_dim, feature_dim_list[v]).to(device) for v in range(view_num)])
    UNet = nn.ModuleList([UncertaintyNet(latent_dim, feature_dim_list[v]).to(device) for v in range(view_num)])
    
    ## 3. Pretraining: optimize the latent space H to reconstruct each view.
    optimizer_pre = torch.optim.Adam(itertools.chain(RNet.parameters(), [H]), lr=learning_rate_pretrain)
    for epoch_pre in range(epoch_num_pretrain):
        epoch_loss = 0.0
        for batch_idx, (x_list, y, idx) in enumerate(dataloader):
            optimizer_pre.zero_grad()
            x_list = [x_list[v].to(device) for v in range(view_num)]
            h = H[idx]
            x_re = [RNet[v](h) for v in range(view_num)]
            loss_pre = sum([F.mse_loss(x_re[v], x_list[v], reduction='mean') for v in range(view_num)])
            loss_pre.backward()
            optimizer_pre.step()
            epoch_loss += loss_pre.item()
        print('Pretraining Epoch: {}'.format(epoch_pre + 1), 'Loss: {:.6f}'.format(epoch_loss / len(dataloader))) if verbose else None
    
    ## 4. Fine-tuning: optimize the latent space H to reconstruct each view and predict uncertainty.
    optimizer = torch.optim.Adam(itertools.chain(RNet.parameters(), UNet.parameters(), [H]), lr=learning_rate_finetune)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    for epoch in range(epoch_num_finetune):
        epoch_loss = 0.0
        for batch_idx, (x_list, y, idx) in enumerate(dataloader):
            optimizer.zero_grad()
            x_list = [x_list[v].to(device) for v in range(view_num)]
            h = H[idx]
            x_re = [RNet[v](h) for v in range(view_num)]
            log_sigma_2 = [UNet[v](h) for v in range(view_num)]
            loss = sum([0.5 * torch.mean((x_re[v] - x_list[v])**2 * torch.exp(-log_sigma_2[v]) + log_sigma_2[v]) for v in range(view_num)])
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        print('Fine-tuning Epoch: {}'.format(epoch + 1), 'Loss: {:.6f}'.format(epoch_loss / len(dataloader))) if verbose else None
    
    ## 5. Evaluate the model.
    nmi, ari, acc, pur = validation(H, dataset, data_size, class_num, verbose=verbose)
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DUANET_HDUA')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--latent_dim', default=20, type=int)
    parser.add_argument('--batch_size', default=2000, type=int)
    parser.add_argument('--epoch_num_pretrain', default=200, type=int)
    parser.add_argument('--epoch_num_finetune', default=100, type=int)
    parser.add_argument('--learning_rate_pretrain', default=5e-3, type=float)
    parser.add_argument('--learning_rate_finetune', default=1e-3, type=float)
    parser.add_argument('--seed', default=2, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_2021_AAAI_DUANET_HDUA(
        dataset_name=args.dataset,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epoch_num_pretrain=args.epoch_num_pretrain,
        epoch_num_finetune=args.epoch_num_finetune,
        learning_rate_pretrain=args.learning_rate_pretrain,
        learning_rate_finetune=args.learning_rate_finetune,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
