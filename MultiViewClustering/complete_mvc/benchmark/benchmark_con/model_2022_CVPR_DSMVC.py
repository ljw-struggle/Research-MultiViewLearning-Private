import os, sys, argparse, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, feature_dim), nn.ReLU())

    def forward(self, x):
        return self.encoder(x)

class WeightedMean(nn.Module):
    def __init__(self, view):
        super().__init__()
        self.weights = nn.Parameter(torch.full((view,), 1 / view), requires_grad=True) # shape: (view,)

    def forward(self, inputs): # inputs.shape: view * (batch_size, feature_dim)
        weights = F.softmax(self.weights, dim=0) # shape: (view,), sum to 1
        out = torch.sum(weights[None, None, :] * torch.stack(inputs, dim=-1), dim=-1) # shape: (batch_size, feature_dim)
        return out # shape: (batch_size, feature_dim)

class SiMVC(nn.Module):
    def __init__(self, view, input_size, feature_dim):
        super(SiMVC, self).__init__()
        self.view = view
        self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim) for v in range(view)])
        self.fusion_module = WeightedMean(view) # shape: (view,)

    def forward(self, xs):
        zs = [self.encoders[v](xs[v]) for v in range(self.view)]
        fused = self.fusion_module(zs) # shape: (batch_size, feature_dim)
        return zs, fused # zs.shape: view * (batch_size, feature_dim), fused.shape: (batch_size, feature_dim)

class DDC(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_dim, 100), nn.ReLU(), nn.BatchNorm1d(num_features=100))
        self.output = nn.Sequential(nn.Linear(100, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden
    
class DSMVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num):
        super(DSMVC, self).__init__()
        self.view = view
        self.model_view_pre = SiMVC(view - 1, input_size, feature_dim) # model for the first view-1 views
        self.model_view_all = SiMVC(view, input_size, feature_dim) # model for all views
        self.model_view_new = Encoder(input_size[view-1], feature_dim) # model for the last view
        self.fusion_module = WeightedMean(3) # model for the gate, shape: (3,)
        self.clustering_module = DDC(feature_dim, class_num) # DDC: Deep Discriminative Clustering
        self.apply(self.he_init_weights)
        
    @staticmethod
    def he_init_weights(module): # Initialize network weights using the He (Kaiming) initialization strategy.
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)

    def forward(self, xs):
        zs_pre, fused_pre = self.model_view_pre(xs) # zs_pre.shape: view-1 * (batch_size, feature_dim), fused_pre.shape: (batch_size, feature_dim)
        zs_all, fused_all = self.model_view_all(xs) # zs_all.shape: view * (batch_size, feature_dim), fused_all.shape: (batch_size, feature_dim)
        z_new = self.model_view_new(xs[self.view-1]) # z_new.shape: (batch_size, feature_dim)
        fused = self.fusion_module([fused_pre, fused_all, z_new]) # fused.shape: (batch_size, feature_dim)
        output, hidden = self.clustering_module(fused) # output.shape: (batch_size, class_num), hidden.shape: (batch_size, 100)
        return zs_pre, zs_all, output, hidden
    
class Loss(nn.Module): # Adopted from https://github.com/DanielTrosten/mvc
    def __init__(self, class_num):
        super(Loss, self).__init__()
        self.class_num = class_num

    def forward_cluster(self, hidden, output):
        # hidden.shape: (batch_size, 100); output.shape: (batch_size, class_num)
        hidden_kernel = self.gaussian_kernel(hidden, relative_sigma=0.15) # kernel matrix from the hidden features
        l1 = self.DDC1(output, hidden_kernel, self.class_num)
        l2 = self.DDC2(output)
        l3 = self.DDC3(output, hidden_kernel, self.class_num)
        return l1 + l2 + l3 # sum of the three losses, shape: (1,)

    def DDC1(self, output, hidden_kernel, n_clusters):
        A = output; K = hidden_kernel
        # Cauchy-Schwarz divergence.
        nom = A.T @ K @ A # nom.shape: (n_clusters, n_clusters)
        nom = torch.where(nom < 1e-9, nom.new_tensor(1e-9), nom) # set values less than 1e-9 to 1e-9
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0) # dnom_squared.shape: (n_clusters, 1) @ (1, n_clusters) = (n_clusters, n_clusters)
        dnom_squared = torch.where(dnom_squared < 1e-9**2, dnom_squared.new_tensor(1e-9**2), dnom_squared) # set values less than 1e-9**2 to 1e-9**2
        temp = torch.sum(torch.triu(nom / torch.sqrt(dnom_squared), diagonal=1)) # sum of the strictly upper triangular part
        cauchy_schwarz_divergence = 2 / (n_clusters * (n_clusters - 1)) * temp # Cauchy-Schwarz divergence, shape: (1,)
        return cauchy_schwarz_divergence # Cauchy-Schwarz divergence, shape: (1,)

    def DDC2(self, output):
        n_samples = output.size(0) # number of samples, shape: (1,)
        temp = torch.sum(torch.triu(output @ torch.t(output), diagonal=1)) # sum of the strictly upper triangular part
        cauchy_schwarz_divergence = 2 / (n_samples * (n_samples - 1)) * temp # Cauchy-Schwarz divergence, shape: (1,)
        return cauchy_schwarz_divergence # Cauchy-Schwarz divergence, shape: (1,)

    def DDC3(self, output, hidden_kernel, n_clusters):
        output = torch.exp(-self.cdist_squared(output, torch.eye(n_clusters, device=output.device))) # kernel matrix from the output and eye, shape: (batch_size, class_num)
        A = output; K = hidden_kernel
        # Cauchy-Schwarz divergence.
        nom = A.T @ K @ A # nom.shape: (n_clusters, n_clusters)
        nom = torch.where(nom < 1e-9, nom.new_tensor(1e-9), nom) # set values less than 1e-9 to 1e-9
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0) # dnom_squared.shape: (n_clusters, 1) @ (1, n_clusters) = (n_clusters, n_clusters)
        dnom_squared = torch.where(dnom_squared < 1e-9 ** 2, dnom_squared.new_tensor(1e-9 ** 2), dnom_squared) # set values less than 1e-9**2 to 1e-9**2
        temp = torch.sum(torch.triu(nom / torch.sqrt(dnom_squared), diagonal=1)) # sum of the strictly upper triangular part
        cauchy_schwarz_divergence = 2 / (n_clusters * (n_clusters - 1)) * temp # Cauchy-Schwarz divergence, shape: (1,)
        return cauchy_schwarz_divergence # Cauchy-Schwarz divergence, shape: (1,)

    def gaussian_kernel(self, x, relative_sigma=0.15, min_sigma=1e-9):
        dist2 = self.cdist_squared(x, x) # square of the distance matrix, shape: (batch_size, batch_size)
        dist2 = F.relu(dist2) # set values less than 0 to 0, because the distance matrix can sometimes contain negative values due to floating point errors.
        sigma2 = relative_sigma * torch.median(dist2).detach() # median of the distance matrix, detach to disable gradient, shape: (1,)
        sigma2 = torch.maximum(sigma2, torch.tensor(min_sigma, device=sigma2.device, dtype=sigma2.dtype)) # max of the sigma2 and min_sigma, shape: (1,)
        k = torch.exp(- dist2 / (2 * sigma2)) # gaussian kernel matrix, shape: (batch_size, batch_size)
        return k # gaussian kernel matrix, shape: (batch_size, batch_size)
    
    @staticmethod
    def cdist_squared(X, Y): # shape: (batch_size, feature_dim) @ (batch_size, feature_dim).T = (batch_size, batch_size)
        # torch.cdist(X, Y, p=2)**2 # it is not used because it has different results from the previous implementation. but it is more efficient.
        xyT = X @ torch.t(Y) # shape: (batch_size, batch_size)
        x2 = torch.sum(X**2, dim=1, keepdim=True) # shape: (batch_size, 1)
        y2 = torch.sum(Y**2, dim=1, keepdim=True) # shape: (batch_size, 1)
        d = x2 - 2 * xyT + torch.t(y2) # shape: (batch_size, batch_size)
        return d # shape: (batch_size, batch_size)

def validation(model, dataset, view, data_size, class_num, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, _ = next(iter(test_loader))
    x_list = [x_list[v].to(device) for v in range(view)]
    y = y.numpy()
    with torch.no_grad():
        _, _, output, _ = model(x_list) # shape: (batch_size, class_num)
    final_pred = torch.argmax(output, dim=1) # shape: (batch_size,)
    nmi, ari, acc, pur = evaluate(y, final_pred.cpu().numpy())
    print("Clustering on latent output (k-means):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

def benchmark_2022_CVPR_DSMVC(dataset_name="caltech_5m",
                              batch_size=128,
                              feature_dim=256,
                              view_num=2,
                              num_epochs=120,
                              seed=1,
                              verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = DSMVC(view_num, dims, feature_dim, class_num).to(device)
    criterion = Loss(class_num)
    
    ## 3. Train the model.
    PARAM_GROUPS = {'theta': ['model_view_pre', 'model_view_all', 'model_view_new', 'clustering_module'], 'lambda': ['fusion_module']}
    for epoch in range(num_epochs):
        if (epoch//20) % 2 == 0: # Update theta parameters
            modules_theta = [getattr(model, group) for group in PARAM_GROUPS['theta']]
            optimizer_theta = torch.optim.Adam(chain(*[module.parameters() for module in modules_theta]), lr=1e-3)
            theta_loss_list = []
            for batch_idx, (xs, _, _) in enumerate(dataloader):
                xs = [xs[v].to(device) for v in range(view)]
                optimizer_theta.zero_grad()
                _, _, output, hidden = model(xs)
                loss = criterion.forward_cluster(hidden, output)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(chain(*[module.parameters() for module in modules_theta]), 5.0)
                optimizer_theta.step()
                theta_loss_list.append(loss.item())
            print("Epoch: {} Theta Loss: {:.4f}".format(epoch + 1, np.mean(theta_loss_list))) if verbose else None
        if (epoch//20) % 2 == 1: # Update lambda parameters
            modules_lambda = [getattr(model, group) for group in PARAM_GROUPS['lambda']]
            optimizer_lambda = torch.optim.Adam(chain(*[module.parameters() for module in modules_lambda]), lr=1e-3)
            lambda_loss_list = []
            for batch_idx, (xs, _, _) in enumerate(dataloader):
                xs = [xs[v].to(device) for v in range(view)]
                optimizer_lambda.zero_grad()
                _, _, output, hidden = model(xs)
                loss = criterion.forward_cluster(hidden, output)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(chain(*[module.parameters() for module in modules_lambda]), 5.0)
                optimizer_lambda.step()
                lambda_loss_list.append(loss.item())
            print("Epoch: {} Lambda Loss: {:.4f}".format(epoch + 1, np.mean(lambda_loss_list))) if verbose else None
    
    ## 4. Evaluate the model.
    nmi, ari, acc, pur = validation(model, dataset, view, data_size, class_num, verbose=verbose)
    return nmi, ari, acc, pur

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DSMVC")
    parser.add_argument("--dataset", default="caltech_5m", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--view_num", default=2, type=int)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    
    nmi, ari, acc, pur = benchmark_2022_CVPR_DSMVC(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        view_num=args.view_num,
        num_epochs=args.epochs,
        seed=args.seed,
        verbose=False
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
    