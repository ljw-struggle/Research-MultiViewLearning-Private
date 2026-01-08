import os, sys, argparse, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=32, hidden_dim_list=[32, 32], activation='relu', use_bn=False, use_bias=True):
        super().__init__()
        dims = [input_dim] + hidden_dim_list + [output_dim]
        self.mlp = nn.Sequential()
        for i in range(len(dims) - 1):
            self.mlp.add_module('Linear_%d' % i, nn.Linear(dims[i], dims[i+1], bias=use_bias))
            if use_bn:
                self.mlp.add_module('BatchNorm_%d' % i, nn.BatchNorm1d(dims[i+1]))
            if activation is not None and activation.lower() != 'none':
                self.mlp.add_module('Activation_%d' % i, {'relu': lambda: nn.ReLU(), 'sigmoid': lambda: nn.Sigmoid(), 'tanh': lambda: nn.Tanh(), 'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2)}[activation]())

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.mlp(x)

class DDC(nn.Module): # Deep Discriminative Clustering Module
    def __init__(self, input_dim=32, hidden_dim=100, n_clusters=3, use_bn=True):
        super().__init__()
        self.n_clusters = n_clusters
        hidden_layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=hidden_dim))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(hidden_dim, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden

class MeanFusion(nn.Module): # Mean Fusion
    def forward(self, inputs):
        return torch.mean(torch.stack(inputs, -1), dim=-1)

class WeightedMeanFusion(nn.Module): # Weighted Mean Fusion
    def __init__(self, n_views):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_views) / n_views)
    
    def forward(self, inputs):
        weights = F.softmax(self.weights, dim=0)
        return torch.sum(weights[None, None, :] * torch.stack(inputs, dim=-1), dim=-1)

class SiMVC(nn.Module): # Simple Multi-View Clustering
    def __init__(self, n_views=2, n_clusters=3, dims=[2, 2], 
                 backbone_hidden_dim_list=[32, 32], backbone_output_dim=32, backbone_activation='relu', backbone_use_bias=True, backbone_use_bn=False,
                 fusion_type='weighted_mean',
                 ddc_hidden_dim=100, ddc_use_bn=True):
        super().__init__()
        self.n_views = n_views
        self.view_specific_encoder_list = nn.ModuleList([MLP(input_dim=dims[i], output_dim=backbone_output_dim, hidden_dim_list=backbone_hidden_dim_list, activation=backbone_activation, use_bias=backbone_use_bias, use_bn=backbone_use_bn) for i in range(n_views)])
        if fusion_type == 'mean':
            self.fusion = MeanFusion()
        else:
            self.fusion = WeightedMeanFusion(n_views)
        self.deep_discriminative_clustering_layer = DDC(input_dim=backbone_output_dim, hidden_dim=ddc_hidden_dim, n_clusters=n_clusters, use_bn=ddc_use_bn)
        self.apply(self.he_init_weights)
    
    @staticmethod
    def he_init_weights(module): # Initialize network weights using the He (Kaiming) initialization strategy.
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
    
    def forward(self, x_list):
        view_specific_outputs = [view_specific_encoder(x) for view_specific_encoder, x in zip(self.view_specific_encoder_list, x_list)] # shape: n_views * (batch_size, output_dim)
        fused_output = self.fusion(view_specific_outputs) # shape: (batch_size, output_dim)
        clustering_output, hidden = self.deep_discriminative_clustering_layer(fused_output) # shape: (batch_size, n_clusters), (batch_size, hidden_dim)
        return view_specific_outputs, fused_output, clustering_output, hidden

class Loss(nn.Module):
    def __init__(self, batch_size, n_clusters, rel_sigma=0.15, device='cuda'):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.rel_sigma = rel_sigma

    def forward(self, view_specific_outputs, fused_output, clustering_output, hidden):
        ddc1_loss = self.DDC1(clustering_output, hidden) # shape: (1,)
        ddc2_loss = self.DDC2(clustering_output) # shape: (1,)
        ddc3_loss = self.DDC3(clustering_output, hidden) # shape: (1,)
        return {'ddc1': ddc1_loss, 'ddc2': ddc2_loss, 'ddc3': ddc3_loss, 'tot': ddc1_loss + ddc2_loss + ddc3_loss}
    
    def DDC1(self, clustering_output, hidden): # Cauchy-Schwarz Divergence between output and hidden kernel
        hidden_kernel = self.vector_kernel(hidden, relative_sigma=self.rel_sigma)
        return self.cauchy_schwarz_divergence(clustering_output, hidden_kernel, self.n_clusters)
    
    def DDC2(self, clustering_output): # Orthogonality constraint (upper triangular of output @ output.T)
        n = self.batch_size # number of samples, shape: (1,)
        temp = torch.sum(torch.triu(clustering_output @ torch.t(clustering_output), diagonal=1)) # sum of the strictly upper triangular part
        cauchy_schwarz_divergence = 2 / (n * (n - 1)) * temp # Cauchy-Schwarz divergence, shape: (1,)
        return cauchy_schwarz_divergence # Cauchy-Schwarz divergence, shape: (1,)
    
    def DDC3(self, clustering_output, hidden): # Cluster center constraint
        hidden_kernel = self.vector_kernel(hidden, relative_sigma=self.rel_sigma)
        output = torch.exp(-self.cdist_squared(clustering_output, torch.eye(self.n_clusters, device=clustering_output.device)))
        return self.cauchy_schwarz_divergence(output, hidden_kernel, self.n_clusters)
    
    def vector_kernel(self, x, relative_sigma=0.15, min_sigma=1e-9): # gaussian kernel matrix, shape: (batch_size, batch_size)
        dist2 = self.cdist_squared(x, x) # square of the distance matrix, shape: (batch_size, batch_size)
        dist2 = F.relu(dist2) # set values less than 0 to 0, because the distance matrix can sometimes contain negative values due to floating point errors.
        sigma2 = relative_sigma * torch.median(dist2).detach() # median of the distance matrix, detach to disable gradient, shape: (1,)
        sigma2 = torch.where(sigma2 < min_sigma, torch.tensor(min_sigma, device=sigma2.device, dtype=sigma2.dtype), sigma2)
        k = torch.exp(- dist2 / (2 * sigma2)) # gaussian kernel matrix, shape: (batch_size, batch_size)
        return k # gaussian kernel matrix, shape: (batch_size, batch_size)

    def cauchy_schwarz_divergence(self, A, K, n_clusters):
        nom = torch.t(A) @ K @ A
        nom = torch.where(nom < 1e-9, nom.new_tensor(1e-9), nom)
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)
        dnom_squared = torch.where(dnom_squared < 1e-9**2, dnom_squared.new_tensor(1e-9**2), dnom_squared)
        temp = torch.sum(torch.triu(nom / torch.sqrt(dnom_squared), diagonal=1))
        cauchy_schwarz_divergence = 2 / (n_clusters * (n_clusters - 1)) * temp
        return cauchy_schwarz_divergence
        
    @staticmethod
    def cdist_squared(X, Y): # L2 distance squared, shape: (batch_size, batch_size)
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
        view_specific_outputs, fused_output, clustering_output, hidden = model(x_list)
        final_pred = torch.argmax(clustering_output, dim=1) # shape: (batch_size,)
    final_pred = final_pred.detach().cpu().numpy() # shape: (batch_size,)
    nmi, ari, acc, pur = evaluate(y, final_pred)
    print("Clustering on latent output (k-means):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

def benchmark_2021_CVPR_MVC_SiMVC(dataset_name="rgbd",
                                  batch_size=100,
                                  num_epochs=100,
                                  eval_interval=4,
                                  clip_norm=5.0,
                                  learning_rate=1e-3,
                                  betas=(0.95, 0.999),
                                  backbone_hidden_dim_list=[512, 512],
                                  backbone_output_dim=256,
                                  backbone_activation="relu",
                                  backbone_use_bias=True,
                                  backbone_use_bn=False,
                                  fusion_type="weighted_mean",
                                  ddc_n_hidden=100,
                                  ddc_use_bn=True,
                                  loss_rel_sigma=0.15,
                                  seed=10,
                                  verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, n_views, data_size, n_clusters = load_data(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    model = SiMVC(n_views=n_views, n_clusters=n_clusters, dims=dims, 
                 backbone_hidden_dim_list=backbone_hidden_dim_list, backbone_output_dim=backbone_output_dim, backbone_activation=backbone_activation, backbone_use_bias=backbone_use_bias, backbone_use_bn=backbone_use_bn,
                 fusion_type=fusion_type,
                 ddc_hidden_dim=ddc_n_hidden, ddc_use_bn=ddc_use_bn).to(device)
    loss_fn = Loss(batch_size=batch_size, n_clusters=n_clusters, rel_sigma=loss_rel_sigma, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)
    
    ## 3. Train the model.
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (xs, y, idx) in enumerate(dataloader):
            xs = [x.to(device) for x in xs]
            optimizer.zero_grad()
            view_specific_outputs, fused_output, clustering_output, hidden = model(xs)
            loss_dict = loss_fn(view_specific_outputs, fused_output, clustering_output, hidden)
            loss_dict['tot'].backward()
            if clip_norm is not None and clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            epoch_loss += loss_dict['tot'].item()
        print("Epoch: {} Total Loss: {:.4f}".format(epoch + 1, epoch_loss / len(dataloader))) if verbose else None
    
    ## 4. Evaluate the model.
    nmi, ari, acc, pur = validation(model, dataset, n_views, data_size, n_clusters, verbose=verbose)
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiMVC")
    parser.add_argument("--dataset", type=str, default="rgbd", choices=["BDGP", "blobs_overlap_5", "blobs_overlap", "rgbd", "voc"], help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--eval_interval", type=int, default=4, help="Evaluation interval (epochs)")
    parser.add_argument("--clip_norm", type=float, default=5.0, help="Gradient clipping norm (None to disable)")
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--betas", type=tuple, default=(0.95, 0.999), help="Beta parameters for Adam optimizer")
    # Model: Backbone parameters
    parser.add_argument("--backbone_hidden_dim_list", type=list, default=[512, 512], help="Hidden dimension list for the backbones")
    parser.add_argument("--backbone_output_dim", type=int, default=256, help="Output dimension for the backbones")
    parser.add_argument("--backbone_activation", type=str, default="relu", help="Activation function for the backbones")
    parser.add_argument("--backbone_use_bias", type=bool, default=True, help="Use bias for the backbones")
    parser.add_argument("--backbone_use_bn", type=bool, default=False, help="Use batch normalization for the backbones")
    # Model: Fusion parameters
    parser.add_argument("--fusion_type", type=str, default="weighted_mean", choices=["mean", "weighted_mean"], help="Fusion type")
    # Model: DDC parameters
    parser.add_argument("--ddc_n_hidden", type=int, default=100, help="Hidden dimension for DDC")
    parser.add_argument("--ddc_use_bn", type=bool, default=True, help="Use batch normalization for DDC")
    # Model: Loss parameters
    parser.add_argument("--loss_rel_sigma", type=float, default=0.15, help="Relative sigma parameter for Loss")
    parser.add_argument("--seed", type=int, default=10, help="Seed")
    args = parser.parse_args()
    
    ## Adjust parameters based on dataset
    if args.dataset == 'blobs_overlap_5':
        args.backbone_hidden_dim_list = [32, 32]; args.backbone_output_dim = 32; args.lr = 1e-3
    if args.dataset == 'blobs_overlap':
        args.backbone_hidden_dim_list = [32, 32]; args.backbone_output_dim = 32; args.lr = 1e-3
    if args.dataset == 'rgbd':
        args.backbone_hidden_dim_list = [512, 512]; args.backbone_output_dim = 256; args.lr = 1e-3
    if args.dataset == 'voc':
        args.backbone_hidden_dim_list = [512, 512]; args.backbone_output_dim = 256; args.lr = 1e-3
    
    nmi, ari, acc, pur = benchmark_2021_CVPR_MVC_SiMVC(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        eval_interval=args.eval_interval,
        clip_norm=args.clip_norm,
        learning_rate=args.lr,
        betas=args.betas,
        backbone_hidden_dim_list=args.backbone_hidden_dim_list,
        backbone_output_dim=args.backbone_output_dim,
        backbone_activation=args.backbone_activation,
        backbone_use_bias=args.backbone_use_bias,
        backbone_use_bn=args.backbone_use_bn,
        fusion_type=args.fusion_type,
        ddc_n_hidden=args.ddc_n_hidden,
        ddc_use_bn=args.ddc_use_bn,
        loss_rel_sigma=args.loss_rel_sigma,
        seed=args.seed,
        verbose=False
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
