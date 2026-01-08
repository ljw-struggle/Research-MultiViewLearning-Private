import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, random, numpy as np
from torch.utils.data import DataLoader
from _utils import load_data, evaluate

class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=32, hidden_dim_list=[32, 32], activation='relu', use_bn=False, use_bias=True):
        super().__init__()
        dims = [input_dim] + hidden_dim_list + [output_dim]
        self.mlp = nn.Sequential()
        for i in range(len(dims) - 1):
            self.mlp.add_module('Linear_%d' % i, nn.Linear(dims[i], dims[i+1], bias=use_bias))
            self.mlp.add_module('BatchNorm_%d' % i, nn.BatchNorm1d(dims[i+1])) if use_bn else None
            if activation is not None and activation.lower() != 'none':
                self.mlp.add_module('Activation_%d' % i, {'relu': lambda: nn.ReLU(), 'sigmoid': lambda: nn.Sigmoid(), 'tanh': lambda: nn.Tanh(), 'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2)}[activation]())

    def forward(self, x):
        return self.mlp(x)

class Discriminator(nn.Module): # EAMC Discriminator
    def __init__(self, input_dim=32, output_dim=1, hidden_dim_list=[32, 32, 32], activation='leaky_relu', use_bn=False, use_bias=True):
        super().__init__()
        dims = [input_dim] + hidden_dim_list
        self.mlp = nn.Sequential()
        for i in range(len(dims) - 1):
            self.mlp.add_module('Linear_%d' % i, nn.Linear(dims[i], dims[i+1], bias=use_bias))
            self.mlp.add_module('BatchNorm_%d' % i, nn.BatchNorm1d(dims[i+1])) if use_bn else None
            if activation is not None and activation.lower() != 'none':
                self.mlp.add_module('Activation_%d' % i, {'relu': lambda: nn.ReLU(), 'sigmoid': lambda: nn.Sigmoid(), 'tanh': lambda: nn.Tanh(), 'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2)}[activation]())
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim_list[-1], output_dim, bias=True), nn.Sigmoid())

    def forward(self, view_hidden_0, view_hidden_i):
        real_output = self.output_layer(self.mlp(view_hidden_0)) # shape: (batch_size, 1)
        fake_output = self.output_layer(self.mlp(view_hidden_i)) # shape: (batch_size, 1)
        return [real_output.squeeze(), fake_output.squeeze()]

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, n_views, hidden_dim_list=[100, 50], activation='None', use_bn=False, use_bias=True, tau=10.0):
        super().__init__()
        self.tau = tau
        dims = [input_dim] + hidden_dim_list
        self.mlp = nn.Sequential()
        for i in range(len(dims) - 1):
            self.mlp.add_module('Linear_%d' % i, nn.Linear(dims[i], dims[i+1], bias=use_bias))
            self.mlp.add_module('BatchNorm_%d' % i, nn.BatchNorm1d(dims[i+1])) if use_bn else None
            if activation is not None and activation.lower() != 'none':
                self.mlp.add_module('Activation_%d' % i, {'relu': lambda: nn.ReLU(), 'sigmoid': lambda: nn.Sigmoid(), 'tanh': lambda: nn.Tanh(), 'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2)}[activation]())
        self.output_layer = nn.Linear(hidden_dim_list[-1], n_views, bias=True)

    def forward(self, xs):
        h = torch.cat(xs, dim=1)
        act = self.output_layer(self.mlp(h))
        e = nn.functional.softmax(torch.sigmoid(act) / self.tau, dim=1) # shape: (batch_size, n_views)
        weights = torch.mean(e, dim=0) # shape: (n_views,)
        return weights # shape: (n_views,)
    
class DDC(nn.Module): # Deep Discriminative Clustering Module
    def __init__(self, input_dim=32, hidden_dim=100, n_clusters=3, use_bn=True):
        super().__init__()
        hidden_layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]; 
        hidden_layers.append(nn.BatchNorm1d(num_features=hidden_dim)) if use_bn else None
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(hidden_dim, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden

class EAMC(nn.Module): # End-to-End Adversarial Multi-View Clustering
    def __init__(self, n_views=2, n_clusters=3, dims=[2, 2], 
                 backbone_hidden_dim_list=[32, 32], backbone_output_dim=32, backbone_activation='relu', backbone_use_bias=True, backbone_use_bn=False,
                 discriminator_hidden_dim_list=[32, 32, 32], discriminator_output_dim=1, discriminator_activation='leaky_relu', discriminator_use_bias=True, discriminator_use_bn=False,
                 attention_hidden_dim_list=[100, 50], attention_activation='None', attention_use_bias=True, attention_use_bn=False, attention_tau=10.0, 
                 ddc_hidden_dim=100, ddc_use_bn=False):
        super().__init__()
        self.n_views = n_views
        self.view_specific_encoder_list = nn.ModuleList([MLP(input_dim=dims[i], output_dim=backbone_output_dim, hidden_dim_list=backbone_hidden_dim_list, activation=backbone_activation, use_bias=backbone_use_bias, use_bn=backbone_use_bn) for i in range(n_views)])
        self.discriminator_pair_list = nn.ModuleList([Discriminator(input_dim=backbone_output_dim, output_dim=discriminator_output_dim, hidden_dim_list=discriminator_hidden_dim_list, activation=discriminator_activation, use_bias=discriminator_use_bias, use_bn=discriminator_use_bn) for _ in range(n_views-1)])
        self.attention_fusion_layer = AttentionLayer(input_dim=backbone_output_dim * n_views, n_views=n_views, tau=attention_tau, hidden_dim_list=attention_hidden_dim_list, activation=attention_activation, use_bias=attention_use_bias, use_bn=attention_use_bn)
        self.deep_discriminative_clustering_layer = DDC(input_dim=backbone_output_dim, hidden_dim=ddc_hidden_dim, n_clusters=n_clusters, use_bn=ddc_use_bn)
        self.apply(self.he_init_weights)
    
    @staticmethod
    def he_init_weights(module): # Initialize network weights using the He (Kaiming) initialization strategy.
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
    
    def forward(self, x_list):
        view_specific_outputs = [view_specific_encoder(x) for view_specific_encoder, x in zip(self.view_specific_encoder_list, x_list)] # shape: n_views * (batch_size, 32)
        discriminator_outputs = [self.discriminator_pair_list[i](view_specific_outputs[0], view_specific_outputs[i+1]) for i in range(self.n_views-1)] # shape: n_views-1 * 2 * (batch_size, 1)
        view_specific_weights = self.attention_fusion_layer(view_specific_outputs) # shape: (n_views,)
        fused_output = torch.sum(view_specific_weights[None, None, :] * torch.stack(view_specific_outputs, dim=-1), dim=-1) # shape: (batch_size, 32)
        clustering_output, hidden = self.deep_discriminative_clustering_layer(fused_output) # shape: (batch_size, 3), (batch_size, 100)
        return view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden
    
class Loss(nn.Module):
    def __init__(self, batch_size, n_clusters, gamma=10.0, device='cuda'):
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.n_clusters = n_clusters

    def forward(self, view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden):
        attention_loss = self.AttentionLoss(view_specific_outputs, fused_output, view_specific_weights) # shape: (1,)
        generator_loss = self.GeneratorLoss(discriminator_outputs) # shape: (1,)
        discriminator_loss = self.DiscriminatorLoss(discriminator_outputs) # shape: (1,)
        ddc1_loss = self.DDC1(clustering_output, hidden) # shape: (1,)
        ddc2_loss = self.DDC2(clustering_output) # shape: (1,)
        ddc2_flipped_loss = self.DDC2Flipped(clustering_output) # shape: (1,)
        ddc3_loss = self.DDC3(clustering_output, hidden) # shape: (1,)
        return {'ddc1': ddc1_loss, 'ddc2': ddc2_loss, 'ddc2_flipped': ddc2_flipped_loss, 'ddc3': ddc3_loss, 'att': attention_loss, 'gen': generator_loss, 'disc': discriminator_loss}

    def forward_discriminator_loss(self,view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden):
        loss_dict = self.forward(view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden)
        total_loss_discriminator = loss_dict['disc'] # shape: (1,)
        return total_loss_discriminator
    
    def forward_clustering_loss(self,view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden):
        loss_dict = self.forward(view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden)
        total_loss_clustering = loss_dict['ddc1'] + loss_dict['ddc2_flipped'] + loss_dict['ddc3'] + loss_dict['att'] + loss_dict['gen'] # shape: (1,)
        return total_loss_clustering
    
    def DDC1(self, clustering_output, hidden): # Cauchy-Schwarz Divergence between output and hidden kernel
        hidden_kernel = self.vector_kernel(hidden, relative_sigma=0.15)
        return self.cauchy_schwarz_divergence(clustering_output, hidden_kernel, self.n_clusters)
    
    def DDC2(self, clustering_output): # Orthogonality constraint (upper triangular of output @ output.T)
        n = self.batch_size # number of samples, shape: (1,)
        temp = torch.sum(torch.triu(clustering_output @ torch.t(clustering_output), diagonal=1)) # sum of the strictly upper triangular part
        cauchy_schwarz_divergence = 2 / (n * (n - 1)) * temp # Cauchy-Schwarz divergence, shape: (1,)
        return cauchy_schwarz_divergence # Cauchy-Schwarz divergence, shape: (1,)
    
    def DDC2Flipped(self, clustering_output): # Orthogonality constraint (upper triangular of output.T @ output)
        n = self.n_clusters # number of clusters, shape: (1,)
        temp = torch.sum(torch.triu(torch.t(clustering_output) @ clustering_output, diagonal=1)) # sum of the strictly upper triangular part
        cauchy_schwarz_divergence = 2 / (n * (n - 1)) * temp # Cauchy-Schwarz divergence, shape: (1,)
        return cauchy_schwarz_divergence # Cauchy-Schwarz divergence, shape: (1,)
    
    def DDC3(self, clustering_output, hidden): # Cluster center constraint
        hidden_kernel = self.vector_kernel(hidden, relative_sigma=0.15)
        output = torch.exp(-self.cdist_squared(clustering_output, torch.eye(self.n_clusters, device=clustering_output.device)))
        return self.cauchy_schwarz_divergence(output, hidden_kernel, self.n_clusters)
    
    def AttentionLoss(self, view_specific_outputs, fused_output, view_specific_weights):
        backbone_kernels = [self.vector_kernel(h, relative_sigma=0.15) for h in view_specific_outputs] # shape: n_views * (batch_size, batch_size)
        fusion_kernel = self.vector_kernel(fused_output, relative_sigma=0.15) # shape: (batch_size, batch_size)
        backbone_kernel_concat = torch.sum(view_specific_weights[None, None, :] * torch.stack(backbone_kernels, dim=-1), dim=-1) # shape: (batch_size, batch_size)
        difference_matrix = (fusion_kernel - backbone_kernel_concat) # shape: (batch_size, batch_size)
        attention_loss = torch.trace(difference_matrix @ torch.t(difference_matrix)) # shape: (1,), trace is the sum of the diagonal elements
        return attention_loss
    
    def GeneratorLoss(self, discriminator_outputs): # discriminator_outputs: shape: n_views-1 * 2 * (batch_size, 1)
        tot = torch.tensor(0., device=self.device)
        target = torch.ones(self.batch_size, device=self.device)
        for d_0, d_v in discriminator_outputs: # d_v: shape: (batch_size, 1)
            # Generator wants discriminator to think view_i+1 is real
            tot += F.binary_cross_entropy(d_v.squeeze(), target)
        return self.gamma * tot
    
    def DiscriminatorLoss(self, discriminator_outputs): # discriminator_outputs: shape: n_views-1 * 2 * (batch_size, 1)
        tot = torch.tensor(0., device=self.device)
        real_target = torch.ones(self.batch_size, device=self.device)
        fake_target = torch.zeros(self.batch_size, device=self.device)
        for d_0, d_v in discriminator_outputs: # d_0: shape: (batch_size, 1), d_v: shape: (batch_size, 1)
            # Discriminator wants to correctly distinguish view0 (real) and view_i+1 (fake)
            tot += F.binary_cross_entropy(d_0.squeeze(), real_target) + F.binary_cross_entropy(d_v.squeeze(), fake_target)
        return tot
    
    def vector_kernel(self, x, relative_sigma=0.15, min_sigma=1e-9): # gaussian kernel matrix, shape: (batch_size, batch_size)
        dist2 = self.cdist_squared(x, x) # square of the distance matrix, shape: (batch_size, batch_size)
        dist2 = F.relu(dist2) # set values less than 0 to 0, because the distance matrix can sometimes contain negative values due to floating point errors.
        sigma2 = relative_sigma * torch.median(dist2).detach() # median of the distance matrix, detach to disable gradient, shape: (1,)
        # sigma2 = torch.maximum(sigma2, torch.tensor(min_sigma, device=sigma2.device, dtype=sigma2.dtype)) # max of the sigma2 and min_sigma, shape: (1,)
        # sigma2 = torch.clamp(sigma2, min=min_sigma)
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
        # torch.cdist(X, Y, p=2)**2 # it is not used because it has different results from the previous implementation. but it is more efficient.
        xyT = X @ torch.t(Y) # shape: (batch_size, batch_size)
        x2 = torch.sum(X**2, dim=1, keepdim=True) # shape: (batch_size, 1)
        y2 = torch.sum(Y**2, dim=1, keepdim=True) # shape: (batch_size, 1)
        d = x2 - 2 * xyT + torch.t(y2) # shape: (batch_size, batch_size)
        return d # shape: (batch_size, batch_size)
    
def validation(model, dataset, view, data_size, class_num, eval_multi_view=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden = model(x_list) 
        final_pred = torch.argmax(clustering_output, dim=1) # shape: (batch_size,)
    final_pred = final_pred.detach().cpu().numpy() # shape: (batch_size,)
    nmi, ari, acc, pur = evaluate(y, final_pred)
    return acc, nmi, pur, ari
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EAMC')
    parser.add_argument('--dataset', type=str, default='voc', choices=['BDGP', 'blobs_overlap_5', 'blobs_overlap', 'rgbd', 'voc'], help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval (epochs)')
    parser.add_argument('--t', type=int, default=1, help='Number of iterations to train generator/clustering')
    parser.add_argument('--t_disc', type=int, default=1, help='Number of iterations to train discriminator')
    parser.add_argument('--clip_norm', type=float, default=0.5, help='Gradient clipping norm (None to disable)')
    # Optimizer parameters
    parser.add_argument('--lr_backbones', type=float, default=1e-5, help='Learning rate for backbones')
    parser.add_argument('--lr_disc', type=float, default=1e-3, help='Learning rate for discriminator')
    parser.add_argument('--lr_att', type=float, default=1e-4, help='Learning rate for attention layer')
    parser.add_argument('--lr_clustering', type=float, default=1e-5, help='Learning rate for clustering module')
    parser.add_argument('--betas_backbones', type=tuple, default=(0.95, 0.999), help='Beta parameters for the encoders')
    parser.add_argument('--betas_disc', type=tuple, default=(0.5, 0.999), help='Beta parameters for the discriminator')
    parser.add_argument('--betas_att', type=tuple, default=(0.95, 0.999), help='Beta parameters for the attention net')
    parser.add_argument('--betas_clustering_module', type=tuple, default=(0.95, 0.999), help='Beta parameters for the clustering module')
    # Model: Backbone parameters
    parser.add_argument('--backbone_hidden_dim_list', type=list, default=[512, 512], help='Hidden dimension list for the backbones')
    parser.add_argument('--backbone_output_dim', type=int, default=256, help='Output dimension for the backbones')
    parser.add_argument('--backbone_activation', type=str, default='relu', help='Activation function for the backbones')
    parser.add_argument('--backbone_use_bias', type=bool, default=True, help='Use bias for the backbones')
    parser.add_argument('--backbone_use_bn', type=bool, default=False, help='Use batch normalization for the backbones')
    # Model: Discriminator parameters
    parser.add_argument('--discriminator_hidden_dim_list', type=list, default=[256, 256], help='Hidden dimension list for the discriminator')
    parser.add_argument('--discriminator_output_dim', type=int, default=1, help='Output dimension for the discriminator')
    parser.add_argument('--discriminator_activation', type=str, default='leaky_relu', help='Activation function for the discriminator')
    parser.add_argument('--discriminator_use_bias', type=bool, default=True, help='Use bias for the discriminator')
    parser.add_argument('--discriminator_use_bn', type=bool, default=False, help='Use batch normalization for the discriminator')
    # Model: Attention parameters
    parser.add_argument('--attention_tau', type=float, default=10.0, help='Temperature parameter for attention')
    parser.add_argument('--attention_hidden_dim_list', type=list, default=[100, 50], help='Hidden dimension list for the attention net')
    parser.add_argument('--attention_activation', type=str, default='None', help='Activation function for the attention net')
    parser.add_argument('--attention_use_bias', type=bool, default=True, help='Use bias for the attention net')
    parser.add_argument('--attention_use_bn', type=bool, default=False, help='Use batch normalization for the attention net')
    # Model: DDC parameters
    parser.add_argument('--ddc_n_hidden', type=int, default=100, help='Hidden dimension for DDC')
    parser.add_argument('--ddc_use_bn', type=bool, default=True, help='Use batch normalization for DDC')
    # Model: Loss parameters
    parser.add_argument('--loss_gamma', type=float, default=10.0, help='Gamma parameter for Loss')
    parser.add_argument('--loss_rel_sigma', type=float, default=0.15, help='Relative sigma parameter for Loss')
    parser.add_argument('--seed', type=int, default=10, help='Seed')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name, batch_size, num_epochs, eval_interval, t, t_disc, clip_norm = args.dataset, args.batch_size, args.num_epochs, args.eval_interval, args.t, args.t_disc, args.clip_norm
    lr_backbones, lr_disc, lr_att, lr_clustering, betas_backbones, betas_disc, betas_att, betas_clustering_module = args.lr_backbones, args.lr_disc, args.lr_att, args.lr_clustering, args.betas_backbones, args.betas_disc, args.betas_att, args.betas_clustering_module
    backbone_hidden_dim_list, backbone_output_dim, backbone_activation, backbone_use_bias, backbone_use_bn = args.backbone_hidden_dim_list, args.backbone_output_dim, args.backbone_activation, args.backbone_use_bias, args.backbone_use_bn
    discriminator_hidden_dim_list, discriminator_output_dim, discriminator_activation, discriminator_use_bias, discriminator_use_bn = args.discriminator_hidden_dim_list, args.discriminator_output_dim, args.discriminator_activation, args.discriminator_use_bias, args.discriminator_use_bn
    attention_tau, attention_hidden_dim_list, attention_activation, attention_use_bias, attention_use_bn = args.attention_tau, args.attention_hidden_dim_list, args.attention_activation, args.attention_use_bias, args.attention_use_bn
    ddc_n_hidden, ddc_use_bn, loss_gamma, loss_rel_sigma, seed = args.ddc_n_hidden, args.ddc_use_bn, args.loss_gamma, args.loss_rel_sigma, args.seed
    if dataset_name == 'blobs_overlap_5':
        backbone_hidden_dim_list = [32, 32]; backbone_output_dim = 32; discriminator_hidden_dim_list = [32, 32, 32]; lr_backbones = 2e-4; lr_disc = 1e-5;
    if dataset_name == 'blobs_overlap':
        backbone_hidden_dim_list = [32, 32]; backbone_output_dim = 32; discriminator_hidden_dim_list = [32, 32, 32]; lr_backbones = 2e-4; lr_disc = 1e-5;
    if dataset_name == 'rgbd':
        backbone_hidden_dim_list = [200, 200]; backbone_output_dim = 500; lr_backbones = 6e-5; lr_disc = 2e-5
    
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    dataset, dims, n_views, data_size, n_clusters = load_data(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    model = EAMC(n_views=n_views, n_clusters=n_clusters, dims=dims, 
                 backbone_hidden_dim_list=backbone_hidden_dim_list, backbone_output_dim=backbone_output_dim, backbone_activation=backbone_activation, backbone_use_bias=backbone_use_bias, backbone_use_bn=backbone_use_bn,
                 discriminator_hidden_dim_list=discriminator_hidden_dim_list, discriminator_output_dim=discriminator_output_dim, discriminator_activation=discriminator_activation, discriminator_use_bias=discriminator_use_bias, discriminator_use_bn=discriminator_use_bn,
                 attention_hidden_dim_list=attention_hidden_dim_list, attention_activation=attention_activation, attention_use_bias=attention_use_bias, attention_use_bn=attention_use_bn, attention_tau=attention_tau,
                 ddc_hidden_dim=ddc_n_hidden, ddc_use_bn=ddc_use_bn).to(device)
    loss_fn = Loss(batch_size=batch_size, n_clusters=n_clusters, gamma=loss_gamma, device=device)
    clustering_optimizer_spec = [
        {"params": model.view_specific_encoder_list.parameters(), "lr": lr_backbones, "betas": betas_backbones},
        {"params": model.deep_discriminative_clustering_layer.parameters(), "lr": lr_clustering, "betas": betas_clustering_module},
        {"params": model.attention_fusion_layer.parameters(), "lr": lr_att, "betas": betas_att}
    ]
    clustering_optimizer = torch.optim.Adam(clustering_optimizer_spec)
    discriminator_optimizer_spec = [
        {"params": model.discriminator_pair_list.parameters(), "lr": lr_disc, "betas": betas_disc}
    ]
    discriminator_optimizer = torch.optim.Adam(discriminator_optimizer_spec)
    
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (xs, y, idx) in enumerate(dataloader):
            train_mode = "gen" if (batch_idx % (t + t_disc) < t) else "disc"
            xs = [x.to(device) for x in xs]
            if train_mode == "disc":
                discriminator_optimizer.zero_grad()
                view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden = model(xs)
                discriminator_loss = loss_fn.forward_discriminator_loss(view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden)
                discriminator_loss.backward() # disc_loss
                discriminator_optimizer.step()
            else:
                clustering_optimizer.zero_grad()
                view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden = model(xs)
                clustering_loss = loss_fn.forward_clustering_loss(view_specific_outputs, discriminator_outputs, view_specific_weights, fused_output, clustering_output, hidden)
                clustering_loss.backward() # ddc1 + ddc2_flipped + ddc3 + att + gen
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm) # clip gradient
                clustering_optimizer.step()
        
        # Evaluate periodically
        if (epoch + 1) % args.eval_interval == 0:
            validation_results = validation(model, dataset, n_views, data_size, n_clusters)
            print(f"  Evaluation - NMI: {validation_results[0]:.4f}, ARI: {validation_results[1]:.4f}, ACC: {validation_results[2]:.4f}, PUR: {validation_results[3]:.4f}")
