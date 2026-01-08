import os, random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn import preprocessing
from _utils import load_data, evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, output_dim))

    def forward(self, x):
        return self.decoder(x)

class VAE(nn.Module):
    def __init__(self, dims, z_variables=10, class_num=10, temperature=0.67, view_num=2, hidden_dim=256, device=None):
        super(VAE, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.view_num = view_num
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        # Latent specification: continuous latent (view-peculiar variable), discrete latent (view-common variable)
        self.latent_spec = {'cont': z_variables, 'disc': class_num}
        self.latent_cont_dim = self.latent_spec['cont']
        self.latent_disc_dim = self.latent_spec['disc']
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim
        self.encoders = nn.ModuleList([Encoder(input_dim=dims[i], feature_dim=hidden_dim) for i in range(self.view_num)])
        self.decoders = nn.ModuleList([Decoder(feature_dim=self.latent_dim, output_dim=dims[i]) for i in range(self.view_num)])
        # Continuous latent (view-peculiar variable)
        self.means = nn.ModuleList([nn.Linear(hidden_dim, self.latent_cont_dim) for i in range(self.view_num)])
        self.log_vars = nn.ModuleList([nn.Linear(hidden_dim, self.latent_cont_dim) for i in range(self.view_num)])
        # Discrete latent (view-common variable)
        self.fc_alpha = nn.Linear(hidden_dim * self.view_num, self.latent_disc_dim)

    def encode(self, X):
        hidden_view_list = [self.encoders[i](X[i]) for i in range(self.view_num)] # shape: view_num * (batch_size, hidden_dim)
        hidden_fusion = torch.cat(hidden_view_list, dim=1) # shape: (batch_size, hidden_dim * view_num)
        # 1. Continuous latent (view-peculiar variable)
        means_continuous_list = [self.means[i](hidden_view_list[i]) for i in range(self.view_num)] # shape: view_num * (batch_size, latent_dim)
        log_vars_continuous_list = [self.log_vars[i](hidden_view_list[i]) for i in range(self.view_num)] # shape: view_num * (batch_size, latent_dim)
        # 2. Discrete latent (view-common variable)
        alpha_discrete = F.softmax(self.fc_alpha(hidden_fusion), dim=1) # shape: (batch_size, class_num)
        # Return the latent distribution
        latent_distribution = {f'cont{i+1}': [means_continuous_list[i], log_vars_continuous_list[i]] for i in range(self.view_num)}
        latent_distribution['disc'] = alpha_discrete  # shape: (batch_size, class_num)
        return latent_distribution

    def reparameterize(self, latent_distribution): # Samples from latent distribution using reparameterization trick
        """
        Returns a dictionary of sampled latent variables:
        - 'cont1', 'cont2', ..., 'contV': continuous latents for each view
        - 'disc': discrete latent (shared across all views)
        """
        latent_samples = {}
        # Sample continuous latents for each view
        for i in range(self.view_num):
            cont_name = 'cont' + str(i+1)
            mean, logvar = latent_distribution[cont_name]
            latent_samples[cont_name] = self.sample_normal(mean, logvar)  # shape: (batch_size, latent_cont_dim)
        # Sample discrete latent (shared across all views)
        latent_samples['disc'] = self.sample_gumbel_softmax(latent_distribution['disc'])  # shape: (batch_size, latent_disc_dim)
        return latent_samples

    def sample_normal(self, mean, logvar): # Samples from a normal distribution using reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size(), device=self.device).normal_()
            return mean + std * eps
        else:
            return mean

    def sample_gumbel_softmax(self, alpha): # Samples from a gumbel-softmax distribution
        if self.training:
            unif = torch.rand(alpha.size(), device=self.device)
            gumbel = -torch.log(-torch.log(unif + 1e-12) + 1e-12)
            log_alpha = torch.log(alpha + 1e-12)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1) # shape: (batch_size, class_num)
        else:
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size(), device=self.device)
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1), 1)
            return one_hot_samples # shape: (batch_size, class_num)

    def decode(self, latent_samples):
        recon_list = []
        for i in range(self.view_num):
            recon = self.decoders[i](latent_samples[i])
            recon_list.append(recon)
        return recon_list # shape: (batch_size, input_dim)

    def forward(self, X):
        latent_dist = self.encode(X)
        latent_samples = self.reparameterize(latent_dist)
        
        # Prepare decoder inputs: for each view, concatenate its continuous latent with the shared discrete latent
        decoder_inputs = []
        for i in range(self.view_num):
            cont_name = 'cont' + str(i+1)
            view_latent = torch.cat([latent_samples[cont_name], latent_samples['disc']], dim=1)  # shape: (batch_size, latent_cont_dim + latent_disc_dim)
            decoder_inputs.append(view_latent)
        
        recon_list = self.decode(decoder_inputs)  # shape: view_num * (batch_size, input_dim)
        return recon_list, latent_dist

class MultiVAELoss(nn.Module):
    def __init__(self, capacity, beta, class_num, iters_add_capacity, device=None):
        super(MultiVAELoss, self).__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.num_steps = 0
        # Calculate capacity parameters: [max_capacity, gamma, iters_add_capacity]
        self.cont_capacity = [capacity, beta, iters_add_capacity]
        self.disc_capacity = [np.log(class_num), beta, iters_add_capacity] 
        
    def forward(self, data, recon_data, latent_dist, num_steps, beta=1):
        self.num_steps = num_steps
        recon_loss = F.mse_loss(recon_data, data, reduction='sum')
        # Continuous capacity loss
        cont_max, cont_gamma, iters_add_cont_max = self.cont_capacity
        mean, logvar = latent_dist['cont']
        kl_cont_loss = self._kl_normal_loss(mean, logvar)
        step = cont_max / iters_add_cont_max
        cont_cap_current = min(step * num_steps, cont_max)
        cont_gamma = cont_gamma * beta
        cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)
        # Discrete capacity loss
        disc_max, disc_gamma, iters_add_disc_max = self.disc_capacity
        kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
        step = disc_max / iters_add_disc_max
        disc_cap_current = min(step * num_steps, disc_max)
        disc_gamma = disc_gamma * beta
        disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)
        total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss
        return total_loss
    
    def _kl_normal_loss(self, mean, logvar):
        """Calculates KL divergence for normal distribution"""
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
        kl_means = torch.mean(kl_values, dim=0)
        kl_loss = torch.sum(kl_means)
        return kl_loss
    
    def _kl_multiple_discrete_loss(self, alpha):
        """
        Calculates KL divergence for discrete distribution
        
        Note: alpha is a categorical distribution (softmax probabilities), 
        not one-hot vectors. The goal is to learn distributions that can 
        generate one-hot vectors (discrete class assignments) during inference.
        """
        return self._kl_discrete_loss(alpha)
    
    def _kl_discrete_loss(self, alpha):
        """
        Calculates KL divergence between categorical distribution and uniform distribution
        
        Parameters:
        -----------
        alpha : torch.Tensor
            Categorical distribution parameters (softmax probabilities), shape: (batch_size, class_num)
            During training: softmax probabilities
            During inference: can be converted to one-hot via argmax
        
        Returns:
        --------
        kl_loss : torch.Tensor
            KL(P || Uniform) = log(class_num) - H(P)
            where H(P) is the entropy of distribution P
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)]).to(self.device)  # log(class_num), entropy of uniform distribution
        neg_entropy = torch.sum(alpha * torch.log(alpha + 1e-12), dim=1)  # -H(P) for each sample
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)  # mean negative entropy
        kl_loss = log_dim + mean_neg_entropy  # KL = log(dim) - H(P)
        return kl_loss

def validation(model, dataset, view_num, data_size, class_num, device):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, idx = next(iter(test_loader))
    x_list = [x_list[v].to(device) for v in range(view_num)]; y = y.numpy()
    with torch.no_grad():
        encodings = model.encode(x_list)
        latent_distribution_discrete = encodings['disc'].cpu().detach().numpy() # shape: (batch_size, class_num), Discrete latent (view-common variable C)
        multiview_cz = [] # shape: (batch_size, latent_dim), Continuous latents (view-peculiar variables z1, z2, ..., zV) concatenated with discrete latent (view-common variable C)
        min_max_scaler = preprocessing.MinMaxScaler()
        for i in range(view_num):
            name = 'cont' + str(i + 1)
            # select the mean of the continuous latent distribution
            latent_distribution_continuous = encodings[name][0].cpu().detach().numpy() # shape: (batch_size, latent_dim), Continuous latent (view-peculiar variable zi)
            latent_distribution_continuous_normalized = min_max_scaler.fit_transform(latent_distribution_continuous) # shape: (batch_size, latent_dim), Continuous latent (view-peculiar variable zi) normalized
            multiview_cz.append(latent_distribution_continuous_normalized)
        kmeans = KMeans(n_clusters=class_num, n_init=100, random_state=42)
        print("Clustering results on discrete latent C (k-means):")
        p = kmeans.fit_predict(latent_distribution_discrete); nmi, ari, acc, pur = evaluate(y, p)
        print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
        print("Clustering results on discrete latent C (argmax):")
        p = latent_distribution_discrete.argmax(1); nmi, ari, acc, pur = evaluate(y, p)
        print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
        print("Clustering results on all continuous latents [z1, z2, ..., zV]:")
        latent_distribution_continuous_concatenated = np.concatenate(multiview_cz, axis=1)
        p = kmeans.fit_predict(latent_distribution_continuous_concatenated); nmi, ari, acc, pur = evaluate(y, p)
        print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
        print("Clustering results on all latents [C, z1, z2, ..., zV]:")
        multiview_cz.append(latent_distribution_discrete)
        latent_distribution_concatenated = np.concatenate(multiview_cz, axis=1)
        p = kmeans.fit_predict(latent_distribution_concatenated); nmi, ari, acc, pur = evaluate(y, p)
        print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-VAE')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--z_variables', default=10, type=int)
    parser.add_argument('--capacity', default=5, type=float)
    parser.add_argument('--beta', default=30, type=float)
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
    dataset, dims, view_num, data_size, class_num = load_data(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = VAE(dims=dims, z_variables=args.z_variables, class_num=class_num, view_num=view_num, device=device, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iters_add_capacity = data_size // args.batch_size * args.epochs # number of iterations to add capacity
    criterion = MultiVAELoss(capacity=args.capacity, beta=args.beta, class_num=class_num, iters_add_capacity=iters_add_capacity, device=device)
    
    ## 3. Train the model.
    model.train()
    num_steps = 0
    beta = [1.0] * view_num
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_idx, (x_list, y, idx) in enumerate(dataloader):
            num_steps += 1
            x_list = [x_list[v].to(device) for v in range(view_num)]
            optimizer.zero_grad()
            recon_list, latent_dist = model(x_list)
            # Calculate reconstruction losses for beta calculation on first iteration
            recon_losses = [F.mse_loss(recon_list[v], x_list[v], reduction='sum') for v in range(view_num)]
            if num_steps == 1:
                max_recon_loss = max(recon_losses)
                beta = [float(recon_losses[v]) / float(max_recon_loss) for v in range(view_num)]
            # Calculate total loss
            loss_list = []
            for v in range(view_num):
                loss = criterion(x_list[v], recon_list[v], {'cont': latent_dist['cont' + str(v + 1)], 'disc': latent_dist['disc']}, num_steps=num_steps, beta=beta[v])
                loss_list.append(loss)
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch: {}'.format(epoch + 1), 'Total Loss: {:.4f}'.format(epoch_loss / data_size))
    
    ## 4. Evaluate the model.
    validation(model, dataset, view_num, data_size, class_num, device)
