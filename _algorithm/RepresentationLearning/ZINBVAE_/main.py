import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, distribution='zinb', activation='softplus'):
        super(VAE, self).__init__()
        self.input_dim = input_dim; self.hidden_dim = hidden_dim; self.latent_dim = latent_dim; self.distribution = distribution; self.activation = activation
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_3 = nn.Linear(latent_dim, hidden_dim)
        self.fc_4 = nn.Linear(hidden_dim, input_dim)
        self.fc_5 = nn.Linear(hidden_dim, input_dim)
        self.log_theta = torch.nn.Parameter(torch.randn(input_dim)) # weights initialization with normal distribution
        self.activation = {'relu': F.relu, 'softplus': F.softplus}[self.activation]
        
    def encode(self, x):
        x = self.activation(self.fc_1(x))
        mean = self.fc_2_mean(x)
        logvar = self.fc_2_logvar(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        x = self.activation(self.fc_3(z))
        mean = self.fc_4(x)
        dropout = self.fc_5(x)
        theta = self.log_theta
        return mean, theta, dropout

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        de_mean, de_theta, de_dropout = self.decode(z) #zinb distribution
        return de_mean, de_theta, de_dropout, mean, logvar
    
    def get_latent_representation(self, x):
        mean, logvar = self.encode(x)
        return mean

    def generate(self, x, sample_shape, random=False):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar) if random else mean
        de_mean, de_theta, de_dropout = self.decode(z)
        de_mean = de_mean.exp()
        de_theta = de_theta.exp()
        nb_logits = (de_mean + 1e-4).log() - (de_theta + 1e-4).log()
        if self.distribution == 'zinb':
            distribution = ZeroInflatedNegativeBinomial(total_count=de_theta, logits=nb_logits, gate_logits = de_dropout, validate_args=False)
        elif self.distribution == 'nb':
            distribution = NegativeBinomial(total_count=de_theta, logits=nb_logits, validate_args=False)
        return distribution.sample(sample_shape) if random else distribution.mean # return the sample of zinb distribution or mean of zinb distribution
        
    def loss_function(self, x, de_mean, de_theta, de_dropout, mean, logvar):
        de_mean = de_mean.exp()
        de_theta = de_theta.exp()
        nb_logits = (de_mean + 1e-5).log() - (de_theta + 1e-5).log()
        if self.distribution == 'zinb':
            distribution = ZeroInflatedNegativeBinomial(total_count=de_theta, logits=nb_logits, gate_logits = de_dropout, validate_args=False)
        elif self.distribution == 'nb':
            distribution = NegativeBinomial(total_count=de_theta, logits=nb_logits, validate_args=False)
        reconstruction_loss = distribution.log_prob(x).sum(-1).mean()
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return - reconstruction_loss + kl_divergence


def main(data_dir, output_dir, lable_name, lr, num_epochs, batch_size, distribution):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adata = sc.read_h5ad(data_dir)
    # adata = preprocess(adata) if the data need to be preprocessed
    assert np.min(adata.X) >= 0, 'Your data has negative values, pleaes preprocess the data if you still want to use this data'
    data_loader = DataLoader(adata.X, batch_size=batch_size)
    vae = VAE(input_dim=adata.shape[1], hidden_dim=400, latent_dim=40, distribution=distribution) # distribution = 'nb' to use negative binomial distribution
    vae = vae.to(device)
    optimizer = optim.Adam(lr=lr, params=vae.parameters())
    vae.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            de_mean, de_theta, de_dropout, mean, logvar = vae(data)
            loss = vae.loss_function(data, de_mean, de_theta, de_dropout, mean, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
        
    data_latent_representation = []
    vae.eval()
    for x in data_loader:
        x = x.to(device)
        latent_representation = vae.get_latent_representation(x)
        latent_representation = latent_representation.cpu().detach().numpy() if device == 'cuda' else latent_representation.detach().numpy()
        data_latent_representation.append(latent_representation)
    data_latent_representation = np.concatenate(data_latent_representation, axis=0)
    
    adata.obsm['latent'] = data_latent_representation
    data_name = data_dir.split('/')[-1].split('.')[0]
    sc.pp.neighbors(adata, use_rep='latent')
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=lable_name, size=50)
    plt.savefig(os.path.join(output_dir, 'umap_{}_{}_embedding.png'.format(data_name, distribution)))
    plt.close()
    sc.tl.leiden(adata)
    sc.pl.umap(adata, color='leiden', size=50)
    plt.savefig(os.path.join(output_dir, 'umap_{}_{}_leiden.png'.format(data_name, distribution)))
    plt.close()
    torch.save(vae, os.path.join(output_dir, 'vae_{}_{}.pt'.format(data_name, distribution)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/example.h5ad', help='data directory')
    parser.add_argument('--output_dir', type=str, default='./result/', help='the output directory of the model and plots')
    parser.add_argument('--lable_name', type=str, default=None, help='the name of ground truth lable if applicable')
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate of the model')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--distribution', type=str, default='zinb', help='one distribution of [zinb,nb]')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.lable_name, args.lr, args.num_epochs, args.batch_size, args.distribution)
    