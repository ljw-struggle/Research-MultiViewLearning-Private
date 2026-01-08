import os, argparse, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from _utils import MMDataset, overall_performance_report
from torch.distributions import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from scipy.stats import nbinom

class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2_mu = nn.Linear(128, output_dim)
        self.fc2_theta = nn.Linear(128, output_dim)
        self.fc2_pi = nn.Linear(128, output_dim)

    def forward(self, z):
        h = F.relu(self.ln1(self.fc1(z)))
        mu = torch.exp(self.fc2_mu(h))
        theta = torch.exp(self.fc2_theta(h)) + 1e-4
        pi = torch.sigmoid(self.fc2_pi(h)).clamp(1e-3, 0.95)
        return mu, theta, pi

class NoisyEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, latent_dim * 2)

    def forward(self, x):
        noise = torch.randn_like(x) * 0.05  # Input noise
        x = x + noise
        h = F.relu(self.ln1(self.fc1(x)))
        mean, logvar = torch.chunk(self.fc2(h), 2, dim=-1)
        mean = mean.clamp(min=-5, max=5)
        logvar = F.softplus(logvar).clamp(min=1e-4, max=2)
        return mean, logvar
    
class ZINBVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = NoisyEncoder(input_dim, latent_dim)
        self.decoder = ZINBDecoder(latent_dim, input_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar).clamp(1e-3, 5)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        mu, theta, pi = self.decoder(z)
        return mu, theta, pi, mean, logvar
    
class ZINBLoss(nn.Module):
    def forward(self, x, mu, theta, pi):
        eps = 1e-6
        mu = torch.clamp(F.softplus(mu), min=eps, max=1e3)
        theta = torch.clamp(F.softplus(theta), min=eps, max=1e3)
        pi = torch.clamp(pi, min=1e-4, max=1-1e-4)

        negbinom = (
            torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1)
            + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
            + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        )

        zero_case = torch.log(pi + (1 - pi) * torch.exp(negbinom) + eps)
        nonzero_case = torch.log(1 - pi + eps) + negbinom

        loss = -torch.where(x < eps, zero_case, nonzero_case).mean()
        return loss

class ZINBVAE(nn.Module):
    def __init__(self, embed_dim=200, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[512, 512, 512], distribution='zinb'):
        super(ZINBVAE, self).__init__()
        self.embed_dim = embed_dim; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims; self.distribution = distribution
        self.encoder_list = nn.ModuleList([nn.Sequential(nn.Linear(feature_dims[i], hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(),
                                                         nn.Linear(hidden_dims[i], embed_dim)) for i in range(num_views)])
        self.fusion_net_mean = nn.Linear(num_views*embed_dim, embed_dim)
        self.fusion_net_logvar = nn.Linear(num_views*embed_dim, embed_dim)
        self.decoder_list = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU()) for i in range(num_views)])
        self.zinb_log_mean_list = nn.ModuleList([nn.Linear(hidden_dims[i], feature_dims[i]) for i in range(num_views)])
        self.zinb_dropout_list = nn.ModuleList([nn.Linear(hidden_dims[i], feature_dims[i]) for i in range(num_views)])
        self.zinb_log_theta_list = nn.ParameterList([nn.Parameter(torch.randn(feature_dims[i])) for i in range(num_views)])
    
    def forward(self, x):
        encoded_output_list = [self.encoder_list[i](x[i]) for i in range(self.num_views)]
        mean = self.fusion_net_mean(torch.cat(encoded_output_list, dim=1))
        logvar = self.fusion_net_logvar(torch.cat(encoded_output_list, dim=1))
        z = self.reparameterize(mean, logvar)
        decoded_output_list = [self.decoder_list[i](z) for i in range(self.num_views)]
        zinb_log_mean_list = [self.zinb_log_mean_list[i](decoded_output_list[i]) for i in range(self.num_views)]
        zinb_dropout_list = [self.zinb_dropout_list[i](decoded_output_list[i]) for i in range(self.num_views)]
        zinb_log_theta_list = [self.zinb_log_theta_list[i] for i in range(self.num_views)]
        return decoded_output_list, mean, logvar, zinb_log_mean_list, zinb_dropout_list, zinb_log_theta_list
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def reconstruction_loss(self, x):
        decoded_output_list, mean, logvar = self.forward(x)
        reconstruction_loss = sum([F.mse_loss(decoded_output_list[v], x[v], reduction='mean') for v in range(self.num_views)])
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kld_loss
    
    def generate(self, x, random=False):
        encoded_output_list = [self.encoder_list[i](x[i]) for i in range(self.num_views)]
        mean = self.fusion_net_mean(torch.cat(encoded_output_list, dim=1))
        logvar = self.fusion_net_logvar(torch.cat(encoded_output_list, dim=1))
        z = self.reparameterize(mean, logvar) if random else mean
        decoded_output_list = [self.decoder_list[i](z) for i in range(self.num_views)]
        zinb_log_mean_list = [self.zinb_log_mean_list[i](decoded_output_list[i]) for i in range(self.num_views)]
        zinb_dropout_list = [self.zinb_dropout_list[i](decoded_output_list[i]) for i in range(self.num_views)]
        zinb_log_theta_list = [self.zinb_log_theta_list[i] for i in range(self.num_views)]
        zinb_mean_list = [zinb_log_mean_list[i].exp() for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        zinb_theta_list = [zinb_log_theta_list[i].exp() for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        nb_logits_list = [((zinb_mean_list[i] + 1e-4).log() - (zinb_theta_list[i] + 1e-4).log()) for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        if self.distribution == 'zinb':
            distribution = [ZeroInflatedNegativeBinomial(total_count=zinb_theta_list[i], logits=nb_logits_list[i], gate_logits = zinb_dropout_list[i], validate_args=False) for i in range(self.num_views)]
            generated_output_list = [distribution[i].sample() if random else distribution[i].mean for i in range(self.num_views)]
        elif self.distribution == 'nb':
            distribution = [NegativeBinomial(total_count=zinb_theta_list[i], logits=nb_logits_list[i], validate_args=False) for i in range(self.num_views)]
            generated_output_list = [distribution[i].sample() if random else distribution[i].mean for i in range(self.num_views)]
        return generated_output_list # shape: [num_views, batch_size, feature_dim]
    
    def reconstruction_loss(self, x):
        decoded_output_list, mean, logvar, zinb_log_mean_list, zinb_dropout_list, zinb_log_theta_list = self.forward(x)
        zinb_mean_list = [zinb_log_mean_list[i].exp() for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        zinb_theta_list = [zinb_log_theta_list[i].exp() for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        nb_logits_list = [((zinb_mean_list[i] + 1e-4).log() - (zinb_theta_list[i] + 1e-4).log()) for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        if self.distribution == 'zinb':
            distribution = [ZeroInflatedNegativeBinomial(total_count=zinb_theta_list[i], logits=nb_logits_list[i], gate_logits = zinb_dropout_list[i], validate_args=False) for i in range(self.num_views)]
        elif self.distribution == 'nb':
            distribution = [NegativeBinomial(total_count=zinb_theta_list[i], logits=nb_logits_list[i], validate_args=False) for i in range(self.num_views)]
        reconstruction_loss = sum([distribution[i].log_prob(x[i]).sum(-1).mean() for i in range(self.num_views)]) # Maximum Likelihood Estimation (MLE)
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) # Kullback-Leibler Divergence (KLD)
        return - reconstruction_loss + kld_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='the data dir [default: ./data/data_bulk_multiomics/BRCA/]')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/ZINBVAE/BRCA/', help='the output dir [default: ./result/data_bulk_multiomics/ZINBVAE/BRCA/]')
    parser.add_argument('--seed', default=0, type=int, help='random seed [default: 0]')
    parser.add_argument('--latent_dim', type=float, default=20, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', type=int, default=[100], help='number of epochs to train [default: 500]')
    parser.add_argument('--learning_rate', type=float, default=[5e-3], help='learning rate [default: 1e-3]')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--times', type=int, default=5, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    for t in range(args.times):
        dataset = MMDataset(args.data_dir, concat_data=False); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
        data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = ZINBVAE(embed_dim=args.latent_dim, num_views=data_views, feature_dims=data_features, hidden_dims=[512, 512, 512], distribution='zinb').to(device) # distribution = 'nb' to use negative binomial distribution
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
        for epoch in range(args.epoch_num[0]):
            model.train()
            losses = []
            for batch_idx, (x, y, _) in enumerate(dataloader):
                x = [x[i].to(device) for i in range(len(x))]; y = y.to(device)
                optimizer.zero_grad()
                loss = model.reconstruction_loss(x)
                loss.backward()
                # # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                losses.append(loss.item())
            print(f'Training Epoch: {epoch} Loss: {np.mean(losses):.4f}')
            
        model.eval()
        encoded_output_list, mean, logvar, zinb_log_mean_list, zinb_dropout_list, zinb_log_theta_list = model.forward(data)
        multi_times_embedding_list.append(mean.detach().cpu().numpy())
    
    overall_performance_report(multi_times_embedding_list, None, label, args.output_dir)
