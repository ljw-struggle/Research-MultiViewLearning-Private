import os, sys, random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler # normalize the data to [0, 1] at column for default
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, embed_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(embed_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, output_dim))

    def forward(self, x):
        return self.decoder(x)

class MultiVAE(nn.Module):
    def __init__(self, view_num, feature_dim_list, hidden_dim=256, latent_cont_dim=10, latent_disc_dim=10, temperature=0.67):
        super(MultiVAE, self).__init__()
        self.view_num = view_num; self.hidden_dim = hidden_dim; self.latent_cont_dim = latent_cont_dim; self.latent_disc_dim = latent_disc_dim
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim 
        self.temperature = temperature # temperature for gumbel-softmax sampling
        # 1. View-specific encoders and decoders
        self.encoder_list = nn.ModuleList([Encoder(input_dim=feature_dim_list[i], embed_dim=self.hidden_dim) for i in range(self.view_num)])
        self.decoder_list = nn.ModuleList([Decoder(embed_dim=self.latent_dim, output_dim=feature_dim_list[i]) for i in range(self.view_num)])
        # 2. View-specific continuous latent inference modules
        self.mean_encoder_list = nn.ModuleList([nn.Linear(self.hidden_dim, self.latent_cont_dim) for i in range(self.view_num)]) # mean of continuous latent distribution
        self.logvar_encoder_list = nn.ModuleList([nn.Linear(self.hidden_dim, self.latent_cont_dim) for i in range(self.view_num)]) # log variance of continuous latent distribution
        # 3. View-common discrete latent inference module
        self.alpha_encoder = nn.Linear(self.hidden_dim * self.view_num, self.latent_disc_dim) # logits of categorical distribution
        
    def forward(self, input_list):
        mean_list, logvar_list, alpha = self.encode(input_list) # shape: view_num * (batch_size, latent_cont_dim), view_num * (batch_size, latent_cont_dim), (batch_size, latent_disc_dim)
        cont_sample_list, disc_sample = self.reparameterize(mean_list, logvar_list, alpha) # shape: view_num * (batch_size, latent_cont_dim), (batch_size, latent_disc_dim)
        latent_list = [torch.cat([cont_sample_list[i], disc_sample], dim=1) for i in range(self.view_num)] # shape: view_num * (batch_size, latent_dim)
        recon_list = self.decode(latent_list)  # shape: view_num * (batch_size, input_dim)
        return recon_list, (mean_list, logvar_list, alpha)

    def encode(self, input_list):
        hidden_list = [self.encoder_list[i](input_list[i]) for i in range(self.view_num)] # shape: view_num * (batch_size, hidden_dim)
        hidden_fusion = torch.cat(hidden_list, dim=1) # shape: (batch_size, hidden_dim * view_num)
        mean_list = [self.mean_encoder_list[i](hidden_list[i]) for i in range(self.view_num)] # shape: view_num * (batch_size, latent_cont_dim)
        logvar_list = [self.logvar_encoder_list[i](hidden_list[i]) for i in range(self.view_num)] # shape: view_num * (batch_size, latent_cont_dim)
        alpha = F.softmax(self.alpha_encoder(hidden_fusion), dim=1) # shape: (batch_size, latent_disc_dim)
        return mean_list, logvar_list, alpha
    
    def decode(self, latent_list):
        recon_list = [self.decoder_list[i](latent_list[i]) for i in range(self.view_num)] # shape: view_num * (batch_size, input_dim)
        return recon_list

    def reparameterize(self, mean_list, logvar_list, alpha): # Samples from latent distribution using reparameterization trick
        cont_sample_list = [self.sample_normal(mean_list[i], logvar_list[i]) for i in range(self.view_num)]  # shape: view_num * (batch_size, latent_cont_dim)
        disc_sample = self.sample_uniform_gumbel_softmax(alpha)  # shape: (batch_size, latent_disc_dim)
        return cont_sample_list, disc_sample

    def sample_normal(self, mean, logvar): # Samples from a normal distribution using reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar) # shape: (batch_size, latent_cont_dim)
            eps = torch.randn_like(std) # shape: (batch_size, latent_cont_dim), samples from standard normal distribution
            return mean + std * eps # shape: (batch_size, latent_cont_dim), samples from normal distribution
        else:
            return mean # shape: (batch_size, latent_cont_dim)

    def sample_uniform_gumbel_softmax(self, alpha, eps=1e-12): # Samples from a uniform gumbel-softmax distribution using reparameterization trick
        if self.training:
            unif = torch.rand(alpha.size(), device=alpha.device) # shape: (batch_size, latent_disc_dim), samples from uniform distribution
            gumbel = -torch.log(-torch.log(unif + eps) + eps) # shape: (batch_size, latent_disc_dim), samples from gumbel distribution
            log_alpha = torch.log(alpha + eps) # shape: (batch_size, latent_disc_dim), log of alpha
            logit = (log_alpha + gumbel) / self.temperature # shape: (batch_size, latent_disc_dim), logit of uniform gumbel-softmax distribution
            return F.softmax(logit, dim=1) # shape: (batch_size, latent_disc_dim), samples from uniform gumbel-softmax distribution
        else:
            _, max_alpha = torch.max(alpha, dim=1) # shape: (batch_size,), index of the maximum value in each sample
            one_hot_samples = torch.zeros(alpha.size(), device=alpha.device) # shape: (batch_size, latent_disc_dim)
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1), 1) # shape: (batch_size, latent_disc_dim)
            return one_hot_samples # shape: (batch_size, latent_disc_dim), one-hot encoding of the maximum value in each sample

class MultiVAELoss(nn.Module):
    def __init__(self, capacity_cont, capacity_disc, beta, iters_add_capacity): # shape: scalar, scalar, scalar, scalar
        super(MultiVAELoss, self).__init__()
        # Calculate capacity parameters: [max_capacity, gamma, iters_add_capacity]
        self.cont_capacity = [capacity_cont, beta, iters_add_capacity] # shape: (3,)
        self.disc_capacity = [capacity_disc, beta, iters_add_capacity] # shape: (3,)
        
    def forward(self, data, recon_list, mean_list, logvar_list, alpha, num_steps, beta_adaptive=1):
        recon_loss = F.mse_loss(recon_list, data, reduction='sum')
        # Continuous capacity loss
        cont_max, cont_gamma, iters_add_cont_max = self.cont_capacity
        kl_cont_loss = self._kl_continuous_loss(mean_list, logvar_list)
        step = cont_max / iters_add_cont_max
        cont_cap_current = min(step * num_steps, cont_max)
        cont_gamma = cont_gamma * beta_adaptive
        cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)
        # Discrete capacity loss
        disc_max, disc_gamma, iters_add_disc_max = self.disc_capacity
        kl_disc_loss = self._kl_discrete_loss(alpha)
        step = disc_max / iters_add_disc_max
        disc_cap_current = min(step * num_steps, disc_max)
        disc_gamma = disc_gamma * beta_adaptive
        disc_capacity_loss = disc_gamma * torch.abs(disc_cap_current - kl_disc_loss)
        total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss
        return total_loss
    
    def _kl_continuous_loss(self, mean, logvar):
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()) # shape: (batch_size, latent_cont_dim)
        kl_loss = torch.mean(torch.sum(kl_values, dim=1)) # shape: (batch_size, latent_cont_dim) -> (batch_size,) -> scalar
        return kl_loss # shape: scalar, KL divergence between normal distribution and standard normal distribution, [0, inf)
    
    def _kl_discrete_loss(self, alpha, eps=1e-12):
        log_dim = torch.Tensor([np.log(int(alpha.size()[-1]))]).to(alpha.device)  # shape: (1,), log(latent_disc_dim)
        neg_entropy = torch.sum(alpha * torch.log(alpha + eps), dim=1)  # shape: (batch_size,), negative entropy (-H(alpha)) for each sample
        kl_loss = torch.mean(neg_entropy) + log_dim # shape: scalar, KL = -H(alpha) + log(latent_disc_dim)
        return kl_loss # shape: scalar, KL divergence between categorical distribution and uniform distribution, [0, log(latent_disc_dim)]

def validation(model, dataset, view_num, data_size, class_num, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, idx = next(iter(test_loader))
    x_list = [x_list[v].to(device) for v in range(view_num)]; y = y.numpy()
    with torch.no_grad():
        mean_list, logvar_list, alpha = model.encode(x_list)
    mean_list = [MinMaxScaler().fit_transform(mean.cpu().detach().numpy()) for mean in mean_list] # shape: view_num * (batch_size, latent_cont_dim), Continuous latents (view-peculiar variables z1, z2, ..., zV) normalized
    alpha = alpha.cpu().detach().numpy() # shape: (batch_size, latent_disc_dim), Discrete latent (view-common variable C) normalized
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    print("Clustering results on discrete latent C (k-means):") if verbose else None
    p = kmeans.fit_predict(alpha); nmi, ari, acc, pur = evaluate(y, p)
    print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur)) if verbose else None
    print("Clustering results on discrete latent C (argmax):") if verbose else None
    p = alpha.argmax(1); nmi, ari, acc, pur = evaluate(y, p)
    print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur)) if verbose else None
    print("Clustering results on all continuous latents [z1, z2, ..., zV]:") if verbose else None
    mean_concat = np.concatenate(mean_list, axis=1) # shape: (batch_size, latent_cont_dim * view_num), Continuous latents (view-peculiar variables z1, z2, ..., zV)
    p = kmeans.fit_predict(mean_concat); nmi, ari, acc, pur = evaluate(y, p)
    print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur)) if verbose else None
    print("Clustering results on all latents [C, z1, z2, ..., zV]:") if verbose else None
    all_latents = np.concatenate([alpha, mean_concat], axis=1) # shape: (batch_size, latent_cont_dim * view_num + latent_disc_dim), All latents concatenated
    p = kmeans.fit_predict(all_latents); nmi, ari, acc, pur = evaluate(y, p)
    print('ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur
    
def benchmark_2021_ICCV_MultiVAE(dataset_name="BDGP", 
                                 hidden_dim=256, 
                                 latent_cont_dim=10, 
                                 capacity_cont=5, 
                                 beta=30, 
                                 batch_size=256, 
                                 num_epochs=100, 
                                 learning_rate=5e-4, 
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
    dataset, feature_dim_list, view_num, data_size, latent_disc_dim = load_data(dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MultiVAE(view_num=view_num, feature_dim_list=feature_dim_list, hidden_dim=hidden_dim, latent_cont_dim=latent_cont_dim, latent_disc_dim=latent_disc_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MultiVAELoss(capacity_cont=capacity_cont, capacity_disc=np.log(latent_disc_dim), beta=beta, iters_add_capacity=len(dataloader) * num_epochs)
    
    ## 3. Train the model.
    model.train()
    num_steps = 0
    beta_adaptive_list = [1.0] * view_num
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x_list, y, idx) in enumerate(dataloader):
            num_steps += 1
            x_list = [x_list[v].to(device) for v in range(view_num)]
            optimizer.zero_grad()
            recon_list, (mean_list, logvar_list, alpha) = model(x_list)
            if num_steps == 1: # Calculate reconstruction losses for beta calculation on first iteration
                recon_loss_list = [F.mse_loss(recon_list[v], x_list[v], reduction='sum') for v in range(view_num)] # shape: view_num, reconstruction losses
                beta_adaptive_list = [float(recon_loss_list[v]) / float(max(recon_loss_list)) for v in range(view_num)] # shape: view_num, beta values
            loss_list = [criterion(x_list[v], recon_list[v], mean_list[v], logvar_list[v], alpha, num_steps=num_steps, beta_adaptive=beta_adaptive_list[v]) for v in range(view_num)] # shape: view_num, loss values
            loss = sum(loss_list) # shape: scalar, total loss
            loss.backward() # backpropagation
            optimizer.step() # update model parameters
            epoch_loss += loss.item() # update epoch loss
        print('Epoch: {}'.format(epoch + 1), 'Total Loss: {:.4f}'.format(epoch_loss / data_size)) if verbose else None # print epoch loss
    
    ## 4. Evaluate the model.
    nmi, ari, acc, pur = validation(model, dataset, view_num, data_size, class_num=latent_disc_dim, verbose=verbose) # evaluate the model on the test set
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-VAE')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--latent_cont_dim', default=10, type=int)
    parser.add_argument('--capacity_cont', default=5, type=float)
    parser.add_argument('--beta', default=30, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--seed', default=2, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_2021_ICCV_MultiVAE(
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        latent_cont_dim=args.latent_cont_dim,
        capacity_cont=args.capacity_cont,
        beta=args.beta,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))