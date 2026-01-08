import os, sys, argparse, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate
    
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.Linear(128, latent_dim))
        
    def forward(self, x):
        return self.encoder(x)

class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.LayerNorm(128))
        self.mu_decoder = nn.Linear(128, output_dim)
        self.theta_decoder = nn.Linear(128, output_dim)
        self.pi_decoder = nn.Linear(128, output_dim)

    def forward(self, x):
        h = self.decoder(x)
        mu = torch.exp(self.mu_decoder(h))
        theta = torch.exp(self.theta_decoder(h))
        pi = torch.sigmoid(self.pi_decoder(h))
        return mu, theta, pi

class DCA(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = ZINBDecoder(latent_dim, input_dim)

    def forward(self, x):
        latent = self.encoder(x)
        mu, theta, pi = self.decoder(latent)
        return mu, theta, pi, latent

class ZINBLoss(nn.Module):
    def __init__(self, eps=1e-6, max_value=1e6):
        super().__init__()
        self.eps = eps
        self.max_value = max_value
        
    def forward(self, x, mu, theta, pi):
        """ Negative Log-Likelihood Loss for Zero-Inflated Negative Binomial Distribution (ZINB)
            - Negative Binomial Distribution (NB): 
              - p(x|mu,theta) = Gamma(theta+x) / (Gamma(theta) * Gamma(x+1)) * (theta/(theta+mu))^theta * (mu/(theta+mu))^x
            - Zero-Inflated Negative Binomial Distribution (ZINB): 
              - p(x|mu,theta,pi) = pi * delta(x == 0) + (1 - pi) * p(x|mu,theta)
        Args:
            x: true value [batch_size, feature_dim]
            mu: predicted mean [batch_size, feature_dim]
            theta: predicted dispersion [batch_size, feature_dim]
            pi: predicted zero-inflation probability [batch_size, feature_dim]
        Returns:
            loss: negative log-likelihood loss
        """
        mu = torch.clamp(mu, min=self.eps, max=self.max_value)
        theta = torch.clamp(theta, min=self.eps, max=self.max_value)
        pi = torch.clamp(pi, min=self.eps, max=1-self.eps)
        
        ## ZINB Loss by manual implementation:
        log_nonzero_case = torch.log(1 - pi) + torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1) + \
                           theta * (torch.log(theta) - torch.log(theta + mu)) + x * (torch.log(mu) - torch.log(theta + mu)) # shape: [batch_size, feature_dim]
        log_zero_case = torch.log(pi + (1 - pi) * torch.pow(theta / (theta + mu), theta)) # shape: [batch_size, feature_dim]
        loss = -torch.where(x < self.eps, log_zero_case, log_nonzero_case).mean() # shape: [1], whether x < eps is zero value or nonzero value
        return loss
    
        ## ZINB Loss by torch.distributions.ZeroInflatedNegativeBinomial API implementation: (Same as manual implementation)
        # total_count = theta
        # logits = torch.log(mu) - torch.log(theta)
        # gate_logits = torch.log(pi) - torch.log(1 - pi)
        # dist = ZeroInflatedNegativeBinomial(total_count=total_count, logits=logits, gate_logits=gate_logits, validate_args=False)
        # loss = -dist.log_prob(x).mean() # shape: [batch_size, feature_dim] -> shape: [1], whether x == 0 is zero value or nonzero value
        # return loss

def validation(model, dataset, data_size, class_num, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, _ = next(iter(test_loader))
    x_cat = torch.cat([x.to(device) for x in x_list], dim=1)
    y = y.numpy()
    with torch.no_grad():
        latent = model.encode(x_cat)
    latent = MinMaxScaler().fit_transform(latent.cpu().numpy())
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    preds = kmeans.fit_predict(latent)
    nmi, ari, acc, pur = evaluate(y, preds)
    print("Clustering on latent z (k-means):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

def benchmark_2019_NC_DCA_ZINB(dataset_name="BDGP",
                               latent_dim=20,
                               batch_size=256,
                               num_epochs=100,
                               learning_rate=5e-4,
                               seed=2,
                               verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    ## Attention: For DCA-ZINB model: the input x is the count data (non-negative integers).
    dataset, feature_dim_list, view_num, data_size, class_num = load_data(dataset_name)
    input_dim = sum(feature_dim_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = DCA(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = ZINBLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    ## 3. Train the model.
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x_list, y, idx) in enumerate(dataloader):
            x_cat = torch.cat([x.to(device) for x in x_list], dim=1)
            optimizer.zero_grad()
            mu, theta, pi, z = model(x_cat)
            loss = criterion(x_cat, mu, theta, pi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        print("Epoch: {} Total Loss: {:.4f}".format(epoch + 1, epoch_loss / data_size)) if verbose else None

    ## 4. Evaluate the model.
    nmi, ari, acc, pur = validation(model, dataset, data_size, class_num=class_num, verbose=verbose)
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCA-ZINB")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--latent_dim", default=20, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--seed", default=2, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_2019_NC_DCA_ZINB(
        dataset_name=args.dataset,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))