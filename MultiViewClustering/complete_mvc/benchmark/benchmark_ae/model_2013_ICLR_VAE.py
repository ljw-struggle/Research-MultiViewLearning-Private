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
except ImportError:
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

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=10):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.mean_encoder = nn.Linear(hidden_dim, latent_dim)
        self.logvar_encoder = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, input):
        mean, logvar = self.encode(input) # shape: (batch_size, latent_dim), (batch_size, latent_dim) -> mean, logvar
        latent = self.reparameterize(mean, logvar) # shape: (batch_size, latent_dim) -> latent
        recon = self.decode(latent) # shape: (batch_size, input_dim) -> recon
        return recon, (mean, logvar) # shape: (batch_size, input_dim), (batch_size, latent_dim), (batch_size, latent_dim) -> recon, (mean, logvar)

    def encode(self, input): # Encode the input into mean and logvar
        hidden = self.encoder(input) # shape: (batch_size, hidden_dim)
        mean = self.mean_encoder(hidden) # shape: (batch_size, latent_dim)
        logvar = self.logvar_encoder(hidden) # shape: (batch_size, latent_dim)
        # mean = mean.clamp(min=-5, max=5)
        # logvar = logvar.clamp(min=-10, max=2)
        return mean, logvar # shape: (batch_size, latent_dim), (batch_size, latent_dim) -> (mean, logvar)
    
    def decode(self, latent): # Decode the latent into reconstruction
        return self.decoder(latent) # shape: (batch_size, input_dim)

    def reparameterize(self, mean, logvar): # Samples from latent distribution using reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar) # shape: (batch_size, latent_dim)
            eps = torch.randn_like(std) # shape: (batch_size, latent_dim)
            return mean + eps * std # shape: (batch_size, latent_dim) -> latent
        else:
            return mean # shape: (batch_size, latent_dim) -> latent

class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta
        
    def forward(self, input, recon, mean, logvar):
        recon_loss = F.mse_loss(recon, input, reduction='sum')
        kl_loss = self.kl_divergence(mean, logvar)
        return recon_loss + self.beta * kl_loss

    def kl_divergence(self, mean, logvar):
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()) # shape: (batch_size, latent_dim)
        return torch.mean(torch.sum(kl_values, dim=1)) # shape: (batch_size,) -> scalar

def validation(model, dataset, data_size, class_num, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, _ = next(iter(test_loader))
    x_cat = torch.cat([x.to(device) for x in x_list], dim=1); y = y.numpy()
    with torch.no_grad():
        mean, _ = model.encode(x_cat)
    latent = MinMaxScaler().fit_transform(mean.cpu().numpy()) # shape: (batch_size, latent_dim) -> latent
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    preds = kmeans.fit_predict(latent) # shape: (batch_size,) -> preds
    nmi, ari, acc, pur = evaluate(y, preds) # shape: scalar, scalar, scalar, scalar -> nmi, ari, acc, pur
    print("Clustering on latent z (k-means):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

def benchmark_2013_ICLR_VAE(dataset_name="BDGP", 
                            hidden_dim=256, 
                            latent_dim=10, 
                            beta=1.0, 
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
    dataset, feature_dim_list, view_num, data_size, class_num = load_data(dataset_name)
    input_dim = sum(feature_dim_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    criterion = VAELoss(beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    ## 3. Train the model.
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x_list, y, idx) in enumerate(dataloader):
            x_cat = torch.cat([x.to(device) for x in x_list], dim=1) # shape: (batch_size, input_dim)
            optimizer.zero_grad() # reset gradients
            recon, (mean, logvar) = model(x_cat) # shape: (batch_size, input_dim), (batch_size, latent_dim), (batch_size, latent_dim) -> recon, (mean, logvar)
            loss = criterion(x_cat, recon, mean, logvar) # shape: scalar, total loss
            loss.backward() # backpropagation
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step() # update model parameters
            epoch_loss += loss.item() # update epoch loss
        print("Epoch: {} Total Loss: {:.4f}".format(epoch + 1, epoch_loss / data_size)) if verbose else None

    ## 4. Evaluate the model.
    nmi, ari, acc, pur = validation(model, dataset, data_size, class_num=class_num, verbose=verbose) # evaluate the model on the test set
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--latent_dim", default=10, type=int)
    parser.add_argument("--beta", default=1.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--seed", default=2, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_2013_ICLR_VAE(
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))