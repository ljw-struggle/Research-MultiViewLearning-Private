import os, sys, random, argparse, numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
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
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, output_dim))

    def forward(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, 64), nn.LeakyReLU(0.2, inplace=True), nn.Linear(64, 1), nn.Sigmoid())
        
    def forward(self, x):
        return self.model(x)

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, dim=0)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

class DAMC(nn.Module):
    def __init__(self, view=2, input_dims=[1750, 79], feature_dim=10, class_num=10, alpha=1.0):
        super(DAMC, self).__init__()
        self.view = view
        self.feature_dim = feature_dim
        # Encoders for each view
        self.encoders = nn.ModuleList([Encoder(input_dims[v], feature_dim) for v in range(view)])
        # Decoders for each view
        self.decoders = nn.ModuleList([Decoder(feature_dim, input_dims[v]) for v in range(view)])
        # Clustering layer for global latent
        self.clustering_layer = ClusteringLayer(class_num, feature_dim, alpha)
        # Discriminators for each view
        self.discriminators = nn.ModuleList([Discriminator(input_dims[v]) for v in range(view)])
    
    def forward(self, x_list):
        # Encode each view to latent
        latent_list = [self.encoders[v](x_list[v]) for v in range(self.view)]
        # Average latent to get global latent
        global_latent = sum(latent_list) / self.view
        # Decode global latent to reconstruct each view
        decoded_list = [self.decoders[v](global_latent) for v in range(self.view)]
        # Clustering on global latent
        q = self.clustering_layer(global_latent)
        return decoded_list, global_latent, q

def validation(model, data, label, view, class_num, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        _, global_latent, q = model(data)
    q_np = q.detach().cpu().numpy()
    y_pred = np.argmax(q_np, axis=1)
    nmi, ari, acc, pur = evaluate(label.cpu().detach().numpy(), y_pred)
    if verbose:
        print("Clustering on global latent:")
        print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur))
    return nmi, ari, acc, pur

def benchmark_2019_IJCAI_DAMC(dataset_name='BDGP', batch_size=256, pretrain_learning_rate=0.001, pretrain_epochs=500, 
                               gan_learning_rate=0.001, gan_epochs=100, dec_learning_rate=0.001, maxiter=30000, 
                               update_interval=1000, lambda_mse=1.0, lambda_gan=1.0, lambda_clustering=0.1, 
                               verbose=False, random_state=42):
    """
    DAMC: Deep Adversarial Multi-view Clustering
    
    Three training stages:
    1. Autoencoder pretraining: Learn to encode and decode
    2. Adversarial training: Use discriminators to improve reconstruction quality
    3. DEC clustering: Joint optimization with clustering loss
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    data = [data[v].to(device) for v in range(view)]
    label = label.to(device)

    ## 3. Initialize model and optimizers.
    model = DAMC(view=view, input_dims=dims, feature_dim=10, class_num=class_num).to(device)
    criterion_mse = nn.MSELoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean', log_target=False)
    criterion_bce = nn.BCELoss()
    
    ## 4. Stage 1: Pretrain the AutoEncoder.
    print('Stage 1: AutoEncoder Pretraining') if verbose else None
    optimizer_pretrain = torch.optim.Adam(list(model.encoders.parameters()) + list(model.decoders.parameters()), lr=pretrain_learning_rate)
    model.train()
    for epoch in range(pretrain_epochs):
        perm = torch.randperm(data_size)
        total_loss_list = []
        for i in range(0, data_size, batch_size):
            optimizer_pretrain.zero_grad()
            data_batch = [data[v][perm[i:i+batch_size]] for v in range(view)]
            decoded_list, _, _ = model(data_batch)
            loss = sum([criterion_mse(decoded_list[v], data_batch[v]) for v in range(view)])
            loss.backward()
            optimizer_pretrain.step()
            total_loss_list.append(loss.item())
        print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(total_loss_list):.4f}') if verbose else None
    
    ## 5. Initialize cluster centers with KMeans.
    model.eval()
    with torch.no_grad():
        _, global_latent, _ = model(data)
    global_latent_np = global_latent.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    kmeans.fit(global_latent_np)
    model.clustering_layer.cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device))
    
    ## 6. Stage 2: Adversarial Training.
    print('Stage 2: Adversarial Training') if verbose else None
    optimizer_G = torch.optim.Adam(list(model.encoders.parameters()) + list(model.decoders.parameters()), lr=gan_learning_rate)
    optimizer_D = [torch.optim.Adam(model.discriminators[v].parameters(), lr=gan_learning_rate) for v in range(view)]
    
    for epoch in range(gan_epochs):
        perm = torch.randperm(data_size)
        for i in range(0, data_size, batch_size):
            data_batch = [data[v][perm[i:i+batch_size]] for v in range(view)]
            # Train Discriminators
            for v in range(view):
                optimizer_D[v].zero_grad()
                # Real data
                real_pred = model.discriminators[v](data_batch[v])
                real_target = torch.ones_like(real_pred)
                loss_D_real = criterion_bce(real_pred, real_target)
                # Fake data (from decoder)
                with torch.no_grad():
                    decoded_list, _, _ = model(data_batch)
                fake_pred = model.discriminators[v](decoded_list[v].detach())
                fake_target = torch.zeros_like(fake_pred)
                loss_D_fake = criterion_bce(fake_pred, fake_target)
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D[v].step()
            # Train Generator (Encoder + Decoder)
            optimizer_G.zero_grad()
            decoded_list, _, _ = model(data_batch)
            # Reconstruction loss (MSE)
            recon_loss = sum([criterion_mse(decoded_list[v], data_batch[v]) for v in range(view)])
            # Adversarial loss
            gan_loss = 0
            for v in range(view):
                fake_pred = model.discriminators[v](decoded_list[v])
                gan_target = torch.ones_like(fake_pred)
                gan_loss += criterion_bce(fake_pred, gan_target)
            # Total generator loss
            loss_G = lambda_mse * recon_loss + lambda_gan * gan_loss
            loss_G.backward()
            optimizer_G.step()
        if verbose and (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                _, global_latent, _ = model(data)
            global_latent_np = global_latent.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(global_latent_np)
            nmi, ari, acc, pur = evaluate(label.cpu().detach().numpy(), y_pred)
            print(f'[Adversarial] Epoch {epoch + 1}, ACC: {acc:.4f}, NMI: {nmi:.4f}') if verbose else None
    
    ## 7. Stage 3: DEC Clustering Training.
    print('Stage 3: DEC Clustering Training') if verbose else None
    optimizer_train = torch.optim.Adam(model.parameters(), lr=dec_learning_rate)
    index_array = np.arange(data_size)
    p = None
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            model.eval()
            with torch.no_grad():
                _, _, q = model(data)
            q_np = q.detach().cpu().numpy()
            p_np = ClusteringLayer.target_distribution(torch.tensor(q_np)).numpy()
            p = torch.tensor(p_np, dtype=torch.float32).to(device)
            y_pred = np.argmax(q_np, axis=1)
            nmi, ari, acc, pur = evaluate(label.cpu().detach().numpy(), y_pred)
            print("Iteration: {}; ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(ite, acc, nmi, ari, pur)) if verbose else None
        
        model.train()
        optimizer_train.zero_grad()
        idx = index_array[(ite * batch_size) % data_size:min((ite * batch_size) % data_size + batch_size, data_size)]
        if len(idx) == 0:
            continue
        data_batch = [data[v][idx] for v in range(view)]
        decoded_list, _, q_batch = model(data_batch)
        
        # Reconstruction loss
        recon_loss = sum([criterion_mse(decoded_list[v], data_batch[v]) for v in range(view)])
        # Clustering loss
        if p is not None:
            clustering_loss = criterion_kl(torch.log(q_batch + 1e-10), p[idx])
        else:
            clustering_loss = torch.tensor(0.0).to(device)
        
        loss = lambda_mse * recon_loss + lambda_clustering * clustering_loss
        loss.backward()
        optimizer_train.step()
    
    ## 8. Final evaluation.
    nmi, ari, acc, pur = validation(model, data, label, view, class_num, verbose=verbose)
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAMC")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--pretrain_learning_rate", default=0.001, type=float)
    parser.add_argument("--pretrain_epochs", default=500, type=int)
    parser.add_argument("--gan_learning_rate", default=0.001, type=float)
    parser.add_argument("--gan_epochs", default=100, type=int)
    parser.add_argument("--dec_learning_rate", default=0.001, type=float)
    parser.add_argument("--maxiter", default=30000, type=int)
    parser.add_argument("--update_interval", default=1000, type=int)
    parser.add_argument("--lambda_mse", default=1.0, type=float)
    parser.add_argument("--lambda_gan", default=1.0, type=float)
    parser.add_argument("--lambda_clustering", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    
    nmi, ari, acc, pur = benchmark_2019_IJCAI_DAMC(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        pretrain_learning_rate=args.pretrain_learning_rate,
        pretrain_epochs=args.pretrain_epochs,
        gan_learning_rate=args.gan_learning_rate,
        gan_epochs=args.gan_epochs,
        dec_learning_rate=args.dec_learning_rate,
        maxiter=args.maxiter,
        update_interval=args.update_interval,
        lambda_mse=args.lambda_mse,
        lambda_gan=args.lambda_gan,
        lambda_clustering=args.lambda_clustering,
        verbose=False,
        random_state=args.seed
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
