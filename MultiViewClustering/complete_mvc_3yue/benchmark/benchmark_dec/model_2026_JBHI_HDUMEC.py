import os, sys, random, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class DUCMME(nn.Module):
    def __init__(self, embed_dim=200, num_samples=10000, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[512, 512, 512], n_clusters=10, alpha=1.0):
        super(DUCMME, self).__init__()
        self.embed_dim = embed_dim; self.num_samples = num_samples; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims; self.n_clusters = n_clusters; self.alpha = alpha
        # 1. Multi-view Feature Extraction by Fusion-Net
        self.fusion_net_encoder = nn.ModuleList([nn.Sequential(nn.Linear(feature_dims[i], hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(),
                                                               nn.Linear(hidden_dims[i], embed_dim)) for i in range(num_views)]) # encode each view
        self.fusion_net_mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=10, batch_first=True) # batch_first=True: (batch_size, seq_len, hidden_dim)
        self.fusion_net_linear = nn.Linear(num_views*embed_dim, embed_dim) # linear projection of the fused encoded features
        # 2. Uncertainty-Aware Reconstruction by Reconstruction-Net and Uncertainty-Net
        self.reconstruct_net_list = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(), 
                                                                 nn.Linear(hidden_dims[i], feature_dims[i])) for i in range(num_views)]) # reconstruct each view
        self.uncertainty_net_list = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(), 
                                                                 nn.Linear(hidden_dims[i], feature_dims[i])) for i in range(num_views)]) # predict uncertainty for each view
        # 3. Deep Embedding Clustering by DEC
        self._cluster_centers = nn.Parameter(torch.Tensor(self.n_clusters, self.embed_dim))
        nn.init.xavier_uniform_(self._cluster_centers.data)
        
    def forward_embedding(self, x):
        encoded_output_list = [self.fusion_net_encoder[i](x[i]) for i in range(self.num_views)] # encode each view
        encoded_output_list = torch.stack(encoded_output_list, dim=1) # stack the encoded features from all views, (batch_size, num_views, embed_dim)
        encoded_output_list, _ = self.fusion_net_mha(encoded_output_list, encoded_output_list, encoded_output_list) # fuse the encoded features from all views by a multihead attention, (batch_size, num_views, embed_dim)
        encoded_output_list = encoded_output_list.contiguous().view(encoded_output_list.shape[0], -1) # flatten the encoded features, (batch_size, num_views*embed_dim)
        embedding = self.fusion_net_linear(encoded_output_list) # linear projection of the fused encoded features
        return embedding # get the embedding of the latent space H, (batch_size, embed_dim)

    def forward_uncertainty_aware_reconstruction(self, x):
        embedding = self.forward_embedding(x) # shape: [batch_size, embed_dim]
        reconstructions = [self.reconstruct_net_list[i](embedding) for i in range(self.num_views)] # reconstruct each view
        uncertainties = [self.uncertainty_net_list[i](embedding) for i in range(self.num_views)] # predict uncertainty for each view
        return reconstructions, uncertainties
        
    def forward_similarity_matrix_q(self, x): # calculate the similarity matrix q using t-distribution
        embedding = self.forward_embedding(x) # shape: [batch_size, embed_dim]
        q = 1.0 / (1.0 + torch.sum((embedding.unsqueeze(1) - self._cluster_centers) ** 2, dim=2) / self.alpha) # shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # , shape: [batch_size, n_clusters]
        q = q / torch.sum(q, dim=1, keepdim=True) # Normalize q to sum to 1 across clusters, shape: [batch_size, n_clusters]
        return q, embedding # q can be regarded as the probability of the sample belonging to each cluster
    
    @property
    def cluster_centers(self):
        return self._cluster_centers.data.detach().cpu().numpy() # shape: (n_clusters, embed_dim)
    
    @cluster_centers.setter
    def cluster_centers(self, centers): # shape: (n_clusters, embed_dim)
        centers = torch.tensor(centers, dtype=torch.float32, device=self._cluster_centers.device)
        self._cluster_centers.data.copy_(centers) # copy the cluster centers to the model, set the cluster centers to the new cluster centers
        
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, dim=0) # shape: [batch_size, n_clusters]
        p = weight / torch.sum(weight, dim=1, keepdim=True) # Normalize p to sum to 1 across clusters, shape: [batch_size, n_clusters]
        return p.clone().detach()
    
    def reconstruction_loss(self, x):
        x_rec, _ = self.forward_uncertainty_aware_reconstruction(x) # reconstruct each view and predict uncertainty
        return sum([F.mse_loss(x_rec[v], x[v], reduction='mean') for v in range(self.num_views)]) # sum the losses from all views
    
    def uncertainty_aware_reconstruction_loss(self, x):
        x_rec, log_sigma_2 = self.forward_uncertainty_aware_reconstruction(x) # reconstruct each view and predict uncertainty
        # Clip log_sigma_2 to prevent numerical instability (exp(-log_sigma_2) overflow/underflow)
        # Clamp to reasonable range: -10 to 10, which corresponds to sigma^2 from exp(-10) to exp(10)
        log_sigma_2 = [torch.clamp(log_sigma_2[v], min=-10.0, max=10.0) for v in range(self.num_views)] # shape: [num_views, batch_size, feature_dim] for numerically stable computation
        return sum([0.5 * torch.mean((x_rec[v] - x[v])**2 * torch.exp(-log_sigma_2[v]) + log_sigma_2[v]) for v in range(self.num_views)]) # uncertainty is equal to log_sigma_2
    
    def clustering_loss(self, x, p):
        q, _ = self.forward_similarity_matrix_q(x) # shape: [batch_size, n_clusters]
        return F.kl_div(q.log(), p, reduction='batchmean') # shape: ()


def benchmark_2026_JBHI_HDUMEC(dataset_name='BDGP', latent_dim=20, epoch_num=[200, 100, 100], learning_rate=[5e-3, 1e-3, 1e-3], tol=1e-3, verbose=False, random_state=0):
    # HDUMEC: Heterogeneous Data Uncertainty-Aware Multi-view Clustering.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state); 
    torch.cuda.manual_seed_all(random_state); 
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False); data, label, idx = next(iter(dataloader)); data = [x.to(device) for x in data]; label = label.to(device) # shape: [view, data_size, feature_dim], [data_size]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=True)
    model = DUCMME(embed_dim=latent_dim, feature_dims=dims, num_views=view, hidden_dims=[512, 512, 512], num_samples=data_size, n_clusters=class_num, alpha=1.0).to(device)
    ## === Stage 1: Uncertainty-Aware Reconstruction Pretraining ===
    print("\n=== Stage 1: Uncertainty-Aware Reconstruction Pretraining ===") if verbose else None
    print("Basic reconstruction training...") if verbose else None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
    for epoch in range(epoch_num[0]):
        model.train()
        losses = []
        for batch_idx, (x, y, idx) in enumerate(dataloader):
            x = [x.to(device) for x in x]
            optimizer.zero_grad()
            loss = model.reconstruction_loss(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}') if verbose else None
    # model.eval()
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=random_state)
    preds = kmeans.fit_predict(embedding)
    nmi_val, ari_val, acc_val, pur_val = evaluate(label.cpu().detach().numpy(), preds)
    print(f"Pretraining completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}") if verbose else None
    print("Uncertainty-aware reconstruction finetuning...") if verbose else None
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[1])
    for epoch in range(epoch_num[1]):
        model.train()
        losses = []
        for batch_idx, (x, y, idx) in enumerate(dataloader):
            x = [x.to(device) for x in x]
            optimizer.zero_grad()
            loss = model.uncertainty_aware_reconstruction_loss(x)
            loss.backward()
            # # Gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}') if verbose else None
    # model.eval()
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=random_state)
    preds = kmeans.fit_predict(embedding)
    nmi_val, ari_val, acc_val, pur_val = evaluate(label.cpu().detach().numpy(), preds)
    print(f"Finetuning completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}") if verbose else None

    ## === Stage 2: Deep Embedding Clustering by DEC ===
    print("\n=== Stage 2: Deep Embedding Clustering ===") if verbose else None
    print("Cluster center initialization...") if verbose else None
    model.eval()
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=class_num, n_init=20, random_state=random_state)
    preds = kmeans.fit_predict(embedding)
    nmi_val, ari_val, acc_val, pur_val = evaluate(label.cpu().detach().numpy(), preds)
    print(f"Cluster center initialization completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}") if verbose else None
    model.cluster_centers = kmeans.cluster_centers_ # shape: (n_clusters, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[2])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) # learning rate = 1e-3 * 0.9^(20/20) = 9e-4
    losses = []
    for epoch in range(epoch_num[2]):
        # Update target distribution periodically
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                q, embedding = model.forward_similarity_matrix_q(data)
                p = model.target_distribution(q)
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            nmi_val, ari_val, acc_val, pur_val = evaluate(label.cpu().detach().numpy(), y_pred)
            if epoch == 0:
                delta_label = 1.0
                y_pred_last = y_pred.copy()
                print(f'[Epoch {epoch}] ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}, Delta: {delta_label:.4f}') if verbose else None
            else:
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()
                print(f'[Epoch {epoch}] ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}, Delta: {delta_label:.4f}') if verbose else None
                if delta_label < tol:
                    print('Converged, stopping training...') if verbose else None
                    break
        # Training based on the target distribution that is updated periodically
        model.train()
        losses = []
        for batch_idx, (x, y, idx) in enumerate(dataloader):
            x = [x.to(device) for x in x]
            optimizer.zero_grad()
            loss = model.clustering_loss(x, p[idx])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}') if verbose else None
    print(f'Final ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}') if verbose else None
    return nmi_val, ari_val, acc_val, pur_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDUMEC")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--latent_dim", default=20, type=int)
    parser.add_argument("--epoch_num", default=[200, 100, 100], type=int, nargs=3, help="Epoch numbers for [pretrain, finetune, clustering]")
    parser.add_argument("--learning_rate", default=[5e-3, 1e-3, 1e-3], type=float, nargs=3, help="Learning rates for [pretrain, finetune, clustering]")
    parser.add_argument("--tol", default=1e-3, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    
    nmi, ari, acc, pur = benchmark_2026_JBHI_HDUMEC(
        dataset_name=args.dataset,
        latent_dim=args.latent_dim,
        epoch_num=args.epoch_num,
        learning_rate=args.learning_rate,
        tol=args.tol,
        verbose=False,
        random_state=args.seed
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))