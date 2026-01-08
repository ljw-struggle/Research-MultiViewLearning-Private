import os, random, argparse, itertools, numpy as np, scipy.io as sio
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from _utils import MMDataset, clustering_acc, overall_performance_report

class DUCMME(nn.Module):
    def __init__(self, embed_dim=200, num_samples=10000, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[512, 512, 512], n_clusters=10, alpha=1.0):
        super(DUCMME, self).__init__()
        self.embed_dim = embed_dim; self.num_samples = num_samples; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims; self.n_clusters = n_clusters; self.alpha = alpha
        # 1. Multi-view Feature Extraction by Fusion-Net
        self.fusion_net_encoder = nn.ModuleList([nn.Sequential(nn.Linear(feature_dims[i], hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(),
                                                               nn.Linear(hidden_dims[i], embed_dim)) for i in range(num_views)]) # encode each view
        self.fusion_net_mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=10, batch_first=True) # batch_first=True: (batch_size, seq_len, hidden_dim)
        self.fusion_net_linear = nn.Linear(3*embed_dim, embed_dim) # linear projection of the fused encoded features
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
        return sum([0.5 * torch.mean((x_rec[v] - x[v])**2 * torch.exp(-log_sigma_2[v]) + log_sigma_2[v]) for v in range(self.num_views)]) # uncertainty is equal to log_sigma_2
    
    def clustering_loss(self, x, p):
        q, _ = self.forward_similarity_matrix_q(x) # shape: [batch_size, n_clusters]
        return F.kl_div(q.log(), p, reduction='batchmean') # shape: ()

if __name__ == "__main__":
    ## === Step 1: Environment & Reproducibility Setup ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/DEC/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', default=20, type=int, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', default=2000, type=int, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', default=[200, 100, 100], type=int, help='number of epochs to train [default: [200, 100, 100]]')
    parser.add_argument('--learning_rate', default=[5e-3, 1e-3, 1e-3], type=float, help='learning rate [default: [5e-3, 1e-3, 1e-3]]')
    parser.add_argument('--log_interval', default=10, type=int, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--update_interval', default=10, type=int, help='how many epochs to wait before updating cluster centers [default: 10]')
    parser.add_argument('--tol', default=1e-3, type=float, help='tolerance for convergence [default: 1e-3]')
    parser.add_argument('--times', default=5, type=int, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    multi_times_label_pred_list = []
    for t in range(args.times):
        dataset = MMDataset(args.data_dir, concat_data=False); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
        data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_samples, shuffle=True)
        model = DUCMME(embed_dim=args.latent_dim, feature_dims=data_features, num_views=data_views, hidden_dims=[512, 512, 512], num_samples=data_samples, n_clusters=data_categories, alpha=1.0).to(device)
        ## === Stage 1: Uncertainty-Aware Reconstruction Pretraining ===
        print("\n=== Stage 1: Uncertainty-Aware Reconstruction Pretraining ===")
        print("Basic reconstruction training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
        for epoch in range(args.epoch_num[0]):
            model.train()
            losses = []
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                x = [x.to(device) for x in x]
                optimizer.zero_grad()
                loss = model.reconstruction_loss(x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
        # embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
        # kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
        # preds = kmeans.fit_predict(embedding)
        # acc_val = clustering_acc(label, preds)
        # nmi_val = normalized_mutual_info_score(label, preds)
        # asw_val = silhouette_score(embedding, preds)
        # ari_val = adjusted_rand_score(label, preds)
        # print(f"Pretraining completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
        # print("Uncertainty-aware reconstruction finetuning...")
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[1])
        # for epoch in range(args.epoch_num[1]):
        #     model.train()
        #     losses = []
        #     for batch_idx, (x, y, idx) in enumerate(dataloader):
        #         x = [x.to(device) for x in x]
        #         optimizer.zero_grad()
        #         loss = model.uncertainty_aware_reconstruction_loss(x)
        #         loss.backward()
        #         optimizer.step()
        #         losses.append(loss.item())
        #     print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
        # embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
        # kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
        # preds = kmeans.fit_predict(embedding)
        # acc_val = clustering_acc(label, preds)
        # nmi_val = normalized_mutual_info_score(label, preds)
        # asw_val = silhouette_score(embedding, preds)
        # ari_val = adjusted_rand_score(label, preds)
        # print(f"Finetuning completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")

        ## === Stage 2: Deep Embedding Clustering by DEC ===
        print("\n=== Stage 2: Deep Embedding Clustering ===")
        print("Cluster center initialization...")
        model.eval()
        embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
        kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
        preds = kmeans.fit_predict(embedding)
        acc_val = clustering_acc(label, preds)
        nmi_val = normalized_mutual_info_score(label, preds)
        asw_val = silhouette_score(embedding, preds)
        ari_val = adjusted_rand_score(label, preds)
        print(f"Cluster center initialization completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
        model.cluster_centers = kmeans.cluster_centers_ # shape: (n_clusters, embed_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[2])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        losses = []
        for epoch in range(args.epoch_num[1]):
            # Update target distribution periodically
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    q, embedding = model.forward_similarity_matrix_q(data)
                    p = model.target_distribution(q)
                y_pred = torch.argmax(q, dim=1).cpu().numpy()
                acc_val = clustering_acc(label, y_pred)
                nmi_val = normalized_mutual_info_score(label, y_pred)
                asw_val = 1 # asw_val = silhouette_score(embedding, y_pred)
                ari_val = adjusted_rand_score(label, y_pred)
                if epoch == 0:
                    delta_label = 1.0
                    y_pred_last = y_pred.copy()
                    print(f'[Epoch {epoch}] ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, Delta: {delta_label:.4f}')
                else:
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = y_pred.copy()
                    print(f'[Epoch {epoch}] ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, Delta: {delta_label:.4f}')
                    if delta_label < args.tol:
                        print('Converged, stopping training...')
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
            print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
        print(f'Final ACC: {clustering_acc(label, y_pred):.4f}')
        # 3\ Evaluation: evaluate the latent space H using clustering and classification
        model.eval()
        embedding = model.forward_embedding(data).detach().cpu().numpy() # get the embedding of the latent space H
        multi_times_embedding_list.append(embedding)
        multi_times_label_pred_list.append(y_pred)
    overall_performance_report(multi_times_embedding_list, multi_times_label_pred_list, label, args.output_dir)