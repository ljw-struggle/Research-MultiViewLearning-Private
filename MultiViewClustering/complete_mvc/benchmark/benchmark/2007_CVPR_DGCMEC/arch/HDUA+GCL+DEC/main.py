import os, random, argparse, numpy as np,scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dropout_edge
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from _utils import MMDataset, clustering_acc, purity_score

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
        # 3. Graph Convolutional Network by GCNConv
        self.graph_encoder_list = nn.ModuleList([GCNConv(embed_dim, embed_dim, add_self_loops=False), nn.BatchNorm1d(embed_dim), nn.ReLU()])
        self.linear_projection_encoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(),
                                                       nn.Linear(embed_dim, embed_dim))
        # 4. Deep Embedding Clustering by DEC
        self._cluster_centers = nn.Parameter(torch.Tensor(self.n_clusters, self.embed_dim))
        nn.init.xavier_uniform_(self._cluster_centers.data)
        
    def forward_embedding(self, x):
        encoded_output_list = [self.fusion_net_encoder[i](x[i]) for i in range(self.num_views)] # encode each view
        encoded_output_list = torch.stack(encoded_output_list, dim=1) # stack the encoded features from all views, (batch_size, num_views, embed_dim)
        encoded_output_list, _ = self.fusion_net_mha(encoded_output_list, encoded_output_list, encoded_output_list) # fuse the encoded features from all views by a multihead attention, (batch_size, num_views, embed_dim)
        encoded_output_list = encoded_output_list.contiguous().view(encoded_output_list.shape[0], -1) # flatten the encoded features, (batch_size, num_views*embed_dim)
        embedding = self.fusion_net_linear(encoded_output_list) # linear projection of the fused encoded features
        return embedding # get the embedding of the latent space H, (batch_size, embed_dim)
    
    def forward_graph_embedding(self, embedding, edge_index):
        embedding = self.graph_encoder_list[0](embedding, edge_index) # shape: [batch_size, embed_dim]
        embedding = self.graph_encoder_list[1](embedding) # shape: [batch_size, embed_dim]
        embedding = self.graph_encoder_list[2](embedding) # shape: [batch_size, embed_dim]
        embedding = self.linear_projection_encoder(embedding) # shape: [batch_size, embed_dim]
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
    
    @staticmethod
    def drop_feature(x, drop_prob):
        # x: list of tensors, each tensor shape: (num_samples, num_features)
        x = [x_i.clone() for x_i in x]
        for i in range(len(x)):
            drop_mask = torch.zeros((x[i].size(1)), device=x[i].device).uniform_(0, 1) >= drop_prob # shape: (num_features)
            x[i] = x[i] * drop_mask.unsqueeze(0) # shape: (num_samples, num_features)
        return x
    
    @staticmethod
    def drop_edge(edge_index, drop_prob):
        # edge_index.shape: (2, num_edges)
        edge_index_drop, _ = dropout_edge(edge_index, p=drop_prob)
        return edge_index_drop # shape: (2, num_edges * (1 - drop_prob))
    
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
    
    def loss_contrast_node(self, h_1, h_2, tau=0.4):
        # h_1: shape: (num_cells, num_hidden_2)
        # h_2: shape: (num_cells, num_hidden_2)
        h_1 = F.normalize(h_1)
        h_2 = F.normalize(h_2)
        intra_sim_h1_h1 = torch.exp(torch.mm(h_1, h_1.t()) / tau) # shape: (num_cells, num_cells)
        inter_sim_h1_h2 = torch.exp(torch.mm(h_1, h_2.t()) / tau) # shape: (num_cells, num_cells)
        intra_sim_h2_h2 = torch.exp(torch.mm(h_2, h_2.t()) / tau) # shape: (num_cells, num_cells)
        inter_sim_h2_h1 = torch.exp(torch.mm(h_2, h_1.t()) / tau) # shape: (num_cells, num_cells)
        l_1 = -torch.log(inter_sim_h1_h2.diag() / (intra_sim_h1_h1.sum(1) + inter_sim_h1_h2.sum(1) - intra_sim_h1_h1.diag())) # shape: (num_cells)
        l_2 = -torch.log(inter_sim_h2_h1.diag() / (intra_sim_h2_h2.sum(1) + inter_sim_h2_h1.sum(1) - intra_sim_h2_h2.diag())) # shape: (num_cells)
        loss = (l_1 + l_2) * 0.5 # shape: (num_cells)
        return loss.mean() # shape: (1)
    
    def loss_contrast_proto(self, cell_embeddings, cell_cluster_centers, cell_cluster_labels, tau=0.4):
        # cell_embeddings.shape: (num_cell, num_hidden)
        # cell_cluster_centers.shape: (num_protos, num_hidden)
        # cell_cluster_labels.shape: (num_cell)
        cell_embeddings = F.normalize(cell_embeddings, dim=1) # shape: (num_cell, num_hidden)
        cell_cluster_centers = F.normalize(cell_cluster_centers, dim=1) # shape: (num_protos, num_hidden)
        cell_cluster_labels = cell_cluster_labels.unsqueeze(1) # shape: (num_cell, 1)
        sim_cell_proto = torch.exp(torch.mm(cell_embeddings, cell_cluster_centers.t()) / tau) # shape: (num_cell, num_protos)
        sim_cell_corresponding_proto = torch.gather(sim_cell_proto, -1, cell_cluster_labels) # shape: (num_cell, 1)
        sim_cell_all_proto = torch.sum(sim_cell_proto, -1, keepdim=True) # shape: (num_cell, 1)
        sim_cell_corresponding_proto = torch.div(sim_cell_corresponding_proto, sim_cell_all_proto) # shape: (num_cell, 1)
        loss = -torch.log(sim_cell_corresponding_proto) # shape: (num_cell, 1)
        return loss.mean() # shape: (1)
    

if __name__ == "__main__":
    ## === Step 1: Environment & Reproducibility Setup ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_sc_multiomics/TEA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_sc_multiomics/DUCMME/TEA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', default=20, type=int, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', default=2000, type=int, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', default=[200, 100, 100], type=int, help='number of epochs to train [default: [200, 100, 100]]')
    parser.add_argument('--learning_rate', default=[5e-3, 1e-3, 1e-3], type=float, help='learning rate [default: [5e-3, 1e-3, 1e-3]]')
    parser.add_argument('--tol', default=1e-3, type=float, help='tolerance for convergence [default: 1e-3]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = MMDataset(args.data_dir, concat_data=False); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
    data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
    model = DUCMME(embed_dim=args.latent_dim, feature_dims=data_features, num_views=data_views, hidden_dims=[512, 512, 512], num_samples=data_samples, n_clusters=data_categories, alpha=1.0).to(device)
    ## === Stage 1: Uncertainty-Aware Reconstruction Pretraining ===
    print("\n=== Stage 1: Uncertainty-Aware Reconstruction Pretraining ===")
    print("Basic reconstruction training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
    for epoch in range(args.epoch_num[0]):
        model.train()
        optimizer.zero_grad()
        loss = model.reconstruction_loss(data)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch} completed. Loss: {loss.item():.4f}')
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    pur_val = purity_score(label, preds)
    print(f"Pretraining completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}")
    print("Uncertainty-aware reconstruction finetuning...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[1])
    for epoch in range(args.epoch_num[1]):
        model.train()
        optimizer.zero_grad()
        loss = model.uncertainty_aware_reconstruction_loss(data)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch} completed. Loss: {loss.item():.4f}')
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    pur_val = purity_score(label, preds)
    print(f"Finetuning completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, Purity: {pur_val:.4f}")
    
    ## === Stage 2: Graph Contrastive Learning ===
    print("\n=== Stage 2: Graph Contrastive Learning ===")
    print("Building graph from fusion embedding...")
    model.eval()
    embedding = model.forward_embedding(data).detach().cpu().numpy() # shape: (num_samples, embed_dim)
    kmeans = KMeans(n_clusters=2000, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    purity_val = purity_score(label, preds)
    print(f"Purity: {purity_val:.4f}")
    
    exit()
    adj_matrix = np.corrcoef(embedding) # shape: (num_samples, num_samples)
    adj_matrix = np.triu(adj_matrix, k=1) # shape: (num_samples, num_samples), upper triangle matrix
    adj_matrix = adj_matrix + adj_matrix.T # shape: (num_samples, num_samples), symmetric adjacency matrix
    threshold = np.percentile(adj_matrix.flatten(), 98) # top 2% edges considering symmetry
    adj_matrix = np.where(adj_matrix > threshold, 1, 0) # shape: (num_samples, num_samples), binary adjacency matrix
    sp_adj_matrix = sp.coo_matrix(adj_matrix)
    edge_index = torch.tensor(np.vstack((sp_adj_matrix.row, sp_adj_matrix.col)), dtype=torch.long).to(device) # shape: (2, num_edges)
    print(f'Graph constructed: {edge_index.shape[1]} edges from {data_samples} nodes')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[2])
    for epoch in range(100):
        model.eval()
        fusion_embedding = model.forward_embedding(data).detach()
        data_1 = model.drop_feature(data, 0.3)
        edge_index_1 = model.drop_edge(edge_index, 0.2)
        fusion_embedding_1 = model.forward_embedding(data_1).detach()
        data_2 = model.drop_feature(data, 0.4)
        edge_index_2 = model.drop_edge(edge_index, 0.4)
        fusion_embedding_2 = model.forward_embedding(data_2).detach()
        model.train()
        optimizer.zero_grad()
        # Node-level contrastive loss
        z_1 = model.forward_graph_embedding(fusion_embedding_1, edge_index_1)
        z_2 = model.forward_graph_embedding(fusion_embedding_2, edge_index_2)
        loss_node = model.loss_contrast_node(z_1, z_2, tau=0.4)
        # # Proto-level contrastive loss
        # graph_embedding = model.forward_graph_embedding(fusion_embedding, edge_index)
        # kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
        # cell_cluster_labels = kmeans.fit_predict(graph_embedding.detach().cpu().numpy())
        # cell_cluster_centers = kmeans.cluster_centers_
        # cell_cluster_labels = torch.tensor(cell_cluster_labels, dtype=torch.long).to(device)
        # cell_cluster_centers = torch.tensor(cell_cluster_centers, dtype=torch.float32).to(device)
        # loss_proto = model.loss_contrast_proto(graph_embedding, cell_cluster_centers, cell_cluster_labels, tau=0.4)
        loss_proto = torch.tensor(0.0, device=device)
        
        # Total loss
        loss = loss_node #+ 0.05 * loss_proto
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch} completed. Loss: {loss.item():.4f}, Loss_node: {loss_node.item():.4f}, Loss_proto: {loss_proto.item():.4f}')
    
    # Evaluate after graph contrastive learning
    model.eval()
    fusion_embedding = model.forward_embedding(data)
    graph_embedding = model.forward_graph_embedding(fusion_embedding, edge_index)
    embedding = graph_embedding.detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    print(f"Graph contrastive learning completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
    embedding = fusion_embedding.detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    print(f"Graph contrastive learning completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) # learning rate = 1e-3 * 0.9^(20/20) = 9e-4
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
        optimizer.zero_grad()
        loss = model.clustering_loss(data, p)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f'Epoch {epoch} completed. Loss: {loss.item():.4f}')
    print(f'Final ACC: {clustering_acc(label, y_pred):.4f}')
