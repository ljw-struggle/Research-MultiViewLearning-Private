import math, os, random, argparse, itertools, numpy as np, scipy.io as sio
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from _utils import MMDataset, clustering_acc
from torch_geometric.utils import dropout_adj 
from scipy import sparse

class DHGNN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, num_views=3, num_samples=10000):
        super(DHGNN, self).__init__()
        self.in_ft = in_ft; self.out_ft = out_ft; self.bias = bias; self.num_views = num_views; self.num_samples = num_samples
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.WE = nn.Parameter(torch.Tensor(num_views * num_samples, num_views * num_samples)) # hyperedge weight matrix (M X M), M = num_views * num_samples
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # initialize the weights of the WE matrix to be the identity matrix
        self.WE.data.copy_(torch.eye(self.num_views * self.num_samples))

    def forward(self, x: torch.Tensor, H: torch.Tensor, HT: torch.Tensor):
        x = x.matmul(self.weight) # node feature transformation
        if self.bias is not None:
            x = x + self.bias
        G = H.matmul(self.WE).matmul(HT) # shape: N X N
        output = torch.matmul(G, x) # hypergraph convolution
        return output, G

class DCLMEC(nn.Module):
    def __init__(self, embed_dim=200, num_samples=10000, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[512, 512, 512], n_clusters=10, alpha=1.0):
        super(DCLMEC, self).__init__()
        self.embed_dim = embed_dim; self.num_samples = num_samples; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims; self.n_clusters = n_clusters; self.alpha = alpha
        # 1. Multi-view Feature Extraction by Fusion-Net
        self.fusion_net_encoder = nn.ModuleList([nn.Sequential(nn.Linear(feature_dims[i], hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(), nn.Linear(hidden_dims[i], embed_dim)) for i in range(num_views)]) # encode each view
        self.fusion_net_linear = nn.Linear(num_views * embed_dim, embed_dim) # linear projection of the fused encoded features
        # 2. Dynamic Hypergraph Convolution by HGNN_conv
        self.DHGNN = DHGNN(embed_dim, embed_dim, num_views=num_views, num_samples=num_samples) # dynamic hypergraph convolution
        # 2. Uncertainty-Aware Reconstruction by Reconstruction-Net and Uncertainty-Net
        self.reconstruct_net_list = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(), nn.Linear(hidden_dims[i], feature_dims[i])) for i in range(num_views)]) # reconstruct each view
        self.uncertainty_net_list = nn.ModuleList([nn.Sequential(nn.Linear(self.embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(), nn.Linear(hidden_dims[i], feature_dims[i])) for i in range(num_views)]) # predict uncertainty for each view
        # 3. Deep Embedding Clustering by DEC
        self._cluster_centers = nn.Parameter(torch.Tensor(self.n_clusters, self.embed_dim))
        nn.init.xavier_uniform_(self._cluster_centers.data)
        
    def forward_embedding(self, x, H, HT):
        encoded_output_list = [self.fusion_net_encoder[i](x[i]) for i in range(self.num_views)] # encode each view
        encoded_output_list = torch.stack(encoded_output_list, dim=1) # stack the encoded features from all views, (batch_size, num_views, embed_dim)
        encoded_output_list = encoded_output_list.contiguous().view(encoded_output_list.shape[0], -1) # flatten the encoded features, (batch_size, num_views*embed_dim)
        embedding = self.fusion_net_linear(encoded_output_list) # linear projection of the fused encoded features
        embedding, G = self.DHGNN(embedding, H, HT) # dynamic hypergraph convolution
        return embedding, G # get the embedding of the latent space H, (batch_size, embed_dim) and the hypergraph structure G, (batch_size, batch_size)

    def forward_uncertainty_aware_reconstruction(self, x, H, HT):
        embedding, _ = self.forward_embedding(x, H, HT) # shape: [batch_size, embed_dim]
        reconstructions = [self.reconstruct_net_list[i](embedding) for i in range(self.num_views)] # reconstruct each view
        uncertainties = [self.uncertainty_net_list[i](embedding) for i in range(self.num_views)] # predict uncertainty for each view
        return reconstructions, uncertainties
        
    def forward_similarity_matrix_q(self, x, H, HT): # calculate the similarity matrix q using t-distribution
        embedding, _ = self.forward_embedding(x, H, HT) # shape: [batch_size, embed_dim]
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
    
    def reconstruction_loss(self, x, H, HT):
        x_rec, _ = self.forward_uncertainty_aware_reconstruction(x, H, HT) # reconstruct each view and predict uncertainty
        return sum([F.mse_loss(x_rec[v], x[v], reduction='mean') for v in range(self.num_views)]) # sum the losses from all views
    
    def uncertainty_aware_reconstruction_loss(self, x, H, HT):
        x_rec, log_sigma_2 = self.forward_uncertainty_aware_reconstruction(x, H, HT) # reconstruct each view and predict uncertainty
        # Clip log_sigma_2 to prevent numerical instability (exp(-log_sigma_2) overflow/underflow)
        # Clamp to reasonable range: -10 to 10, which corresponds to sigma^2 from exp(-10) to exp(10)
        log_sigma_2 = [torch.clamp(log_sigma_2[v], min=-10.0, max=10.0) for v in range(self.num_views)] # shape: [num_views, batch_size, feature_dim] for numerically stable computation
        return sum([0.5 * torch.mean((x_rec[v] - x[v])**2 * torch.exp(-log_sigma_2[v]) + log_sigma_2[v]) for v in range(self.num_views)]) # uncertainty is equal to log_sigma_2
    
    def clustering_loss(self, x, p, H, HT):
        q, _ = self.forward_similarity_matrix_q(x, H, HT) # shape: [batch_size, n_clusters]
        return F.kl_div(q.log(), p, reduction='batchmean') # shape: ()
        
    @staticmethod
    def drop_feature(x, drop_prob):
        # x: list of tensors for multi-view data, each x[i].shape: (num_samples, num_features_i)
        for i in range(len(x)):
            keep_mask = torch.rand(x[i].size(1), device=x[i].device) >= drop_prob # shape: (num_features_i)
            x[i] = x[i] * keep_mask.unsqueeze(0) # shape: (num_samples, num_features_i)
        return x
    
    @staticmethod
    def drop_hyperedge(hyperedge_mat, drop_prob):
        # hyperedge_mat: scipy sparse matrix, shape (num_samples, num_hyperedges)
        # Drop columns (hyperedges) from hyperedge_mat and return a copy
        assert sparse.issparse(hyperedge_mat), 'hyperedge_mat must be a sparse matrix'
        num_hyperedges = hyperedge_mat.shape[1]
        keep_mask = np.random.rand(num_hyperedges) >= drop_prob
        hyperedge_mat_drop = hyperedge_mat.multiply(keep_mask) # shape: (num_samples, num_hyperedges), format: csc
        print(hyperedge_mat_drop.shape)
        return hyperedge_mat_drop

    @staticmethod
    def loss_contrast_node(h_1, h_2, tau=0.4):
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

    @staticmethod
    def loss_contrast_proto(cell_embeddings, cell_cluster_centers, cell_cluster_labels, tau=0.4):
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
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/DCLMEC/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', default=20, type=int, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', default=2000, type=int, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', default=[200, 100, 100], type=int, help='number of epochs to train [default: [200, 100, 100]]')
    parser.add_argument('--learning_rate', default=[5e-3, 1e-3, 1e-3], type=float, help='learning rate [default: [5e-3, 1e-3, 1e-3]]')
    parser.add_argument('--log_interval', default=10, type=int, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--update_interval', default=10, type=int, help='how many epochs to wait before updating cluster centers [default: 10]')
    parser.add_argument('--tol', default=1e-3, type=float, help='tolerance for convergence [default: 1e-3]')
    parser.add_argument('--times', default=5, type=int, help='number of times to run the experiment [default: 30]')
    parser.add_argument('--drop_edge_rate_1', default=0.2, type=float, help='drop edge rate for view 1 [default: 0.2]')
    parser.add_argument('--drop_edge_rate_2', default=0.4, type=float, help='drop edge rate for view 2 [default: 0.4]')
    parser.add_argument('--drop_feature_rate_1', default=0.3, type=float, help='drop feature rate for view 1 [default: 0.3]')
    parser.add_argument('--drop_feature_rate_2', default=0.4, type=float, help='drop feature rate for view 2 [default: 0.4]')
    parser.add_argument('--tau', default=0.4, type=float, help='temperature coefficient [default: 0.4]')
    parser.add_argument('--alpha_contrast', default=0.1, type=float, help='weight for contrastive learning [default: 0.1]')
    parser.add_argument('--num_protos', default=None, type=int, help='number of prototypes for prototype contrast [default: None, use n_clusters]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    multi_times_label_pred_list = []
    dataset = MMDataset(args.data_dir, concat_data=False)
    data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
    data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy(); hyperedge_mat = dataset.get_hyperedge_matrix(k_neighbors=30)
    model = DCLMEC(embed_dim=args.latent_dim, feature_dims=data_features, num_views=data_views, hidden_dims=[512, 512, 512], num_samples=data_samples, n_clusters=data_categories, alpha=1.0).to(device)
    num_protos = args.num_protos if args.num_protos else data_categories
    ## === Stage 1: Uncertainty-Aware Reconstruction Pretraining ===
    print("\n=== Stage 1: Uncertainty-Aware Reconstruction Pretraining ===")
    print("Basic reconstruction training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
    for epoch in range(args.epoch_num[0]):
        model.train()
        losses = []
        H, _, HT = dataset.get_H_WE_HT(hyperedge_mat, dynamic_weight=True)
        H = torch.from_numpy(H.toarray()).float().to(device)
        HT = torch.from_numpy(HT.toarray()).float().to(device)
        optimizer.zero_grad()
        loss = model.reconstruction_loss(data, H, HT)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
    model.eval()
    H, _, HT = dataset.get_H_WE_HT(hyperedge_mat, dynamic_weight=True)
    H = torch.from_numpy(H.toarray()).float().to(device)
    HT = torch.from_numpy(HT.toarray()).float().to(device)
    embedding, _ = model.forward_embedding(data, H, HT)
    embedding = embedding.detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    print(f"Pretraining completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
    
    print("Uncertainty-aware reconstruction finetuning with contrastive learning...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[1])
    for epoch in range(args.epoch_num[1]):
        model.train()
        losses = []
        # Update prototypes periodically (every epoch)
        if epoch % 1 == 0:
            model.eval()
            with torch.no_grad():
                H, _, HT = dataset.get_H_WE_HT(hyperedge_mat, dynamic_weight=True)
                H = torch.from_numpy(H.toarray()).float().to(device)
                HT = torch.from_numpy(HT.toarray()).float().to(device)
                cell_embeddings, _ = model.forward_embedding(data, H, HT)
                cell_embeddings_numpy = cell_embeddings.cpu().detach().numpy()
                kmeans = KMeans(n_clusters=num_protos, n_init=20, random_state=0).fit(cell_embeddings_numpy)
                cell_cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long).to(device)
                cell_cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        loss_recon = model.uncertainty_aware_reconstruction_loss(data, H, HT)
        
        # Contrastive learning: node-level and prototype-level
        # Create two augmented views by dropping hyperedges (columns) from hyperedge_mat
        x_1 = model.drop_feature(data, args.drop_feature_rate_1)
        hyperedge_mat_1 = model.drop_hyperedge(hyperedge_mat, args.drop_edge_rate_1)
        H_1, _, HT_1 = dataset.get_H_WE_HT(hyperedge_mat_1, dynamic_weight=True)
        H_1 = torch.from_numpy(H_1.toarray()).float().to(device)
        HT_1 = torch.from_numpy(HT_1.toarray()).float().to(device)
        x_2 = model.drop_feature(data, args.drop_feature_rate_2)
        hyperedge_mat_2 = model.drop_hyperedge(hyperedge_mat, args.drop_edge_rate_2)
        H_2, _, HT_2 = dataset.get_H_WE_HT(hyperedge_mat_2, dynamic_weight=True)
        H_2 = torch.from_numpy(H_2.toarray()).float().to(device)
        HT_2 = torch.from_numpy(HT_2.toarray()).float().to(device)
        # Get embeddings for two views
        z_1, _ = model.forward_embedding(x_1, H_1, HT_1)
        z_2, _ = model.forward_embedding(x_2, H_2, HT_2)
        # Node-level contrastive loss
        loss_node = model.loss_contrast_node(z_1, z_2, args.tau)
        # Prototype-level contrastive loss
        H, _, HT = dataset.get_H_WE_HT(hyperedge_mat, dynamic_weight=True)
        H = torch.from_numpy(H.toarray()).float().to(device)
        HT = torch.from_numpy(HT.toarray()).float().to(device)
        cell_embeddings, _ = model.forward_embedding(data, H, HT)
        loss_proto = model.loss_contrast_proto(cell_embeddings, cell_cluster_centers, cell_cluster_labels, args.tau)
        loss = loss_recon + args.alpha_contrast * (loss_node + loss_proto)
        
        loss = loss_recon
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}, Loss: {loss.item():.4f}')
    model.eval()
    H, _, HT = dataset.get_H_WE_HT(hyperedge_mat, dynamic_weight=True)
    H = torch.from_numpy(H.toarray()).float().to(device)
    HT = torch.from_numpy(HT.toarray()).float().to(device)
    embedding, _ = model.forward_embedding(data, H, HT)
    embedding = embedding.detach().cpu().numpy() # shape: [num_samples, embed_dim]
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = silhouette_score(embedding, preds)
    ari_val = adjusted_rand_score(label, preds)
    print(f"Finetuning completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")

    ## === Stage 2: Deep Embedding Clustering by DEC ===
    print("\n=== Stage 2: Deep Embedding Clustering ===")
    print("Cluster center initialization...")
    model.eval()
    H, _, HT = dataset.get_H_WE_HT(hyperedge_mat, dynamic_weight=True)
    H = torch.from_numpy(H.toarray()).float().to(device)
    HT = torch.from_numpy(HT.toarray()).float().to(device)
    embedding, _ = model.forward_embedding(data, H, HT)
    embedding = embedding.detach().cpu().numpy() # shape: [num_samples, embed_dim]
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
            H, _, HT = dataset.get_H_WE_HT(hyperedge_mat, dynamic_weight=True)
            H = torch.from_numpy(H.toarray()).float().to(device)
            HT = torch.from_numpy(HT.toarray()).float().to(device)
            with torch.no_grad():
                q, embedding = model.forward_similarity_matrix_q(data, H, HT)
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
        loss = model.clustering_loss(data, p, H, HT)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        scheduler.step()
        print(f'Epoch {epoch} completed. Average Loss: {np.mean(losses):.4f}')
    print(f'Final ACC: {clustering_acc(label, y_pred):.4f}')
