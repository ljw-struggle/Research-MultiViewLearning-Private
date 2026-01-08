import random, argparse, os, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from _utils import clustering_acc, overall_performance_report, MinistDataset, MMDataset

class DEC(nn.Module):
    def __init__(self, embed_dim=10, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[500, 500, 2000], n_clusters=10, alpha=1.0):
        super(DEC, self).__init__()
        self.embed_dim = embed_dim; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims; self.n_clusters = n_clusters; self.alpha = alpha
        self.encoder = nn.Sequential(nn.Linear(sum(feature_dims), hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU(), 
                                     nn.Linear(hidden_dims[0], hidden_dims[1]), nn.BatchNorm1d(hidden_dims[1]), nn.ReLU(), 
                                     nn.Linear(hidden_dims[1], hidden_dims[2]), nn.BatchNorm1d(hidden_dims[2]), nn.ReLU(), 
                                     nn.Linear(hidden_dims[2], embed_dim)) # encode each view
        self.decoder = nn.Sequential(nn.Linear(embed_dim, hidden_dims[2]), nn.BatchNorm1d(hidden_dims[2]), nn.ReLU(), 
                                     nn.Linear(hidden_dims[2], hidden_dims[1]), nn.BatchNorm1d(hidden_dims[1]), nn.ReLU(), 
                                     nn.Linear(hidden_dims[1], hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU(), 
                                     nn.Linear(hidden_dims[0], sum(feature_dims))) # decode each view
        self._cluster_centers = nn.Parameter(torch.Tensor(n_clusters, embed_dim)) # shape: (n_clusters, embed_dim)
        nn.init.xavier_uniform_(self._cluster_centers.data)
        
    def forward_autoencoder(self, x): # shape: [batch_size, input_dim]
        encoded = self.encoder(x) # shape: [batch_size, embed_dim]
        decoded = self.decoder(encoded) # shape: [batch_size, input_dim]
        return encoded, decoded
    
    def forward_embedding(self, x):
        encoded = self.encoder(x) # shape: [batch_size, embed_dim]
        return encoded # shape: [batch_size, embed_dim]
    
    def forward_similarity_matrix_q(self, x): # shape: [batch_size, embed_dim]
        encoded = self.encoder(x) # shape: [batch_size, embed_dim]
        q = 1.0 / (1.0 + torch.sum((encoded.unsqueeze(1) - self._cluster_centers) ** 2, dim=2) / self.alpha) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters], alpha is a hyperparameter to control the sharpness of the t-distribution, more big, more sharp
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize q to sum to 1 across clusters
        return q, encoded # q can be regarded as the probability of the sample belonging to each cluster
    
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
        p = weight / torch.sum(weight, dim=1, keepdim=True) # Normalize p to sum to 1 across clusters
        return p.detach() # shape: [batch_size, n_clusters]
    
    def pretraining_loss(self, x):
        _, decoded = self.forward_autoencoder(x) # shape: [batch_size, embed_dim]
        return F.mse_loss(decoded, x, reduction='mean')
    
    def dec_clustering_loss(self, x, p):
        q, _ = self.forward_similarity_matrix_q(x) # shape: [batch_size, embed_dim]
        return F.kl_div(torch.log(q), p, reduction='batchmean', log_target=False) # KL.shape: [batch_size, n_clusters], for batchmean, result = sum(KL) / batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_MOGONET/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_MOGONET/DEC/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', type=float, default=10, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', type=int, nargs='+', default=[50, 20], help='number of epochs to train [default: 500]')
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[1.0, 1e-2], help='learning rate [default: 1e-3]')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--times', type=int, default=5, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    # random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed(42) # Set random seed for reproducibility
    
    multi_times_embedding_list = []
    for t in range(args.times):
        dataset = MMDataset(args.data_dir, concat_data=True); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
        data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
        model = DEC(embed_dim=args.latent_dim, num_views=data_views, feature_dims=data_features, hidden_dims=[500, 500, 2000], n_clusters=data_categories, alpha=1.0).to(device)
        
        # 1\ Pretraining: optimize the autoencoder to reconstruct the original data
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate[0], momentum=0.9)
        # for epoch in range(args.epoch_num[0]):
        #     model.train()
        #     losses = []
        #     for batch_idx, (x, y, idx) in enumerate(dataloader):
        #         x = x.to(device)
        #         optimizer.zero_grad()
        #         loss = model.pretraining_loss(x) # sum the losses from all views
        #         loss.backward()
        #         optimizer.step()
        #         losses.append(loss.item())
        #     print(f'Pretraining Epoch: {epoch} Loss: {np.mean(losses):.4f}')
            
        # 1\ Pretraining: optimize the autoencoder to reconstruct the original data
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate[0], momentum=0.9)
        model.train()
        for epoch in range(args.epoch_num[0]):
            perm = torch.randperm(data.shape[0])  # Shuffle the data
            losses = []
            for i in range(0, data.size(0), args.batch_size):
                optimizer.zero_grad()
                loss = model.pretraining_loss(data[perm[i:i+args.batch_size]])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(losses):.4f}')
        
        # 2\ Fine-tuning: optimize the cluster centers to cluster the data
        print('Initializing cluster centers with KMeans...')
        model.eval()
        initial_embedding = model.forward_embedding(data).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=model.n_clusters, n_init=20); 
        y_pred = kmeans.fit_predict(initial_embedding)
        acc_val = clustering_acc(label, y_pred)
        nmi_val = normalized_mutual_info_score(label, y_pred)
        asw_val = 1 # asw_val = silhouette_score(initial_embedding, y_pred) # this metric is very slow, so we use 1 as a placeholder
        ari_val = adjusted_rand_score(label, y_pred)
        print(f"DEC Evaluation: Initial clustering completed; ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
        model.cluster_centers = kmeans.cluster_centers_ # shape: (n_clusters, embed_dim)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[1])
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate[1], momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) # learning rate = 1e-3 * 0.9^(10/10) = 9e-4
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        losses = []
        for epoch in range(args.epoch_num[1]):
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    q, encoded = model.forward_similarity_matrix_q(data) # shape: [batch_size, n_clusters], encoded is the embedding of the data
                    p = model.target_distribution(q) # shape: [batch_size, n_clusters], update the target distribution p
                y_pred = torch.argmax(q, dim=1).cpu().numpy()
                acc_val = clustering_acc(label, y_pred)
                nmi_val = normalized_mutual_info_score(label, y_pred)
                asw_val = 1 # asw_val = silhouette_score(encoded.detach().cpu().numpy(), y_pred) # this metric is very slow, so we use 1 as a placeholder
                ari_val = adjusted_rand_score(label, y_pred)
                if epoch == 0:
                    delta_label = 1.0
                    y_pred_last = y_pred.copy()
                    print(f'[Epoch {epoch}] loss: NaN, ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, delta: {delta_label:.4f}')
                else:
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                    y_pred_last = y_pred.copy()
                    print(f'[Epoch {epoch}] loss: {np.mean(losses):.4f}, ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, delta: {delta_label:.4f}')
                    if delta_label < 1e-3:
                        print('Converged, stopping training...'); break
            model.train()
            losses = []
            for batch_idx, (x, y, idx) in enumerate(dataloader): # shape: [batch_size, 784]
                x = x.to(device); y = y.to(device); idx = idx.to(device)
                optimizer.zero_grad()
                loss = model.dec_clustering_loss(x, p[idx]) # shape: [batch_size, n_clusters]
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # scheduler.step()
        print('Final ACC:', clustering_acc(label, y_pred) if label is not None else 'N/A')
        # 3\ Evaluation: evaluate the latent space H using clustering and classification
        H = model.forward_embedding(data).detach().cpu().numpy() # get the embedding of the latent space H
        multi_times_embedding_list.append(H)
    
    overall_performance_report(multi_times_embedding_list, label, args.output_dir) # Evaluate the latent space H using clustering and classification
