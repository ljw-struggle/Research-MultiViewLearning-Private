import random, argparse, os, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from torch.distributions import NegativeBinomial
from pyro.distributions.zero_inflated import ZeroInflatedNegativeBinomial
from _utils import clustering_acc, overall_performance_report, MMDataset

class scMDC(nn.Module): # ZINB + DAE + DEC
    def __init__(self, embed_dim=10, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[512, 512, 512], n_clusters=10, alpha=1.0, noise_factor=0.0001, distribution='zinb'):
        super(scMDC, self).__init__()
        self.embed_dim = embed_dim; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims
        self.n_clusters = n_clusters; self.alpha = alpha; self.noise_factor = noise_factor; self.distribution = distribution
        self.encoder_list = nn.ModuleList([nn.Sequential(nn.Linear(feature_dims[i], hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(),
                                                         nn.Linear(hidden_dims[i], embed_dim)) for i in range(num_views)])
        self.fusion_net = nn.Linear(num_views*embed_dim, embed_dim)
        self.decoder_list = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU()) for i in range(num_views)])
        self.zinb_log_mean_list = nn.ModuleList([nn.Linear(hidden_dims[i], feature_dims[i]) for i in range(num_views)])
        self.zinb_dropout_list = nn.ModuleList([nn.Linear(hidden_dims[i], feature_dims[i]) for i in range(num_views)])
        self.zinb_log_theta_list = nn.ParameterList([nn.Parameter(torch.randn(feature_dims[i])) for i in range(num_views)])
        self._cluster_centers = nn.Parameter(torch.Tensor(n_clusters, embed_dim)) # shape: (n_clusters, embed_dim)
        nn.init.xavier_uniform_(self._cluster_centers.data)
        
    def forward(self, x, add_noise=True): # shape: [batch_size, input_dim]
        if add_noise:
            x = [x[i] + self.noise_factor * torch.randn_like(x[i]) for i in range(self.num_views)]
        encoded_output_list = [self.encoder_list[i](x[i]) for i in range(self.num_views)]
        embedding = self.fusion_net(torch.cat(encoded_output_list, dim=1))
        decoded_output_list = [self.decoder_list[i](embedding) for i in range(self.num_views)]
        zinb_log_mean_list = [self.zinb_log_mean_list[i](decoded_output_list[i]) for i in range(self.num_views)]
        zinb_dropout_list = [self.zinb_dropout_list[i](decoded_output_list[i]) for i in range(self.num_views)]
        zinb_log_theta_list = [self.zinb_log_theta_list[i] for i in range(self.num_views)]
        return decoded_output_list, embedding, zinb_log_mean_list, zinb_dropout_list, zinb_log_theta_list
    
    def forward_embedding(self, x, add_noise=False):
        _, embedding, _, _, _ = self.forward(x, add_noise=add_noise) # shape: [batch_size, embed_dim]
        return embedding # shape: [batch_size, embed_dim]
    
    def forward_similarity_matrix_q(self, x, add_noise=False): # shape: [batch_size, embed_dim]
        _, encoded, _, _, _ = self.forward(x, add_noise=add_noise) # shape: [batch_size, embed_dim]
        q = 1.0 / (1.0 + torch.sum((encoded.unsqueeze(1) - self._cluster_centers) ** 2, dim=2) / self.alpha) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters], alpha is a hyperparameter to control the sharpness of the t-distribution, more big, more sharp
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize q to sum to 1 across clusters
        return q, encoded # q can be regarded as the probability of the sample belonging to each cluster
    
    def pretraining_loss(self, x, add_noise=True):
        decoded_output_list, embedding, zinb_log_mean_list, zinb_dropout_list, zinb_log_theta_list = self.forward(x, add_noise=add_noise)
        zinb_mean_list = [zinb_log_mean_list[i].exp() for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        zinb_theta_list = [zinb_log_theta_list[i].exp() for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        nb_logits_list = [((zinb_mean_list[i] + 1e-4).log() - (zinb_theta_list[i] + 1e-4).log()) for i in range(self.num_views)] # element shape: [batch_size, feature_dim]
        if self.distribution == 'zinb':
            distribution = [ZeroInflatedNegativeBinomial(total_count=zinb_theta_list[i], logits=nb_logits_list[i], gate_logits = zinb_dropout_list[i], validate_args=False) for i in range(self.num_views)]
        elif self.distribution == 'nb':
            distribution = [NegativeBinomial(total_count=zinb_theta_list[i], logits=nb_logits_list[i], validate_args=False) for i in range(self.num_views)]
        reconstruction_loss = sum([distribution[i].log_prob(x[i]).sum(-1).mean() for i in range(self.num_views)]) # Maximum Likelihood Estimation (MLE)
        return - reconstruction_loss
    
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
    
    def dec_clustering_loss(self, x, p, add_noise=False):
        q, _ = self.forward_similarity_matrix_q(x, add_noise=add_noise) # shape: [batch_size, embed_dim]
        return F.kl_div(torch.log(q), p, reduction='batchmean', log_target=False) # KL.shape: [batch_size, n_clusters], for batchmean, result = sum(KL) / batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/DEC/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', type=float, default=10, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', type=int, nargs='+', default=[100, 20], help='number of epochs to train [default: 500]')
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[5e-3, 1e-3], help='learning rate [default: 1e-3]')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--times', type=int, default=5, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    for t in range(args.times):
        dataset = MMDataset(args.data_dir, concat_data=False); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
        data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
        model = scMDC(embed_dim=args.latent_dim, num_views=data_views, feature_dims=data_features, hidden_dims=[512, 512, 512], n_clusters=data_categories, alpha=1.0).to(device)
        # 1\ Pretraining: optimize the autoencoder to reconstruct the original data
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate[0], momentum=0.9)
        for epoch in range(args.epoch_num[0]):
            model.train()
            losses = []
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                x = [x[i].to(device) for i in range(len(x))]
                optimizer.zero_grad()
                loss = model.pretraining_loss(x, add_noise=True) # sum the losses from all views
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f'Pretraining Epoch: {epoch} Loss: {np.mean(losses):.4f}')
        # 2\ Fine-tuning: optimize the cluster centers to cluster the data
        print('Initializing cluster centers with KMeans...')
        model.eval()
        initial_embedding = model.forward_embedding(data, add_noise=False).detach().cpu().numpy()
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
                    q, encoded = model.forward_similarity_matrix_q(data, add_noise=False) # shape: [batch_size, n_clusters], encoded is the embedding of the data
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
                x = [x[i].to(device) for i in range(len(x))]; y = y.to(device); idx = idx.to(device)
                optimizer.zero_grad()
                loss = model.dec_clustering_loss(x, p[idx], add_noise=False) # shape: [batch_size, n_clusters]
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # scheduler.step()
        print('Final ACC:', clustering_acc(label, y_pred) if label is not None else 'N/A')
        # 3\ Evaluation: evaluate the latent space H using clustering and classification
        embedding = model.forward_embedding(data, add_noise=False).detach().cpu().numpy() # get the embedding of the latent space H
        multi_times_embedding_list.append(embedding)
    
    overall_performance_report(multi_times_embedding_list, None, label, args.output_dir) # Evaluate the latent space H using clustering and classification