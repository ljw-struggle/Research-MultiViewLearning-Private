import random, argparse, os, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class MinistDataset(Dataset):
    def __init__(self, data_dir):
        self.dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]))
        self.X, self.Y = next(iter(DataLoader(self.dataset, batch_size=len(self.dataset)))); # Dataloader object is iterable object, but not a iterator. So we need convert it to a iterator.
        self.data_views = 1; self.data_samples = len(self.X); self.data_features = [784]; self.categories = np.unique(self.Y).shape[0] # number of categories, number of views, number of samples, number of features in each view

    def __getitem__(self, index):
        x = self.X[index]; y = self.Y[index] # convert to tensor
        return x, y, index

    def __len__(self):
        return len(self.Y)
    
    def get_data_info(self):
        return self.data_views, self.data_samples, self.data_features, self.categories

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] # accuracy

class DEC(nn.Module):
    def __init__(self, embed_dim=10, feature_dims=[784], num_views=1, hidden_dims=[500, 500, 2000], n_clusters=10, alpha=1.0):
        super(DEC, self).__init__()
        self.embed_dim = embed_dim; self.feature_dims = feature_dims; self.num_views = num_views; self.hidden_dims = hidden_dims; self.n_clusters = n_clusters; self.alpha = alpha
        dims = [sum(feature_dims)] + hidden_dims + [embed_dim]
        encoder_layers = [[nn.Linear(dims[i], dims[i+1]), nn.ReLU()] for i in range(len(dims) - 2)]
        encoder_layers = [layer for sublist in encoder_layers for layer in sublist]  # Flatten the list
        encoder_layers.append(nn.Linear(dims[-2], dims[-1]))  # Last layer without ReLU
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = [[nn.Linear(dims[i], dims[i-1]), nn.ReLU()] for i in range(len(dims) - 1, 1, -1)]
        decoder_layers = [layer for sublist in decoder_layers for layer in sublist]  # Flatten the list
        decoder_layers.append(nn.Linear(dims[1], dims[0]))  # Last layer without ReLU
        self.decoder = nn.Sequential(*decoder_layers)
        
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
        p = weight / torch.sum(weight, dim=1, keepdim=True)  # Normalize p to sum to 1 across clusters
        return p.detach() # shape: [batch_size, n_clusters]
    
    def pretraining_loss(self, x):
        _, decoded = self.forward_autoencoder(x) # shape: [batch_size, embed_dim]
        return F.mse_loss(decoded, x, reduction='mean')
    
    def dec_clustering_loss(self, x, p):
        q, _ = self.forward_similarity_matrix_q(x) # shape: [batch_size, embed_dim]
        return F.kl_div(torch.log(q), p, reduction='batchmean', log_target=False) # KL.shape: [batch_size, n_clusters], for batchmean, result = sum(KL) / batch_size
    
# def train_method_1(lr_ae=1.0, lr_dec=0.01, batch_size=256, epochs=300, maxiter=20000, update_interval=140, tol=1e-3, device='cuda'):
#     dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]))
#     data, label = next(iter(DataLoader(dataset, batch_size=len(dataset)))); data = data.to(device); label = label.numpy()
#     print(f'Data shape: {data.shape}, Label shape: {label.shape}')
#     model = DEC(embed_dim=10, feature_dims=[784], num_views=1, hidden_dims=[500, 500, 2000], n_clusters=10, alpha=1.0).to(device)
    
#     # 1\ Pretraining: optimize the autoencoder to reconstruct the original data
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr_ae, momentum=0.9)
#     model.train()
#     for epoch in range(epochs):
#         perm = torch.randperm(data.shape[0])  # Shuffle the data
#         losses = []
#         for i in range(0, data.size(0), batch_size):
#             optimizer.zero_grad()
#             x_batch = data[perm[i:i+batch_size]]
#             loss = model.pretraining_loss(x_batch)
#             loss.backward()
#             optimizer.step()
#             losses.append(loss.item())
#         print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(losses):.4f}')
        
#     # 2\ Fine-tuning: optimize the cluster centers to cluster the data
#     print('Initializing cluster centers with KMeans...')
#     model.eval()
#     initial_embedding = model.forward_embedding(data).detach().cpu().numpy()
#     kmeans = KMeans(n_clusters=model.n_clusters, n_init=20); 
#     preds = kmeans.fit_predict(initial_embedding)
#     acc_val = clustering_acc(label, preds)
#     nmi_val = normalized_mutual_info_score(label, preds)
#     asw_val = 1 # asw_val = silhouette_score(initial_embedding, preds) # this metric is very slow, so we use 1 as a placeholder
#     ari_val = adjusted_rand_score(label, preds)
#     print(f"DEC Evaluation: Initial clustering completed; ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
#     model.cluster_centers = kmeans.cluster_centers_
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr_dec, momentum=0.9)
#     losses = []
#     for ite in range(int(maxiter)):
#         if ite % update_interval == 0:
#             model.eval()
#             with torch.no_grad():
#                 q, encoded = model.forward_similarity_matrix_q(data)
#                 p = model.target_distribution(q).detach()
#             y_pred = torch.argmax(q, dim=1).cpu().numpy()
#             acc_val = clustering_acc(label, y_pred) if label is not None else None
#             nmi_val = normalized_mutual_info_score(label, y_pred) if label is not None else None
#             asw_val = 1 # asw_val = silhouette_score(encoded.detach().cpu().numpy(), y_pred) if label is not None else None # this metric is very slow, so we use 1 as a placeholder
#             ari_val = adjusted_rand_score(label, y_pred) if label is not None else None
#             if ite == 0:
#                 delta_label = 1.0
#                 y_pred_last = y_pred.copy()
#                 print(f'[Iter {ite}] loss: NaN, ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, delta: {delta_label:.4f}')
#             else:
#                 delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
#                 y_pred_last = y_pred.copy()
#                 print(f'[Iter {ite}] loss: {np.mean(losses):.4f}, ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}, delta: {delta_label:.4f}')
#                 if delta_label < tol:
#                     print('Converged, stopping training...'); break
#         model.train()
#         losses = []
#         optimizer.zero_grad()
#         idx = np.arange(data.shape[0])[(ite * batch_size) % len(data):(ite * batch_size + batch_size) % len(data) if (ite * batch_size + batch_size) < len(data) else len(data)]
#         loss = model.dec_clustering_loss(data[idx], p[idx])
#         loss.backward()
#         optimizer.step()
#         losses.append(loss.item())
#     print('Final acc:', clustering_acc(label, y_pred) if label is not None else 'N/A')
    
def train_method_2(lr_ae=1.0, lr_dec=0.01, batch_size=256, epochs_ae=300, epochs_dec=100, tol=1e-3, device='cuda'):
    dataset = MinistDataset(); data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
    data = dataset.X.clone().to(device); label = dataset.Y.clone().numpy()
    print(f'Data shape: {data.shape}, Label shape: {label.shape}')
    model = DEC(embed_dim=10, feature_dims=[784], num_views=1, hidden_dims=[500, 500, 2000], n_clusters=10, alpha=1.0).to(device)
    
    # 1\ Pretraining: optimize the autoencoder to reconstruct the original data
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # optimizer_pre = torch.optim.SGD(model.parameters(), lr=lr_ae, momentum=0.9)
    # for epoch_pre in range(epochs_ae):
    #     model.train()
    #     losses = []
    #     for batch_idx, (x, y, idx) in enumerate(dataloader):
    #         x = x.to(device)
    #         optimizer_pre.zero_grad()
    #         loss_pre = model.pretraining_loss(x) # sum the losses from all views
    #         loss_pre.backward()
    #         optimizer_pre.step()
    #         losses.append(loss_pre.item())
    #     print(f'Pretraining Epoch: {epoch_pre} Loss: {np.mean(losses):.4f}')
        
    # 1\ Pretraining: optimize the autoencoder to reconstruct the original data
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_ae, momentum=0.9)
    model.train()
    for epoch in range(epochs_ae):
        perm = torch.randperm(data.shape[0])  # Shuffle the data
        losses = []
        for i in range(0, data.size(0), batch_size):
            optimizer.zero_grad()
            loss = model.pretraining_loss(data[perm[i:i+batch_size]])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(losses):.4f}')
    
    # 2\ Fine-tuning: optimize the cluster centers to cluster the data
    print('Initializing cluster centers with KMeans...')
    model.eval()
    initial_embedding = model.forward_embedding(data).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=model.n_clusters, n_init=20); 
    preds = kmeans.fit_predict(initial_embedding)
    acc_val = clustering_acc(label, preds)
    nmi_val = normalized_mutual_info_score(label, preds)
    asw_val = 1 # asw_val = silhouette_score(initial_embedding, preds) # this metric is very slow, so we use 1 as a placeholder
    ari_val = adjusted_rand_score(label, preds)
    print(f"DEC Evaluation: Initial clustering completed; ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
    model.cluster_centers = kmeans.cluster_centers_ # shape: (n_clusters, embed_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_dec, momentum=0.9)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    losses = []
    for epoch in range(epochs_dec):
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
                if delta_label < tol:
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
    print('Final ACC:', clustering_acc(label, y_pred) if label is not None else 'N/A')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_MOGONET/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_MOGONET/DEC/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', type=float, default=10, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', type=int, nargs='+', default=[50, 40], help='number of epochs to train [default: 500]')
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[1.0, 1e-2], help='learning rate [default: 1e-3]')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--times', type=int, default=1, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    # random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed(42) # Set random seed for reproducibility
    
    # train_method_1(lr_ae=args.learning_rate[0], lr_dec=args.learning_rate[1], batch_size=args.batch_size, epochs=args.epoch_num[0], maxiter=20000, update_interval=140, tol=1e-3, device=device)
    train_method_2(lr_ae=args.learning_rate[0], lr_dec=args.learning_rate[1], batch_size=args.batch_size, epochs_ae=args.epoch_num[0], epochs_dec=args.epoch_num[1], tol=1e-3, device=device)
