import random, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from _evaluate import evaluate
from _dataset import load_data

class AutoEncoder(nn.Module):
    def __init__(self, dims):
        super(AutoEncoder, self).__init__()
        encoder_layers = [[nn.Linear(dims[i], dims[i+1]), nn.ReLU()] for i in range(len(dims) - 2)]
        encoder_layers = [layer for sublist in encoder_layers for layer in sublist]  # Flatten the list
        encoder_layers.append(nn.Linear(dims[-2], dims[-1]))  # Last layer without ReLU
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = [[nn.Linear(dims[i], dims[i-1]), nn.ReLU()] for i in range(len(dims) - 1, 1, -1)]
        decoder_layers = [layer for sublist in decoder_layers for layer in sublist]  # Flatten the list
        decoder_layers.append(nn.Linear(dims[1], dims[0]))  # Last layer without ReLU
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters; self.hidden_dim = hidden_dim; self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize q to sum to 1 across clusters
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, dim=0) # shape: [batch_size, n_clusters]
        p = weight / torch.sum(weight, dim=1, keepdim=True)  # Normalize p to sum to 1 across clusters
        return p

class IDEC(nn.Module):
    def __init__(self, dims, n_clusters, alpha=1.0):
        super(IDEC, self).__init__()
        self.auto_encoder = AutoEncoder(dims)
        self.encoder = self.auto_encoder.encoder
        self.clustering_layer = ClusteringLayer(n_clusters, dims[-1], alpha)

    def forward(self, x):
        return self.clustering_layer(self.encoder(x)), self.auto_encoder(x)

    def extract_features(self, x):
        return self.encoder(x)
    
    def set_clustering_centers(self, centers):
        centers = torch.tensor(centers, dtype=torch.float32, device=self.clustering_layer.cluster_centers.device)
        self.clustering_layer.cluster_centers.data.copy_(centers)
    
    def extract_cluster_centers(self):
        return self.clustering_layer.cluster_centers.data.cpu().numpy()

def pretrain_autoencoder(model, data, lr=1.0, batch_size=256, epochs=300):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(data.shape[0])  # Shuffle the data
        losses = []
        for i in range(0, data.size(0), batch_size):
            optimizer.zero_grad()
            x_batch = data[perm[i:i+batch_size]]
            x_recon = model(x_batch)
            loss = criterion(x_batch, x_recon)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(losses):.4f}')

def train_dec(model, data, y_true=None, lr=0.01, batch_size=256, maxiter=20000, update_interval=140, tol=1e-3, lambda_mse=0.1):
    print('Initializing cluster centers with KMeans...')
    model.eval()
    features = model.extract_features(data).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=model.clustering_layer.n_clusters, n_init=20)
    kmeans.fit_predict(features)
    model.set_clustering_centers(kmeans.cluster_centers_)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)  # KL divergence loss
    index_array = np.arange(data.shape[0])
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            model.eval()
            with torch.no_grad():
                q, x_recon = model(data)
                p = model.clustering_layer.target_distribution(q).detach()
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            nmi_val, ari_val, acc_val, pur_val = evaluate(y_true, y_pred)
            if ite == 0:
                delta_label = 1.0
                y_pred_last = y_pred.copy()
                print(f'[Iter {ite}] acc: {acc_val:.4f}, nmi: {nmi_val:.4f}, ari: {ari_val:.4f}, delta: {delta_label:.4f}')
            else:
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred.copy()
                print(f'[Iter {ite}] acc: {acc_val:.4f}, nmi: {nmi_val:.4f}, ari: {ari_val:.4f}, delta: {delta_label:.4f}')
                if delta_label < tol:
                    print('Converged, stopping training...'); break
        model.train()
        optimizer.zero_grad()
        idx = index_array[(ite * batch_size) % len(data):(ite * batch_size + batch_size) % len(data)] # problem: when start < len(data), end > len(data), so we need to use modulo operation to ensure the index is within the range of the data
        q_batch, x_recon = model(data[idx]) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        dec_loss = loss_fn(torch.log(q_batch), p[idx])
        mse_loss = nn.MSELoss()(x_recon, data[idx])
        loss = dec_loss + lambda_mse * mse_loss
        loss.backward()
        optimizer.step()
    nmi_val, ari_val, acc_val, pur_val = evaluate(y_true, y_pred)
    print('Final acc:', acc_val)
    print('Final nmi:', nmi_val)
    print('Final ari:', ari_val)
    print('Final pur:', pur_val)

if __name__ == '__main__':
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dims, view, data_size, class_num = load_data('MNIST-USPS')
    dataloader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    batch = next(iter(dataloader))
    multi_view_data = batch[0] # shape: [data_size, view, dims[view]]
    labels = batch[1].numpy()  # True labels
    X_concat = torch.cat(multi_view_data, dim=1).numpy() # shape: [data_size, sum(dims)]
    model = IDEC([sum(dims), 500, 500, 2000, 10], n_clusters=class_num).to(device)
    pretrain_autoencoder(model.auto_encoder, X_concat, epochs=300, lr=1.0)
    label_pred = train_dec(model, X_concat, y_true=labels, tol=1e-3, maxiter=20000, lambda_mse=0.1)
    