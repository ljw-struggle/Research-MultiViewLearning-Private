import random, numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from ..dataset import load_data
from ..metric import evaluate

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

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
    def __init__(self, input_dim, feature_dim, n_clusters, alpha=1.0):
        super(IDEC, self).__init__()
        self.auto_encoder = AutoEncoder(input_dim, feature_dim)
        self.encoder = self.auto_encoder.encoder
        self.clustering_layer = ClusteringLayer(n_clusters, feature_dim, alpha)

    def forward(self, x):
        encoded, decoded = self.auto_encoder(x)
        q = self.clustering_layer(encoded)
        return q, encoded, decoded
    
    def set_clustering_centers(self, centers):
        self.clustering_layer.cluster_centers.data.copy_(torch.tensor(centers, dtype=torch.float32, device=self.clustering_layer.cluster_centers.device))
    
    def extract_clustering_centers(self):
        return self.clustering_layer.cluster_centers.data.cpu().numpy()
    
def benchmark_2017_IJCAI_IDEC(dataset_name='BDGP', use_view=-1, batch_size=256, pretrain_learning_rate=1.0, pretrain_epochs=300, dec_learning_rate=0.01, maxiter=20000, update_interval=140, tol=1e-3, lambda_mse=0.1, verbose=False, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    if use_view == -1:
        data = torch.cat(data, dim=1).to(device) # shape: [data_size, sum(dims)]
    else:
        data = data[use_view].to(device) # shape: [data_size, dims[view]]
    label = label.to(device) # shape: [data_size]

    ## 3. Initialize model and optimizers.
    if use_view == -1:
        model = IDEC(input_dim=sum(dims), feature_dim=10, n_clusters=class_num).to(device)
    else:
        model = IDEC(input_dim=dims[use_view], feature_dim=10, n_clusters=class_num).to(device)
    optimizer_pretrain = torch.optim.SGD(model.parameters(), lr=pretrain_learning_rate, momentum=0.9)
    optimizer_train = torch.optim.SGD(model.parameters(), lr=dec_learning_rate, momentum=0.9)
    criterion_mse = nn.MSELoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean', log_target=False) # batchmean: average the loss over the batch dimension.
    
    ## 4. Pretrain the AutoEncoder.
    model.train()
    for epoch in range(pretrain_epochs):
        perm = torch.randperm(data_size)  # Shuffle the data
        total_loss_list = []
        for i in range(0, data_size, batch_size):
            optimizer_pretrain.zero_grad()
            data_batch = data[perm[i:i+batch_size]]
            _, _, decoded = model(data_batch) # shape: [batch_size, sum(dims)] if view == -1, shape: [batch_size, dims[view]]
            loss = criterion_mse(data_batch, decoded)
            loss.backward()
            optimizer_pretrain.step()
            total_loss_list.append(loss.item())
        print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(total_loss_list):.4f}') if verbose else None
        
    ## 5. Initialize cluster centers with KMeans.
    model.eval()
    q, encoded, decoded = model(data)
    features = encoded.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=class_num, n_init=20)
    kmeans.fit_predict(features)
    model.set_clustering_centers(kmeans.cluster_centers_)
    
    ## 6. Train the IDEC model.
    index_array = np.arange(data_size)
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            model.eval()
            with torch.no_grad():
                q, encoded, decoded = model(data)
                p = model.clustering_layer.target_distribution(q).detach()
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            nmi, ari, acc, pur = evaluate(label.cpu().numpy(), y_pred)
            if ite == 0:
                delta_label = 1.0; y_pred_last = y_pred.copy()
                print(f'[Iter {ite}] acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, pur: {pur:.4f}, delta: {delta_label:.4f}') if verbose else None
            else:
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]; y_pred_last = y_pred.copy()
                print(f'[Iter {ite}] acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, pur: {pur:.4f}, delta: {delta_label:.4f}') if verbose else None
                if delta_label < tol:
                    print('Converged, stopping training...') if verbose else None
                    break # stop training if the change in the label is less than the tolerance
        model.train()
        optimizer_train.zero_grad()
        idx = index_array[(ite * batch_size) % data_size:min((ite * batch_size + batch_size) % data_size, data_size)] # ensure the index is within the range of the data
        q_batch, encoded_batch, decoded_batch = model(data[idx]) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        loss = criterion_kl(torch.log(q_batch), p[idx]) + lambda_mse * criterion_mse(decoded_batch, data[idx])
        loss.backward()
        optimizer_train.step()
    print(f'Final acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, pur: {pur:.4f}') if verbose else None
    return nmi, ari, acc, pur
