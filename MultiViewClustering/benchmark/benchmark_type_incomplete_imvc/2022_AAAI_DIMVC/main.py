import math, random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from utils import load_data, evaluate

class FAE(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(FAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, embed_dim))
        self.decoder = nn.Sequential(nn.Linear(embed_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim))
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, module): # Initialize weights using variance scaling (scale=1/3, mode='fan_in', distribution='uniform')
        if isinstance(module, nn.Linear):
            input_dim = module.weight.size(1) # module.weight.shape = (out_dim, in_dim)
            limit = np.sqrt(1.0 / (3.0 * input_dim))
            nn.init.uniform_(module.weight, -limit, limit)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, hidden_dim))
    
    def forward(self, inputs):
        dist = torch.sum((inputs.unsqueeze(1) - self.clusters.unsqueeze(0)) ** 2, dim=2) # shape: [data_size_complete, n_clusters]
        q = 1.0 / (1.0 + dist / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        return q
    
    def set_cluster_centers(self, centers):
        with torch.no_grad():
            self.clusters.data.copy_(torch.from_numpy(centers).float().to(self.clusters.device)) # inplace operation, no gradient propagation

class MvDEC(nn.Module):
    def __init__(self, view_num, feature_dim_list, embed_dim=10, n_clusters=10, alpha=1.0):
        super(MvDEC, self).__init__()
        self.view_num = view_num
        self.embed_dim = embed_dim
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder_list = nn.ModuleList([FAE(feature_dim_list[v], embed_dim) for v in range(view_num)])
        self.clustering_layer_list = nn.ModuleList([ClusteringLayer(n_clusters, embed_dim, alpha) for _ in range(view_num)])
    
    def encode(self, x_list):
        encoded_list = [self.autoencoder_list[v].encoder(x_list[v]) for v in range(self.view_num)]
        return encoded_list
    
    def decode(self, encoded_list):
        decoded_list = [self.autoencoder_list[v].decoder(encoded_list[v]) for v in range(self.view_num)]
        return decoded_list
    
    def forward(self, x_list):
        encoded_list = self.encode(x_list)
        decoded_list = self.decode(encoded_list)
        q_list = [self.clustering_layer_list[v](encoded_list[v]) for v in range(self.view_num)]
        return q_list, decoded_list, encoded_list
    
    @staticmethod
    def target_distribution(q):
        weight = q ** 2
        return (weight.T / weight.sum(axis=1)).T
    
    @staticmethod
    def get_matched_matrix(y_true, y_pred): # shape: (num_samples,)
        # get the matched matrix using the Hungarian algorithm
        # For example: y_true = [0, 0, 0, 1, 1, 1], y_pred = [1, 1, 0, 0, 0, 0]
        # The matched matrix is: [[0, 1], [1, 0]], where the row indices are the predicted clusters, and the column indices are the true clusters.
        y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
        w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
        row_ind, col_ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, row_ind is the row indices (predicted clusters), col_ind is the column indices (true clusters)
        matched_matrix = np.zeros((y_pred.max()+1, y_true.max()+1), dtype=np.int64) # shape: [num_pred_clusters, num_true_clusters]
        matched_matrix[row_ind, col_ind] = 1
        return matched_matrix # shape: [num_pred_clusters, num_true_clusters]
    
    @staticmethod
    def compute_similarity(inputs, centers, alpha=1.0):
        dist = np.sum((np.expand_dims(inputs, axis=1) - centers) ** 2, axis=2)
        q = 1.0 / (1.0 + dist / alpha)
        q = q ** ((alpha + 1.0) / 2.0)
        q = (q.T / q.sum(axis=1)).T
        # q = q / np.sum(q, axis=1, keepdims=True) # shape: [batch_size, n_clusters]
        return q # shape: [batch_size, n_clusters]
    
def get_mask(view_num, data_size, missing_rate, missing_view_number=1, seed=42):
    assert missing_view_number < view_num, "missing_view_number must be less than view_num"
    assert missing_rate < 1, "missing_rate must be less than 1"
    np.random.seed(seed)
    matrix = np.ones((data_size, view_num))
    if missing_rate == 0:
        return matrix # shape: (data_size, view_num), mask matrix for the samples
    sample_index = np.arange(data_size)
    missing_sample_number = int(missing_rate * data_size)
    missing_sample_index = np.random.choice(sample_index, size=missing_sample_number, replace=False)
    for i in missing_sample_index:
        missing_views = np.random.choice(view_num, size=missing_view_number, replace=False)
        matrix[i, missing_views] = 0
    return matrix # shape: (data_size, view_num), mask matrix for the samples

def benchmark_2022_AAAI_DIMVC(dataset_name="Caltech",
                              missing_rate=0.0,
                              embed_dim=10,
                              alpha=1.0,
                              lc_weight=1.0,
                              lr_weight=1.0,
                              pretrain_epochs=500,
                              num_iterations=10000,
                              update_interval=1000,
                              batch_size=256,
                              learning_rate=0.001,
                              seed=2,
                              verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    # 1. Load data.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, idx = next(iter(dataloader)); x_list = [x_list[v].numpy() for v in range(view)]; y_numpy = y.numpy()
    mask = get_mask(view_num=view, data_size=data_size, missing_rate=missing_rate, missing_view_number=1, seed=seed)
    x_masked_list = [x_list[v] * mask[:, v][:, np.newaxis] for v in range(view)] # shape: [data_size, view]
    x_complete_list = [x_list[v][mask.sum(axis=1) == view] for v in range(view)] # shape: [data_size_complete, view]
    x_masked_list = [torch.from_numpy(x_masked_list[v]).float().to(device) for v in range(view)] # shape: [data_size, view]
    x_complete_list = [torch.from_numpy(x_complete_list[v]).float().to(device) for v in range(view)] # shape: [data_size_complete, view]
    data_size_complete = len(mask.sum(axis=1) == view)
    
    # 2. Pre-training.
    model = MvDEC(view_num=view, feature_dim_list=dims, embed_dim=embed_dim, n_clusters=class_num, alpha=alpha).to(device)
    optimizers = [torch.optim.Adam(model.autoencoder_list[v].parameters(), lr=learning_rate) for v in range(model.view_num)]
    criterion = nn.MSELoss()
    index_array = np.arange(data_size_complete)
    for epoch in range(pretrain_epochs):
        model.train()
        np.random.shuffle(index_array)
        losses = []
        for batch_idx in range(0, data_size_complete, batch_size):
            idx = index_array[batch_idx: min(batch_idx + batch_size, data_size_complete)]
            x_batch = [x_complete_list[v][idx] for v in range(model.view_num)]
            loss_list = [] # shape: [view]
            for v in range(model.view_num):
                optimizers[v].zero_grad()
                encoded, decoded = model.autoencoder_list[v](x_batch[v])
                loss = criterion(decoded, x_batch[v])
                loss.backward()
                optimizers[v].step()
                loss_list.append(loss.item())
            losses.append(np.mean(loss_list)) # shape: [iteration_per_epoch]
        print(f'Reconstruction Pre-training Epoch {epoch}', f'Loss:{np.mean(losses):.6f}') if verbose else None
    model.eval()
    with torch.no_grad():
        encoded_list = model.encode(x_complete_list) # shape: view * [data_size_complete, embed_dim]
        features_list = [encoded.cpu().numpy() for encoded in encoded_list]
    for v in range(model.view_num):
        kmeans = KMeans(n_clusters=model.n_clusters, n_init=100, random_state=42)
        y_pred = kmeans.fit_predict(features_list[v]) # shape: [data_size_complete]
        model.clustering_layer_list[v].set_cluster_centers(kmeans.cluster_centers_)
        nmi, ari, acc, pur = evaluate(y_numpy, y_pred) # shape: [data_size_complete]
        print(f'KMeans on View {v+1}: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, PUR={pur:.4f}') if verbose else None
    
    # 3. Multi-view Clustering with Alternating Optimization.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_mse = nn.MSELoss(reduction='sum')
    criterion_kl = nn.KLDivLoss(reduction='sum')
    index_array = np.arange(data_size_complete)
    batch_num_per_epoch = math.ceil(data_size_complete / batch_size)
    for ite in range(num_iterations):
        # 3.1. P-step: Update the target distribution.
        if ite % update_interval == 0:
            model.eval()
            with torch.no_grad():
                q_list, _, encoded_list = model(x_complete_list)
                encoded_all = [encoded.cpu().numpy() for encoded in encoded_list]
            centers_list = [model.clustering_layer_list[v].clusters.cpu().detach().numpy() for v in range(model.view_num)] # shape: view * [n_clusters, embed_dim]
            per_view_center_var_list = [MinMaxScaler().fit_transform(centers_list[v]).var() for v in range(model.view_num)]
            per_view_weight_list = 1 + np.log2(1 + np.array(per_view_center_var_list) / sum(per_view_center_var_list)) # shape: [view]
            encoded_all_weighted = np.hstack([MinMaxScaler().fit_transform(encoded_all[v]) * per_view_weight_list[v] for v in range(view)]) # shape: [data_size, feature_dim * view]
            kmeans = KMeans(n_clusters=class_num, n_init=10, random_state=42) # corresponds to the paper's L_com loss (local clustering loss)
            y_pred_global = kmeans.fit_predict(encoded_all_weighted)
            if ite == 0:
                y_pred_previous = y_pred_global
            matched_matrix = model.get_matched_matrix(y_pred_global, y_pred_previous) # shape: [num_y_pred_previous_clusters, num_y_pred_global_clusters]
            y_pred_previous = y_pred_global
            P = model.compute_similarity(encoded_all_weighted, kmeans.cluster_centers_)
            P = model.target_distribution(P)
            P = np.dot(P, matched_matrix.T)  # Align with previous iteration's pseudo-labels
            P = torch.from_numpy(P).float().to(device) # shape: [data_size, n_clusters]
        # 3.2. Q-step: Update the clustering centers.
        model.train()
        if ite % batch_num_per_epoch == 0: np.random.shuffle(index_array) # shuffle index_array when starting a new epoch
        idx = index_array[(ite % batch_num_per_epoch) * batch_size: min((ite % batch_num_per_epoch + 1) * batch_size, data_size_complete)]
        x_batch = [x_complete_list[v][idx] for v in range(model.view_num)]; p_batch = P[idx] # shape: [batch_size, n_clusters]
        optimizer.zero_grad()
        q_list, decoded_list, _ = model(x_batch)
        total_loss = 0
        for v in range(model.view_num): # KL divergence loss (categorical crossentropy equivalent)
            clustering_loss = criterion_kl(torch.log(q_list[v] + 1e-10), p_batch)
            recon_loss = criterion_mse(decoded_list[v], x_batch[v])
            total_loss += lc_weight * clustering_loss + lr_weight * recon_loss
        total_loss.backward()
        optimizer.step()
        print(f'Iteration {ite}: Total Loss = {total_loss.item():.6f}') if verbose else None
        
    # 4. Final evaluation on masked data.
    model.eval()
    with torch.no_grad():
        q_list, _, encoded_list = model(x_masked_list)
        q_list = [q.cpu().numpy() for q in q_list]
    # get prediction according to the mask matrix
    q_masked_list = [q_list[v] * mask[:, v][:, np.newaxis] for v in range(view)]
    # Divide by the actual number of available views for each sample, not the total number of views
    q_global = sum(q_masked_list) / mask.sum(axis=1)[:, np.newaxis] # shape: [data_size, n_clusters]
    y_pred_global = np.argmax(q_global, axis=1)
    nmi, ari, acc, pur = evaluate(y_numpy, y_pred_global)
    print(f'Final Evaluation on Masked Data: ACC={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}, PUR={pur:.4f}') if verbose else None
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DIMVC')
    parser.add_argument('--dataset', default='Caltech', type=str, help='Dataset name (Caltech for incomplete data, or standard dataset names)')
    parser.add_argument('--missing_rate', default=0.0, type=float, help='Missing rate for incomplete data (0.0 to 1.0)')
    parser.add_argument('--embed_dim', default=10, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--lc_weight', default=1.0, type=float)
    parser.add_argument('--lr_weight', default=1.0, type=float)
    parser.add_argument('--pretrain_epochs', default=500, type=int)
    parser.add_argument('--num_iterations', default=10000, type=int)
    parser.add_argument('--update_interval', default=1000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=2, type=int)
    args = parser.parse_args()
    nmi, ari, acc, pur = benchmark_2022_AAAI_DIMVC(
        dataset_name=args.dataset,
        missing_rate=args.missing_rate,
        embed_dim=args.embed_dim,
        alpha=args.alpha,
        lc_weight=args.lc_weight,
        lr_weight=args.lr_weight,
        pretrain_epochs=args.pretrain_epochs,
        num_iterations=args.num_iterations,
        update_interval=args.update_interval,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        verbose=False,
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
