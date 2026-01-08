import os, time, yaml, random, argparse, functools, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
from _utils import clustering_acc, MMDataset


def dropout_feature(x, drop_prob):
    x = x.clone()
    drop_mask = torch.empty((x.size(1)), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x[:, drop_mask] = 0
    return x


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model
        self.k = k; assert self.k >= 2
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1): 
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        intra_sim = torch.exp(self.sim(z1, z1) / self.tau) # shape: [num_node, num_node]
        inter_sim = torch.exp(self.sim(z1, z2) / self.tau) # shape: [num_node, num_node]
        return -torch.log(inter_sim.diag() / (intra_sim.sum(1) + inter_sim.sum(1) - intra_sim.diag())) # shape: [num_node]

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        indices = torch.arange(0, num_nodes).to(z1.device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            intra_sim = torch.exp(self.sim(z1[mask], z1) / self.tau) # shape: [batch_size, num_node]
            inter_sim = torch.exp(self.sim(z1[mask], z2) / self.tau) # shape: [batch_size, num_node]
            losses.append(-torch.log(inter_sim[:, mask].diag() / (intra_sim.sum(1) + inter_sim.sum(1) - intra_sim[:, mask].diag()))) # shape: [batch_size]
        return torch.cat(losses) # shape: [num_node]

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2) if batch_size == 0 else self.batched_semi_loss(h1, h2, batch_size)
        l2 = self.semi_loss(h2, h1) if batch_size == 0 else self.batched_semi_loss(h2, h1, batch_size)
        result = (l1 + l2) * 0.5
        result = result.mean() if mean else result.sum()
        return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    args = parser.parse_args()
    random.seed(12345)
    torch.manual_seed(39788)
    torch.cuda.set_device(0)
    learning_rate = 0.0005; num_hidden = 128; num_proj_hidden = 128; activation = F.relu; base_model = GCNConv; num_layers = 2
    drop_edge_rate_1 = 0.2; drop_edge_rate_2 = 0.4; drop_feature_rate_1 = 0.3; drop_feature_rate_2 = 0.4
    tau = 0.4; num_epochs = 200; weight_decay = 0.00001

    dataset = MMDataset('./data/data_sc_multiomics/TEA/', concat_data=True)
    data = dataset.X; label = dataset.Y; edge_index = dataset.get_hyperedge_index(); data_categories = dataset.categories
    data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    edge_index = torch.from_numpy(edge_index).long().to(device)
    encoder = Encoder(sum(data_features), num_hidden, activation, base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = time.perf_counter()
    prev = start
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        edge_index_1 = dropout_edge(edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_edge(edge_index, p=drop_edge_rate_2)[0]
        x_1 = dropout_feature(data, drop_feature_rate_1)
        x_2 = dropout_feature(data, drop_feature_rate_2)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2, batch_size=0)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        now = time.perf_counter()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")
    model.eval()
    z = model(data, edge_index)
    embedding = z.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=data_categories, n_init=20, random_state=0)
    preds = kmeans.fit_predict(embedding)
    acc_val = clustering_acc(label, preds) * 100
    nmi_val = normalized_mutual_info_score(label, preds) * 100
    asw_val = silhouette_score(embedding, preds) * 100 if np.unique(preds).shape[0] > 1 else 0
    ari_val = adjusted_rand_score(label, preds) * 100
    print(f"Final completed. ACC: {acc_val:.4f}, NMI: {nmi_val:.4f}, ASW: {asw_val:.4f}, ARI: {ari_val:.4f}")
