import os, random, argparse, numpy as np
from utils import get_data, get_clustering_performance, BaseLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(GCNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.act = act

        self.encoder_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            if i == 0:
                self.encoder_layers.append(pyg_nn.GCNConv(self.input_dim, self.hidden_dims[i]))
            else:
                self.encoder_layers.append(pyg_nn.GCNConv(self.hidden_dims[i - 1], self.hidden_dims[i]))
        self.encoder_layers.append(pyg_nn.GCNConv(self.hidden_dims[-1], self.latent_dim))

    def forward(self, features, graph):
        graph.fill_diagonal_(1)
        degree = graph.sum(dim=1)
        degree_inv_sqrt = torch.diag(degree.pow(-0.5))
        graph = degree_inv_sqrt @ graph @ degree_inv_sqrt
        edge_index = graph.nonzero(as_tuple=True)
        edge_index = torch.stack(edge_index).to(torch.long)
        edge_weight = graph[edge_index[0], edge_index[1]]
        data = Data(x=features, edge_index=edge_index, edge_attr=edge_weight)
        z, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for _, layer in enumerate(self.encoder_layers):
            z = layer(z, edge_index, edge_weight=edge_attr)
            z = self.act(z)

        return z


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.act = act

        encoder_layers = []
        for i in range(len(self.hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(self.input_dim, self.hidden_dims[i]))
            else:
                encoder_layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            encoder_layers.append(self.act)
        encoder_layers.append(nn.Linear(self.hidden_dims[-1], self.latent_dim))
        encoder_layers.append(self.act)
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        z = self.encoder(x)

        return z


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, act=nn.ReLU()):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(reversed(hidden_dims))
        self.latent_dim = latent_dim
        self.act = act

        decoder_layers = []

        for i in range(len(self.hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(self.latent_dim, self.hidden_dims[i]))
            else:
                decoder_layers.append(nn.Linear(self.hidden_dims[i - 1], self.hidden_dims[i]))
            decoder_layers.append(self.act)
        decoder_layers.append(nn.Linear(self.hidden_dims[-1], self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x_rec = self.decoder(x)

        return x_rec


class FusionModule(nn.Module):
    def __init__(self, latent_dim, view_num, embedding_dim, fusion_mode="average_sum"):
        super(FusionModule, self).__init__()
        self.latent_dim = latent_dim
        self.view_num = view_num
        self.embedding_dim = embedding_dim
        self.fusion_mode = fusion_mode
        if fusion_mode == "concat":
            self.embedding_layer = nn.Linear(latent_dim * view_num, embedding_dim)
        elif fusion_mode == "weighted_sum":
            self.weight_assignment = nn.Sequential(
                nn.Linear(latent_dim * view_num, view_num)
            )
            self.embedding_layer = nn.Linear(latent_dim, embedding_dim)
        else:
            self.embedding_layer = nn.Linear(latent_dim, embedding_dim)

    def forward(self, specific_x_, observed_indices, sample_mask):
        specific_x = []
        for v in range(self.view_num):
            specific_x.append(observed_indices[v] @ specific_x_[v])
        if self.fusion_mode == "concat":
            fusion_x = torch.cat(specific_x, dim=1)
            joint_x = self.embedding_layer(fusion_x)
        elif self.fusion_mode == "weighted_sum":
            weights = self.weight_assignment(torch.cat(specific_x, dim=1))
            weights = torch.softmax(weights * sample_mask, dim=-1)
            weights_chunk = torch.chunk(weights, self.view_num, dim=1)
            fusion_x = []
            for v in range(self.view_num):
                fusion_x.append(specific_x[v] * weights_chunk[v])
            fusion_x = sum(fusion_x)
            joint_x = self.embedding_layer(fusion_x)
        elif self.fusion_mode == "average_sum":
            sample_mask = sample_mask.unsqueeze(1).repeat(1, self.latent_dim, 1)
            x_stack = torch.stack(specific_x, dim=-1)  # (n, d, v)
            weighted_sum = torch.sum(sample_mask * x_stack, dim=-1)
            weights_sum = sample_mask.sum(dim=-1)
            fusion_x = weighted_sum / weights_sum
            joint_x = self.embedding_layer(fusion_x)
        else:
            joint_x = None

        return joint_x


class MvAEModel(nn.Module):
    def __init__(self,
                 input_dims,
                 view_num,
                 latent_dim,
                 hid_dims=None,
                 cluster_num=None,
                 act=nn.ReLU()
                 ):
        super().__init__()
        if hid_dims is None:
            hid_dims = [192, 128]
        self.input_dims = input_dims
        self.view_num = view_num
        self.latent_dim = latent_dim
        self.hid_dims = hid_dims
        self.cluster_num = cluster_num
        self.act = act
        self.embedding_dim = int(1.5 * latent_dim)
        # encoder decoder
        self.view_specific_encoders = nn.ModuleList()
        self.view_specific_decoders = nn.ModuleList()
        for v in range(self.view_num):
            self.view_specific_encoders.append(GCNEncoder(self.input_dims[v], self.hid_dims, self.latent_dim))
            self.view_specific_decoders.append(Decoder(self.input_dims[v], self.hid_dims, self.embedding_dim))
        # feature fusion layer
        self.fusion_layer = FusionModule(self.latent_dim, self.view_num, self.embedding_dim)
        # clustering layer
        self.cluster_layer = nn.Linear(self.embedding_dim, self.cluster_num)

    def forward(self, x_list, graph_list=None, observed_indices=None, sample_mask=None, is_training=False):
        specific_z = []
        for v in range(self.view_num):
            z = self.view_specific_encoders[v](x_list[v], graph_list[v])
            specific_z.append(z)

        joint_z = self.fusion_layer(specific_z, observed_indices, sample_mask)
        joint_y = self.cluster_layer(joint_z)

        recs = []
        for v in range(self.view_num):
            rec = self.view_specific_decoders[v](observed_indices[v].t() @ joint_z)
            recs.append(rec)

        if is_training:
            return recs, joint_z, joint_y
        else:
            return joint_z, joint_y


import torch
import torch.nn.functional as F


def contrastive_loss(z, mask):
    z = F.normalize(z, dim=1, p=2)
    s = torch.mm(z, z.t())
    pos_mask = mask.fill_diagonal_(0)
    neg_mask = 1 - mask.fill_diagonal_(1)

    tau = 1.0 # 1.0
    s = torch.exp(s / tau)
    pos_loss = (pos_mask * s).sum(1)
    neg_loss = (neg_mask * s).sum(1)
    loss_con = - torch.log(pos_loss / neg_loss).mean()

    return loss_con


def contrastive_loss2(z, mask):
    z = F.normalize(z, dim=1, p=2)
    s = torch.mm(z, z.t())
    pos_mask = mask.fill_diagonal_(0)
    neg_mask = 1 - mask.fill_diagonal_(1)

    tau = 1.0 # 1.0
    s = torch.exp(s / tau)
    pos_loss = (pos_mask * s).sum(1)
    neg_loss = (neg_mask * s).sum(1)
    loss_con = - torch.log(pos_loss / (pos_loss + neg_loss)).mean()

    return loss_con

def knn_graph(distances, k=15):
    N = distances.shape[0]
    idx = torch.argsort(distances, dim=1)[:, :k + 1]  # Shape (N, k + 1)
    neighbors_idx = idx[:, 1:k + 1]  # Exclude the first column (self)
    d = distances[torch.arange(N).unsqueeze(1), neighbors_idx]  # Shape (N, k)
    adjacency_matrix = torch.zeros((N, N), dtype=distances.dtype, device=distances.device)
    eps = 1e-8
    d_k_minus_1 = d[:, -1]  # k-th nearest distance (last in the row)
    sum_d = torch.sum(d, dim=1)  # Sum of distances for each row
    adjacency_matrix[torch.arange(N).unsqueeze(1), neighbors_idx] \
        = (d_k_minus_1.unsqueeze(1) - d) / (k * d_k_minus_1.unsqueeze(1) - sum_d.unsqueeze(1) + eps)
    adjacency_matrix.fill_diagonal_(1)
    adjacency_matrix = 0.5 * (adjacency_matrix + adjacency_matrix.t())
    return adjacency_matrix


def set_seed():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default="MSRCV1-3view", help='Data directory e.g. MSRCV1')
    parser.add_argument('--train_epochs', type=int, default=3000, help='Max. number of epochs')
    parser.add_argument('--alpha', type=int, default=-3, help='Parameter: alpha')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--log_save_dir', default='./logs/', help='Directory to save the results')
    parser.add_argument('--miss_rate', type=float, default=0.5)
    parser.add_argument('--repeat_times', type=float, default=5)
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    set_seed()
    random_numbers_for_kmeans = random.sample(range(10000), 10)
    print(random_numbers_for_kmeans)
    os.makedirs(args.log_save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    latent_dim = 64
    hid_dims = [192, 128]

    metrics = {
        "ACC": [], "NMI": [], "Purity": [],
        "ARI": [], "Fscore": [], "Precision": [], "Recall": []
    }

    logger = BaseLogger(log_save_dir=args.log_save_dir, log_name=args.data_name + '-' + str(args.miss_rate) + '.csv')
    print("====================== start training ======================")
    print("data_name:", args.data_name, "alpha:", args.alpha)
    logger.write_parameters(args.alpha)
    if args.miss_rate == 0:
        args.repeat_times = 1
    for mask_seed in range(1, args.repeat_times + 1):
        mask, data_x_, data_y, view_num, sample_num, cluster_num, input_dims = get_data(args.data_name, args.miss_rate, mask_seed)
        set_seed()
        data_y = torch.from_numpy(data_y).to(device=device)
        data_x = []
        for v in range(view_num):
            data_x.append(torch.from_numpy(data_x_[v]).to(dtype=torch.float32, device=device))

        observed_mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
        missing_mask = 1 - observed_mask

        graph_x = []
        graph_mask = []
        observed_transition = [] # N x N_o
        missing_transition = [] # N x N_m
        for v in range(view_num):
            observed_sample_num = int(observed_mask[:, v].sum().item())
            if observed_sample_num == 0:
                observed_transition_v = torch.zeros(sample_num, 0).to(dtype=torch.float32, device=device)
            else:
                observed_transition_v = torch.eye(sample_num).to(dtype=torch.float32, device=device)[:, observed_mask[:, v].bool()]

            missing_sample_num = int(missing_mask[:, v].sum().item())
            if missing_sample_num == 0:
                missing_transition_v = torch.zeros(sample_num, 0).to(dtype=torch.float32, device=device)
            else:
                missing_transition_v = torch.eye(sample_num).to(dtype=torch.float32, device=device)[:, missing_mask[:, v].bool()]

            observed_transition.append(observed_transition_v)
            missing_transition.append(missing_transition_v)
            # print(observed_transition[v].size())
            # print(missing_transition[v].size())
            graph_mask.append(observed_mask[:, v:v+1] @ observed_mask[:, v:v+1].t())
            data_x[v] = observed_transition[v].t() @ data_x[v]
            graph_x.append(knn_graph(torch.cdist(data_x[v], data_x[v]) ** 2))

        model = MvAEModel(input_dims,
                          view_num,
                          latent_dim=latent_dim,
                          hid_dims=hid_dims,
                          cluster_num=cluster_num
                          )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        losses = []
        mask_z = 0
        for epoch in range(args.train_epochs):
            # Train
            model.train()
            recs, joint_z, joint_y = model(data_x, graph_x, observed_transition, observed_mask, is_training=True)
            loss_rec = 0

            # graph fusion for contrastive learning
            graph_z = knn_graph(torch.cdist(joint_z, joint_z) ** 2)
            graph_masks = [torch.ones(sample_num, sample_num).to(dtype=torch.float32, device=device)] + graph_mask
            graphs = [graph_z]
            for v in range(view_num):
                graphs.append(observed_transition[v] @ graph_x[v] @ observed_transition[v].t())
            graph_masks = torch.stack(graph_masks, dim=-1)
            graphs = torch.stack(graphs, dim=-1)
            weighted_sum = torch.sum(graph_masks * graphs, dim=-1)
            weights_sum = graph_masks.sum(dim=-1)
            fusion_graph = weighted_sum / weights_sum
            loss_con = contrastive_loss(joint_y, fusion_graph)

            for v in range(view_num):
                loss_rec += F.mse_loss(recs[v], data_x[v])

            optimizer.zero_grad()
            loss = loss_rec + 10 ** args.alpha * loss_con
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                print("dataset: %s, epoch: %d, loss_rec: %.6f, loss_con: %.6f" % (
                    args.data_name, epoch, loss_rec.data.item(), loss_con.data.item()))

        # Test
        model.eval()
        joint_z, joint_y = model(data_x, graph_x, observed_transition, observed_mask)
        mask_seed_metrics = get_clustering_performance(
            joint_z.detach().cpu().numpy(),
            data_y.detach().cpu().numpy(),
            cluster_num,
            random_numbers_for_kmeans
        )
        metrics["ACC"].append(mask_seed_metrics["ACC"])
        metrics["NMI"].append(mask_seed_metrics["NMI"])
        metrics["Purity"].append(mask_seed_metrics["Purity"])
        metrics["ARI"].append(mask_seed_metrics["ARI"])
        metrics["Fscore"].append(mask_seed_metrics["Fscore"])
        metrics["Precision"].append(mask_seed_metrics["Precision"])
        metrics["Recall"].append(mask_seed_metrics["Recall"])

    # Calculate average performance metrics
    average_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}
    average_scores = [
        average_metrics["ACC"],
        average_metrics["NMI"],
        average_metrics["Purity"],
        average_metrics["ARI"],
        average_metrics["Fscore"],
        average_metrics["Precision"],
        average_metrics["Recall"],
              ]
    std_scores = [
        std_metrics["ACC"],
        std_metrics["NMI"],
        std_metrics["Purity"],
        std_metrics["ARI"],
        std_metrics["Fscore"],
        std_metrics["Precision"],
        std_metrics["Recall"],
    ]
    logger.write_val(epoch, loss, average_scores)
    logger.write_val(epoch, loss, std_scores)

    logger.close_logger()


