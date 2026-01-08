import math, random, argparse, numpy as np, scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from _utils import load_data, evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim))

    def forward(self, x):
        return self.decoder(x)

class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = 2
        self.enc = nn.ModuleList([nn.Linear(input_dim, 512)])
        for i in range(1, 2):
            if i == 2 - 1:
                self.enc.append(nn.Linear(512, output_dim))
            else:
                self.enc.append(nn.Linear(512, 512))

    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z

    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                h = torch.dropout(h, dropout, train=self.training) if dropout else h
                h = layer(h)
            else:
                h = torch.dropout(h, dropout, train=self.training) if dropout else h
                h = layer(h)
                h = F.tanh(h)
        return h

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, 1)

    def forward(self, q, k, v):
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)  # [b, h, q_len, d_k]
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v).transpose(1, 2)  # [b, h, v_len, d_v]
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v)
        x = self.output_layer(x)
        return x
    
class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, hidden_dim, attention_dropout_rate, num_heads, attn_bias_dim, device):
        super(Network, self).__init__()
        self.encoders = []; self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_models = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        self.attention_net = MultiHeadAttention(hidden_dim, attention_dropout_rate, num_heads, attn_bias_dim)
        self.view = view

    def forward(self, xs):
        xrs = []; zs = []; hs = []; ls = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            h = F.normalize(self.feature_contrastive_models(z), dim=1)
            l = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            zs.append(z); hs.append(h); xrs.append(xr); ls.append(l)
        return xrs, zs, hs, ls

    def forward_cluster(self, xs):
        qs = []; preds = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q); preds.append(pred)
        return qs, preds

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N, A):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        half_N = N // 2
        indices = torch.arange(half_N)
        mask[indices, half_N + indices] = 0
        mask[half_N + indices, indices] = 0
        row_indices, col_indices = torch.where(A == 1)
        mask[row_indices, half_N + col_indices] = 0
        mask[row_indices, col_indices] = 0
        mask[half_N + row_indices, col_indices] = 0
        mask[half_N + row_indices, half_N + row_indices] = 0
        mask = mask.bool()
        return mask

    def mask_correlated_samples_label(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        half_N = N // 2
        indices = torch.arange(half_N)
        mask[indices, half_N + indices] = 0
        mask[half_N + indices, indices] = 0
        mask = mask.bool()
        return mask

    def feature_contrastive_loss(self, h_v, h_w, A, gamma):
        N = 2 * self.batch_size
        h = torch.cat((h_v, h_w), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        S_pos_inter_i_j = torch.diag(sim, self.batch_size)
        S_pos_inter_j_i = torch.diag(sim, -self.batch_size)
        S_pos_inter = torch.cat((S_pos_inter_i_j, S_pos_inter_j_i), dim=0).reshape(N, 1)
        S_pos_intra_i_i = torch.sum(sim[:self.batch_size, :self.batch_size] * A, dim=1)
        S_pos_intra_i_j = torch.sum(sim[:self.batch_size, self.batch_size:] * A, dim=1)
        S_pos_intra_i = S_pos_intra_i_i + S_pos_intra_i_j
        S_pos_intra_j_i = torch.sum(sim[self.batch_size:, :self.batch_size] * A, dim=1)
        S_pos_intra_j_j = torch.sum(sim[self.batch_size:, self.batch_size:] * A, dim=1)
        S_pos_intra_j = S_pos_intra_j_i + S_pos_intra_j_j
        S_pos_intra = torch.cat((S_pos_intra_i, S_pos_intra_j), dim=0).reshape(N, 1)
        positive_samples = S_pos_inter + gamma * S_pos_intra
        mask = self.mask_correlated_samples(N, A)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        q_i = self.target_distribution(q_i)
        q_j = self.target_distribution(q_j)
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j
        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples_label(N)
        negative_clusters = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
    
def inference(dataset, loader, model, view, data_size, device):
    model.eval()
    Hs = []
    Zs = []
    soft_vector = []
    pred_vectors = []
    for v in range(view):
        Hs.append([])
        Zs.append([])
        pred_vectors.append([])
    labels_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, preds = model.forward_cluster(xs)
            _, zs, hs, _ = model.forward(xs)
            q = sum(qs) / view
        for v in range(view):
            hs[v] = hs[v].detach()
            zs[v] = zs[v].detach()
            preds[v] = preds[v].detach()
            pred_vectors[v].extend(preds[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Hs[v] = np.array(Hs[v])
        Zs[v] = np.array(Zs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return Hs, Zs, total_pred, pred_vectors, labels_vector

def clustering(model, dataset, view, data_size, cluster_num, device):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    high_level_vectors, low_level_vectors, total_label, pred_label, label_vector = inference(dataset, test_loader, model, view, data_size, device)
    acc2, nmi1, ari1, pur1 = evaluate(label_vector, total_label)
    print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} pur{}={:.4f}'.format(1, acc2, 1, nmi1, 1, ari1, 1, pur1))
    return acc2, nmi1, ari1, pur1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Calculation of distance matrix in the formula
def pairwise_distance(x, y=None):
    x = x.unsqueeze(0).permute(0, 2, 1)
    if y is None:
        y = x
    y = y.permute(0, 2, 1) # [B, N, f]
    A = -2 * torch.bmm(y, x) # [B, N, N]
    A += torch.sum(y**2, dim=2, keepdim=True) # [B, N, 1]
    A += torch.sum(x**2, dim=1, keepdim=True) # [B, 1, N]
    return A.squeeze()

# KNN adjacency matrix calculation
def create_sparse(I):
    similar = I.reshape(-1).tolist()
    index = np.repeat(range(I.shape[0]), I.shape[1])
    assert len(similar) == len(index)
    indices = torch.tensor([index, similar]).cuda()
    result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])
    return result

# AGC module for multi-view data
def node_filter_adaptive_i(dataset, view, K, reg_zero, beta, alpha, verbose=False):
    adj = []
    new_feat = []
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)
    for v in range(view):
        # Calculate matrix A^v through closed form solution
        D = pairwise_distance(xs[v])  # Calculate distance matrix D
        D = F.normalize(D)
        res = torch.mm(xs[v], torch.transpose(xs[v], 0, 1))  # Calculate H*H.T
        inv = torch.inverse(res + beta * torch.eye(xs[v].shape[0]))  # # Inverse matrix in A^v
        front = res - alpha/2 * D  # The first part of A
        S = torch.mm(front, inv)  # obtain A^v
        S = torch.where(S > 0, torch.sign(S), torch.tensor(0.0))  # Eq.(7) If S > 0, the value is 1
        adj.append(S)
        n, f = xs[v].shape
        adj[v] = adj[v].cpu().numpy()
        # S = D^(-1/2) A D^(-1/2)
        deg = np.array(adj[v].sum(axis=1)).flatten()
        if 0 in deg:
            if verbose:
                print("Added self-loops where degree was 0.")
            idcs = np.argwhere(deg == 0).flatten()
            vec = np.zeros(n)
            vec[idcs] = 1
            deg[idcs] = 1.
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ (adj[v] + sp.spdiags(vec, 0, n, n))
        else:
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ adj[v]
        feats_props = np.empty((K + 1, n, f))  # Initialize T
        feats_props[0, :, :] = xs[v].cpu().numpy()
        for i in range(1, K + 1):
            feats_props[i, :, :] = (adj_norm + 1 * np.eye(n)) @ feats_props[i - 1, :, :]  # Calculate T by Eq.(8)
        coeffs = np.empty((n, K + 1))  # Initialize w
        feats_filtered = np.empty((n, f))  # Initialize E
        feats_props = np.transpose(feats_props, (2, 1, 0))
        reg_vec = np.zeros(K + 1);
        reg_zero = np.sqrt(n) * reg_zero
        reg_vec[0] = np.sqrt(n * reg_zero)
        for node_idx in range(n):
            coeffs[node_idx, :], _, _, _ = np.linalg.lstsq(np.vstack((feats_props[:, node_idx, :], reg_vec[None, :])), np.append(feats_props[:, node_idx, 0], np.zeros(1)), rcond=None)  # Solve the w by Eq.(9)
            feats_filtered[node_idx, :] = feats_props[:, node_idx, :] @ coeffs[node_idx, :]  # Calculate E by Eq.(10)
            if verbose:
                print("Finished node %i of %i." % (node_idx, n))
        feats_filtered = torch.tensor(feats_filtered, dtype=torch.float32)
        new_feat.append(feats_filtered)
    dataset.update_view(new_feat)
    print('node filter finish')

# AGC module for multi-graph data
def node_filter_adaptive_g(dataset, view, K, reg_zero, verbose=False):
    print('graph')
    adj = []
    new_feat = []
    n = len(dataset)
    index = torch.arange(n)
    xs, _, _ = dataset.__getitem__(index)
    graph = dataset.get_graph(index)
    for v in range(view):
        adj.append(graph[v])
        n, f = xs[v].shape
        adj[v] = adj[v].cpu().numpy()
        # S = D^(-1/2) A D^(-1/2)
        deg = np.array(adj[v].sum(axis=1)).flatten()
        if 0 in deg:
            if verbose:
                print("Added self-loops where degree was 0.")
            idcs = np.argwhere(deg == 0).flatten()
            vec = np.zeros(n)
            vec[idcs] = 1
            deg[idcs] = 1.
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ (adj[v] + sp.spdiags(vec, 0, n, n))
        else:
            adj_norm = sp.spdiags(deg ** (-1), 0, n, n) @ adj[v]
        feats_props = np.empty((K + 1, n, f))  # Initialize T
        feats_props[0, :, :] = xs[v].cpu().numpy()
        for i in range(1, K + 1):
            feats_props[i, :, :] = (adj_norm + 1 * np.eye(n)) @ feats_props[i - 1, :, :]  # Calculate T by Eq.(8)
        coeffs = np.empty((n, K + 1))  # Initialize w
        feats_filtered = np.empty((n, f))  # Initialize E
        feats_props = np.transpose(feats_props, (2, 1, 0))
        reg_vec = np.zeros(K + 1);
        reg_zero = np.sqrt(n) * reg_zero
        reg_vec[0] = np.sqrt(n * reg_zero)
        for node_idx in range(n):
            coeffs[node_idx, :], _, _, _ = np.linalg.lstsq(np.vstack((feats_props[:, node_idx, :], reg_vec[None, :])), np.append(feats_props[:, node_idx, 0], np.zeros(1)), rcond=None)  # Solve the w by Eq.(9)
            feats_filtered[node_idx, :] = feats_props[:, node_idx, :] @ coeffs[node_idx, :]  # Calculate E by Eq.(10)
            if verbose:
                print("Finished node %i of %i." % (node_idx, n))
        feats_filtered = torch.tensor(feats_filtered, dtype=torch.float32)
        new_feat.append(feats_filtered)
    dataset.update_view(new_feat)
    print('node filter finish')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default='DHA', type=str, choices=['DHA', 'NGs', 'Web', 'Caltech6', 'acm', 'imdb', 'texas', 'chameleon'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--temperature_l", default=1.0)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--mse_epochs", default=200)
    parser.add_argument("--con_epochs", default=50)
    parser.add_argument("--feature_dim", default=512)
    parser.add_argument("--high_feature_dim", default=128)
    parser.add_argument("--top_k", type=int, default=6, help="The number of neighbors to search")
    parser.add_argument("--num_kmeans", type=int, default=5, help="The number of K-means Clustering for being robust to randomness")
    parser.add_argument("--alpha", default=1, help="Reconstruction error coefficient", type=float)
    parser.add_argument("--beta", default=0.1, help="Independence constraint coefficient", type=float)
    parser.add_argument('--attn_bias_dim', type=int, default=6)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.5)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--K', type=int, default=30)
    parser.add_argument('--reg_zero', default=1e10)
    parser.add_argument('--gamma', default=0.001)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "NGs":
        args.learning_rate = 0.0003; args.gamma = 0.001; args.K = 30; args.top_k = 4; args.con_epochs = 50; seed = 10
    if args.dataset == "Web":
        args.learning_rate = 0.0003; args.gamma = 0.1; args.K = 33; args.top_k = 4; args.con_epochs = 200; seed = 10
    if args.dataset == "DHA":
        args.learning_rate = 0.0003; args.gamma = 0.001; args.K = 13; args.top_k = 4; args.con_epochs = 200; seed = 10
    if args.dataset == 'Caltech6':
        args.learning_rate = 0.0004; args.gamma = 0.1; args.K = 5; args.top_k = 4; args.con_epochs = 100; seed = 5
    if args.dataset == "acm":
        args.learning_rate = 0.001; args.gamma = 0.001; args.K = 1; args.top_k = 6; args.con_epochs = 50; seed = 10
    if args.dataset == 'imdb':
        args.learning_rate = 0.0006; args.gamma = 0.01; args.K = 1; args.top_k = 6; args.con_epochs = 50; seed = 10
    if args.dataset == 'texas':
        args.learning_rate = 0.0008; args.gamma = 0.01; args.K = 24; args.top_k = 6; args.con_epochs = 5; seed = 10
    if args.dataset == 'chameleon':
        args.learning_rate = 0.0002; args.gamma = 0.01; args.K = 25; args.top_k = 6; args.con_epochs = 50; seed = 10
    accs = []; nmis = []; aris = []; purs = []; best_acc1 = 0; best_acc2 = 0; best_nmi2 = 0; best_ari2 = 0; best_pur2 = 0; best_epoch = 0; T = 1; acc_l = []; nmi_l = []; ari_l = []; loss_l = []
    for i in range(T):
        dataset, dims, view, data_size, class_num = load_data(args.dataset)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        setup_seed(seed+i)
        if args.dataset in ['DHA', 'NGs', 'Web', 'Caltech6']:
            node_filter_adaptive_i(dataset, view, args.K, args.reg_zero, args.beta, args.alpha, verbose=False)  # AGC
        elif args.dataset in ['acm', 'imdb', 'texas', 'chameleon']:
            node_filter_adaptive_g(dataset, view, args.K, args.reg_zero, verbose=False)  # AGC
        
        model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, args.batch_size, args.attention_dropout_rate, args.num_heads, args.attn_bias_dim, device)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
        for epoch in range(1, args.mse_epochs+1):
            def pretrain(epoch): # view-specific pre-train module
                tot_loss_list = []
                mse = torch.nn.MSELoss()
                for batch_idx, (xs, _, idx) in enumerate(data_loader):
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    xrs, _, _, _ = model(xs)
                    loss = sum([mse(x, xr) for x, xr in zip(xs, xrs)])
                    loss.backward()
                    optimizer.step()
                    tot_loss_list.append(loss.item())
                print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(tot_loss_list)))
            pretrain(epoch)
        for epoch in range(args.mse_epochs+1, args.mse_epochs+args.con_epochs+1):
            def feature_contrastive_train(epoch): # view-cross contrastive learning module
                tot_loss = 0.
                count = 0
                mes = torch.nn.MSELoss()
                for batch_idx, (xs, ys, idx) in enumerate(data_loader):
                    count += 1
                    xs = [x.to(device) for x in xs]
                    optimizer.zero_grad()
                    xrs, zs, hs, ls = model(xs)
                    loss_list = []
                    hs_tensor = torch.tensor([]).cuda()
                    for v in range(view):
                        hs_tensor = torch.cat((hs_tensor, torch.mean(hs[v], 1).unsqueeze(1)), 1)  # mean by feature dimension and connect, (n * v)
                    hs_tensor = hs_tensor.t()  # (v, n)
                    hs_atten = model.attention_net(hs_tensor, hs_tensor, hs_tensor)  # (v * 1)
                    s_p = torch.nn.Softmax(dim=0)
                    r = s_p(hs_atten)  # Calculate view weight vector b^v
                    adaptive_weight = r
                    feature_fuse = torch.zeros([hs[0].shape[0], hs[0].shape[1]]).cuda()  # Initialize H
                    for v in range(view):
                        feature_fuse = feature_fuse + adaptive_weight[v].item() * hs[v] # obtain global feature H by Eq.(12)
                    # Calculate the closed form solution of global graph A
                    D = pairwise_distance(feature_fuse)  # Calculate distance matrix
                    D = F.normalize(D)
                    res = torch.mm(feature_fuse, torch.transpose(feature_fuse, 0, 1))  # Calculate H*H.T
                    inv = torch.inverse(res + args.beta * torch.eye(feature_fuse.shape[0]).to(device))  # Inverse matrix in A
                    front = res - args.alpha/2 * D  # The first part of A
                    S = torch.mm(front, inv)  # Calculate global graph A
                    S = (S + S.t()) / 2
                    S.fill_diagonal_(float('-inf'))  # to ensure that positive samples do not choose themselves repeatedly
                    _, I_knn = S.topk(k=args.top_k, dim=1, largest=True, sorted=True)  # takes the first k largest values and returns the index
                    knn_neighbor = create_sparse(I_knn)  # Construct KNN matrix
                    knn_neighbor = knn_neighbor.to_dense().to(dtype=torch.float32)
                    for v in range(view):
                        loss_list.append(mes(xs[v], xrs[v]))
                        for w in range(v+1, view):
                            loss_list.append(criterion.forward_label(ls[v], ls[w]))  # CGC loss
                            loss_list.append(criterion.feature_contrastive_loss(hs[v], hs[w], knn_neighbor, args.gamma))  # GGC loss
                    loss = sum(loss_list)
                    loss.backward()
                    optimizer.step()
                    tot_loss += loss.item()
                print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
            feature_contrastive_train(epoch)
            acc, nmi, ari, pur = clustering(model, dataset, view, data_size, class_num, device)
        accs.append(round(acc, 4)); nmis.append(round(nmi, 4)); aris.append(round(ari, 4)); purs.append(round(pur, 4))
    print('acc:',accs, np.mean(accs), np.std(accs)); print('nmi:',nmis, np.mean(nmis), np.std(nmis)); print('ari:',aris, np.mean(aris), np.std(aris)); print('pur:',purs, np.mean(purs), np.std(purs))
