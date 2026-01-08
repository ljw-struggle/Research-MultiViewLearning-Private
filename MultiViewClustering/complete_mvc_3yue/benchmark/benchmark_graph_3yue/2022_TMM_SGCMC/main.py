import argparse
import numpy as np
import scipy.io as sio
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data, load_graph, prepare_graph_data
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from utils import best_map, nmi, ari, f_score

def sparse_softmax(indices, values, shape, dim=0):
    """Sparse softmax along dimension dim (0 = per row for shape [N, N])."""
    # indices: (2, nnz), values: (nnz,)
    # For dim=0 we normalize by row: group by indices[0]
    row = indices[0]
    n_rows = shape[0]
    device = values.device
    # Compute max per row for numerical stability
    row_max = torch.full((n_rows,), float('-inf'), device=device, dtype=values.dtype)
    row_max.scatter_reduce_(0, row, values, reduce='amax')
    # Subtract max and exp
    exp_vals = torch.exp(values - row_max[row])
    # Sum per row
    row_sum = torch.zeros(n_rows, device=device, dtype=values.dtype)
    row_sum.scatter_add_(0, row, exp_vals)
    # Normalize
    out_vals = exp_vals / (row_sum[row] + 1e-12)
    return out_vals

def graph_attention_layer(A_sparse, M, v0, v1, layer_idx):
    """
    A_sparse: torch.sparse_coo_tensor (N, N), coalesced.
    M: (N, hidden_dim), v0, v1: (hidden_dim, 1)
    Returns sparse attention matrix C (N, N).
    """
    # f1 = M @ v0 -> (N, 1); f2 = M @ v1 -> (N, 1)
    f1 = torch.mm(M, v0)   # (N, 1)
    f2 = torch.mm(M, v1)   # (N, 1)
    indices = A_sparse.indices()   # (2, nnz)
    row, col = indices[0], indices[1]
    # logit_ij = f1[i] + f2[j]
    logit_vals = f1[row, 0] + f2[col, 0]
    unnorm_att = torch.sigmoid(logit_vals)
    att_vals = sparse_softmax(indices, unnorm_att, A_sparse.shape, dim=0)
    C = torch.sparse_coo_tensor(indices, att_vals, A_sparse.shape, dtype=M.dtype, device=M.device)
    return C.coalesce()

class GATE(nn.Module):
    def __init__(self, hidden_dims, hidden_dims2, lambda_, num_nodes=10299, n_clusters=6, encoder_dim=256):
        super(GATE, self).__init__()
        self.lambda_ = lambda_
        self.n_layers = len(hidden_dims) - 1
        self.n_layers2 = len(hidden_dims2) - 1
        self.num_nodes = num_nodes
        self.params = {"n_clusters": n_clusters, "encoder_dims": [encoder_dim], "alpha": 1.0}
        self.n_cluster = n_clusters
        # Encoder weights: W[i] (hidden_dims[i], hidden_dims[i+1]), v[i] two vectors (hidden_dims[i+1], 1)
        self.W = nn.ParameterList()
        self.v0_list = nn.ParameterList()
        self.v1_list = nn.ParameterList()
        for i in range(self.n_layers):
            self.W.append(nn.Parameter(torch.empty(hidden_dims[i], hidden_dims[i + 1])))
            self.v0_list.append(nn.Parameter(torch.empty(hidden_dims[i + 1], 1)))
            self.v1_list.append(nn.Parameter(torch.empty(hidden_dims[i + 1], 1)))
        for i in range(self.n_layers):
            nn.init.xavier_uniform_(self.W[i])
            nn.init.xavier_uniform_(self.v0_list[i])
            nn.init.xavier_uniform_(self.v1_list[i])
        # Coefficient matrix
        self.weight = nn.Parameter(1.0e-4 * torch.ones(num_nodes, num_nodes))
        self.mu = nn.Parameter(torch.zeros(n_clusters, encoder_dim))
        # Classifier heads: flatten -> 512 -> 6
        last_dim1 = hidden_dims[-1]
        last_dim2 = hidden_dims2[-1]
        self.fc1 = nn.Linear(last_dim1, 512)
        self.fc_z = nn.Linear(512, 6)
        self.fc2 = nn.Linear(last_dim2, 512)
        self.fc_z2 = nn.Linear(512, 6)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_z.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_z2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc_z.bias)
            nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.fc_z2.bias)
        self._C = {}   # cache attention matrices for decoder

    def _encoder(self, A_sparse, H, layer):
        H = torch.mm(H, self.W[layer])   # (N, in_d) @ (in_d, out_d)
        C = graph_attention_layer(A_sparse, H, self.v0_list[layer], self.v1_list[layer], layer)
        self._C[layer] = C
        H = torch.sparse.mm(C, H)
        return H

    def _decoder(self, H, layer):
        H = torch.mm(H, self.W[layer].T)   # (N, out_d) @ (out_d, in_d)
        H = torch.sparse.mm(self._C[layer], H)
        return H

    def forward(self, A, A2, X, X2, S, S2, R, R2, p, Theta, Labels):
        # A, A2: already torch.sparse_coo_tensor or (indices, data, shape) tuple
        device = X.device
        if not isinstance(A, torch.Tensor):
            ind, data, size = A
            # ind is (num_edges, 2) with [col, row]; PyTorch wants (2, nnz) [row, col]
            ind_pt = torch.from_numpy(ind).long().to(device)
            ind_pt = torch.stack([ind_pt[:, 1], ind_pt[:, 0]], dim=0)
            A = torch.sparse_coo_tensor(ind_pt, torch.from_numpy(data).float().to(device), size, device=device).coalesce()
        if not isinstance(A2, torch.Tensor):
            ind, data, size = A2
            ind_pt = torch.from_numpy(ind).long().to(device)
            ind_pt = torch.stack([ind_pt[:, 1], ind_pt[:, 0]], dim=0)
            A2 = torch.sparse_coo_tensor(ind_pt, torch.from_numpy(data).float().to(device), size, device=device).coalesce()
        coef = self.weight - torch.diag(torch.diagonal(self.weight))
        # Encoder1
        H = X
        for layer in range(self.n_layers):
            H = self._encoder(A, H, layer)
        H_enc1 = H
        HC = torch.mm(coef, H_enc1)
        H = HC
        # Decoder1
        for layer in range(self.n_layers - 1, -1, -1):
            H = self._decoder(H, layer)
        X_recon = H
        # z from H_enc1
        layer_flat = H_enc1.reshape(-1, H_enc1.shape[1])
        layer_full = self.fc1(layer_flat)
        z = self.fc_z(layer_full)
        # Encoder2
        H2 = X2
        for layer in range(self.n_layers2):
            H2 = self._encoder(A2, H2, layer)
        H_enc2 = H2
        HC2 = torch.mm(coef, H_enc2)
        H2 = HC2
        # Decoder2
        for layer in range(self.n_layers2 - 1, -1, -1):
            H2 = self._decoder(H2, layer)
        X2_recon = H2
        layer_flat2 = H_enc2.reshape(-1, H_enc2.shape[1])
        layer_full2 = self.fc2(layer_flat2)
        z2 = self.fc_z2(layer_full2)
        # Losses
        features_loss = (X - X_recon).pow(2).sum() + (X2 - X2_recon).pow(2).sum()
        SE_loss = 0.5 * (H_enc1 - HC).pow(2).sum() + 0.5 * (H_enc2 - HC2).pow(2).sum()
        S_Regular = coef.abs().sum()
        Cq_loss = (coef.T * Theta).abs().sum()
        # Cross entropy: p is one-hot (N, 6)
        if p.dim() == 2 and p.shape[1] == 6:
            target = p.argmax(dim=1)
        else:
            target = p.long()
        dense_loss = F.cross_entropy(z, target, reduction='sum') + F.cross_entropy(z2, target, reduction='sum')
        # Structure loss: -log(sigmoid(inner_product))
        S_emb = H_enc1[S]
        R_emb = H_enc1[R]
        structure_loss1 = -F.logsigmoid((S_emb * R_emb).sum(dim=-1)).sum()
        S_emb2 = H_enc2[S2]
        R_emb2 = H_enc2[R2]
        structure_loss2 = -F.logsigmoid((S_emb2 * R_emb2).sum(dim=-1)).sum()
        structure_loss = structure_loss1 + structure_loss2
        consistent_loss = (H_enc1 - H_enc2).pow(2).sum()
        pre_loss = features_loss + self.lambda_ * structure_loss + 10 * SE_loss + 0.01 * consistent_loss + 1 * S_Regular
        loss = 1e-2 * features_loss + self.lambda_ * structure_loss + 10 * SE_loss + 1e-3 * consistent_loss + 1 * S_Regular + 5 * Cq_loss + 5 * dense_loss
        return pre_loss, loss, dense_loss, features_loss, structure_loss, SE_loss, coef, consistent_loss, S_Regular, Cq_loss, H_enc1, H_enc2

def get_one_hot_Label(Label):
    if Label.min() == 0:
        Label = Label
    else:
        Label = Label - 1
    Label = np.array(Label)
    n_class = 6
    n_sample = Label.shape[0]
    one_hot_Label = np.zeros((n_sample, n_class))
    for i, j in enumerate(Label):
        one_hot_Label[i, j] = 1
    return one_hot_Label

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

def form_Theta(Q):
    Theta = np.zeros((Q.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]):
        Qq = np.tile(Q[i], [Q.shape[0], 1])
        Theta[i, :] = 1 / 2 * np.sum(np.square(Q - Qq), 1)
    return Theta

def form_structure_matrix(idx, K):
    Q = np.zeros((len(idx), K))
    for i, j in enumerate(idx):
        Q[i, j - 1] = 1
    return Q

def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    uu, ss, vv = svds(L, k=K)
    return grp, uu

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while stop == False:
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = args.num_nodes
        self.model = GATE(
            args.hidden_dims_1,
            args.hidden_dims_2,
            args.lambda_,
            num_nodes=num_nodes,
        ).to(self.device)
        self.pre_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def _to_device(self, A, A2, X, X2, S, S2, R, R2, p, Theta):
        """Convert numpy inputs to device tensors. A, A2 stay as (ind, data, shape) for forward."""
        X = torch.from_numpy(X).float().to(self.device)
        X2 = torch.from_numpy(X2).float().to(self.device)
        S = torch.from_numpy(S).long().to(self.device)
        R = torch.from_numpy(R).long().to(self.device)
        S2 = torch.from_numpy(S2).long().to(self.device)
        R2 = torch.from_numpy(R2).long().to(self.device)
        p = torch.from_numpy(p).float().to(self.device)
        Theta = torch.from_numpy(Theta).float().to(self.device)
        return A, A2, X, X2, S, S2, R, R2, p, Theta

    def __call__(self, A, A2, X, X2, S, S2, R, R2, L):
        L = np.asarray(L).flatten()
        Q = form_structure_matrix(L, 6)
        Theta_np = form_Theta(Q) * 0
        A, A2, X, X2, S, S2, R, R2, p_dummy, Theta = self._to_device(
            A, A2, X, X2, S, S2, R, R2,
            np.zeros((X.shape[0], 6)),  # p not used in pre-train
            Theta_np,
        )

        # Pre-train: 1 epoch
        pre_epoch = 0
        while pre_epoch < 1:
            self.model.train()
            self.pre_optimizer.zero_grad()
            pre_loss, _, structure_loss, features_loss, _, consistent_loss, cRegular, SE_loss, _, _, _, _ = self.model(
                A, A2, X, X2, S, S2, R, R2, p_dummy, Theta, None
            )
            pre_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
            self.pre_optimizer.step()
            pre_epoch += 1
            if pre_epoch % 5 == 0:
                print("-------------------------------------------------------------")
                print("pre_epoch: %d" % pre_epoch, "Pre_Loss: %.2f" % pre_loss.item(),
                      "ReLoss-X: %.2f" % features_loss.item(), "ReLoss-A: %.2f" % structure_loss.item(),
                      "pre_SE_loss: %.2f" % SE_loss.item(), "Pre_consistent_loss: %.2f" % consistent_loss.item(),
                      "cRegular: %.2f" % cRegular.item())

        with torch.no_grad():
            _, _, _, _, _, _, coef, _, _, _, _, _ = self.model(A, A2, X, X2, S, S2, R, R2, p_dummy, Theta, None)
        coef = coef.detach().cpu().numpy()

        alpha = max(0.4 - (6 - 1) / 10 * 0.1, 0.1)
        commonZ = thrC(coef, alpha)
        y_x, _ = post_proC(commonZ, 6, 10, 3.5)
        missrate_x = err_rate(L, y_x + 1)
        acc_x = 1 - missrate_x
        print('----------------------------------------------------------------------------------------------------------')
        print("Initial Clustering Results: ")
        print("acc: {:.8f}\t\tnmi: {:.8f}\t\tf_score: {:.8f}\t\tari: {:.8f}".
              format(acc_x, nmi(L, y_x + 1), f_score(L, y_x + 1), ari(L, y_x + 1)))
        print('----------------------------------------------------------------------------------------------------------')
        epoch = 0
        s2_label_subjs = np.array(y_x)
        s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
        s2_label_subjs = np.squeeze(s2_label_subjs)
        one_hot_Label = get_one_hot_Label(s2_label_subjs)
        s2_Q = form_structure_matrix(s2_label_subjs, 6)
        s2_Theta = form_Theta(s2_Q)
        Y = y_x
        while epoch < self.args.n_epochs:
            p_pt = torch.from_numpy(one_hot_Label).float().to(self.device)
            Theta_pt = torch.from_numpy(s2_Theta).float().to(self.device)
            self.model.train()
            self.optimizer.zero_grad()
            pre_loss, cost, dense_loss, features_loss, structure_loss, SE_loss, s2_Coef, consistent_loss, cRegular, cqLoss, H, H2 = self.model(
                A, A2, X, X2, S, S2, R, R2, p_pt, Theta_pt, Y
            )
            cost.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
            self.optimizer.step()
            cost = cost.item()
            st_loss = structure_loss.item()
            f_loss = features_loss.item()
            CrossELoss = dense_loss.item()
            SELoss = SE_loss.item()
            consistent_loss = consistent_loss.item()
            cRegular = cRegular.item()
            cqLoss = cqLoss.item()
            s2_Coef = s2_Coef.detach().cpu().numpy()
            if epoch % 5 == 0:
                s2_label_subjs = np.array(Y)
                s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
                s2_label_subjs = np.squeeze(s2_label_subjs)
                one_hot_Label = get_one_hot_Label(s2_label_subjs)
                s2_Q = form_structure_matrix(s2_label_subjs, 6)
                s2_Theta = form_Theta(s2_Q)
            s2_Coef = thrC(s2_Coef, alpha)
            y_x, Soft_Q = post_proC(s2_Coef, L.max(), 10, 3.5)
            if len(np.unique(y_x)) != 6:
                epoch += 1
                continue
            Y = best_map(Y + 1, y_x + 1) - 1
            Y = Y.astype(np.int64)
            s2_missrate_x = err_rate(L, Y + 1)
            s2_acc_x = 1 - s2_missrate_x
            s2_nmi_x = nmi(L, Y + 1)
            s2_ari_x = ari(L, Y + 1)
            print("epoch: %d" % epoch, "Total_Loss: %.2f" % cost, "ReLoss-X: %.2f" % f_loss, "ReLoss-A: %.2f" % st_loss,
                  "SeLoss: %.2f" % SELoss, "CrossELoss: %.2f" % CrossELoss, "consistent_loss: %.2f" % consistent_loss,
                  "cRegular: %.2f" % cRegular, "cqLoss: %.2f" % cqLoss)
            print("Rearrange:", "\033[1;31;43m SGCMC_Acc:%.4f \033[0m" % s2_acc_x)
            print("Rearrange:", "\033[1;31;43m SGCMC_Nmi:%.4f \033[0m" % s2_nmi_x)
            print("Rearrange:", "\033[1;31;43m SGCMC_Ari:%.4f \033[0m" % s2_ari_x)
            epoch += 1
            fh = open('Harr_Results.txt', 'a')
            fh.write('Fin_epoch=%d, ACC=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (epoch, s2_acc_x, s2_nmi_x, s2_ari_x))
            fh.write('\r\n')
            fh.flush()
            fh.close()
            fh2 = open('Harr_Loss.txt', 'a')
            fh2.write('Fin_epoch=%d, Fin_Total: %.2f, Re_Loss: %.2f, SEloss: %.2f, dense_loss: %.2f, consistent_loss: %.2f, cRegular: %.2f, Cq_loss: %.2f ' % (
                    epoch, cost, f_loss + st_loss, SELoss, CrossELoss, consistent_loss, cRegular, cqLoss))
            fh2.write('\r\n')
            fh2.flush()
            fh2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gate.")
    parser.add_argument('--dataset', nargs='?', default='cora', help='Input dataset')
    parser.add_argument('--seed', type=int, default=30) #33
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate. Default is 0.001.')
    parser.add_argument('--n-epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('--hidden-dims-1', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--hidden-dims-2', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--lambda-', default=1.0, type=float, help='Parameter controlling the contribution of edge reconstruction in the loss function.')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout.')
    parser.add_argument('--gradient_clipping', default=3.0, type=float, help='gradient clipping')
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Load HARR Dataset.
    dataset = load_data('hhar')
    G = load_graph('hhar', 5)
    X = dataset.x
    Label = dataset.y
    # prepare the data
    G_tf, S, R = prepare_graph_data(G)
    G_tf2 = G_tf
    X2 = sio.loadmat('./data/X2.mat')
    X2_dict = dict(X2)
    X2 = X2_dict['X2']
    S2 = S
    R2 = R
    # add feature dimension size to the beginning of hidden_dims
    feature_dim1 = X.shape[1]
    args.hidden_dims_1 = [feature_dim1] + args.hidden_dims_1
    feature_dim2 = X2.shape[1]
    args.hidden_dims_2 = [feature_dim2] + args.hidden_dims_2
    args.num_nodes = X.shape[0]
    print('Dim_hidden_1: ' + str(args.hidden_dims_1))
    print('Dim_hidden_2: ' + str(args.hidden_dims_2))
    trainer = Trainer(args)
    trainer(G_tf, G_tf2, X, X2, S, S2, R, R2, Label)
