import torch
import torch.nn as nn
import torch.nn.functional as F
import os, glob, math, argparse, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
        
class GraphConvolution(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight) # node feature transformation
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(G, x) # graph convolution
        return x

def count_gcn_flops(module, input, output):
    """
    m: GraphConvolution module (sparse matrix)
    x: tuple(inputs) -> (x, adj)
    y: output tensor (N, F_out)
    # count the MACs, including: (the Multiplication and Addition)
    # MACs = 2 flops (multiply + add), there we calculate the MACs
    # 1) support = x @ W: N * F_in * F_out
    # 2) output = adj @ support (sparse-dense): nnz(adj) * F_out
    # 3) bias addition: N * F_out (if exists)
    """
    x, G = input # N: node number, F_in: input feature dimension
    N, F_in = x.shape
    F_out = output.shape[1] # output feature dimension
    # 1) 1. Linear Transformation: H * W + Bias / dense matmul x @ W
    macs_dense = N * F_in * F_out
    # 2) 2. Graph Convolution: D^(-1/2) * H * W * D^(-1/2) / sparse mm adj @ support
    ### torch.sparse.mm is a sparse matrix multiplication function, which is faster than torch.matmul for sparse matrices.
    # if G.layout is torch.sparse_coo or str(G.layout).startswith("torch.sparse"): # sparse tensor
    #     nnz = int(G._nnz()) # number of non-zero elements
    # else:
    #     nnz = int((G != 0).sum().item()) if not G.is_sparse else int(G._nnz()) # dense tensor, count the non-zero elements
    # macs_graphconv = nnz * F_out
    # ### torch.mm is a dense matrix multiplication function, which is faster than torch.sparse.mm for dense matrices.
    macs_graphconv = N * N * F_out
    # 3) 3. Bias Addition: Bias / bias add
    # bias_ops = (N * F_out) if (m.bias is not None) else 0
    # total_macs = macs_dense + macs_sparse + bias_ops
    MACs = macs_dense + macs_graphconv
    module.total_ops += MACs
    
class GCN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=None):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hid, n_out)

    def forward(self, input, adj):
        x = self.dp1(F.elu(self.gc1(input, adj)))
        x = self.dp2(F.elu(self.gc2(x, adj)))
        x = self.fc(x)
        return x
    
def load_data(data_dir, threshold=0.005):
    adj = pd.read_csv(os.path.join(data_dir, 'SNF_fused_matrix.csv'), header=0, index_col=0).values
    fea = pd.read_csv(os.path.join(data_dir, 'latent_data.csv'), header=0, index_col=None).values
    label_train = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv'), header=None, index_col=None).values
    label_test = pd.read_csv(os.path.join(data_dir, 'labels_te.csv'), header=None, index_col=None).values
    label = np.concatenate([label_train, label_test], axis=0)
    print('Calculating the laplace adjacency matrix...')
    ## The SNF matrix is a completed connected graph, it is better to filter edges with a threshold
    adj[adj<threshold] = 0 # Including self-loop
    adj = (adj != 0) * 1.0 
    # np.savetxt('result/adjacency_matrix.csv', exist, delimiter=',', fmt='%d')
    ## Calculate the degree matrix
    degree_matrix = np.diag(np.sum(adj, axis=1))
    # np.savetxt('result/diag.csv', diag_matrix, delimiter=',', fmt='%d')
    ## Calculate the adj_hat, D^{-1/2} (A+I) D^{-1/2} or D^{-1} A
    adj_hat = np.linalg.inv(degree_matrix).dot(adj) # adj_hat = adj / np.sum(degree_matrix, axis=1)[:, None]
    #########################################################
    # deg = adj.sum(axis=1)
    # d_inv_sqrt = np.power(deg, -0.5, where=deg>0)
    # d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    # adj_hat = (d_inv_sqrt[:, None] * A) * d_inv_sqrt[None, :]
    return adj_hat, fea, label

def train(epoch, optimizer, features, adj, labels, idx_train):
    GCN_model.train()
    optimizer.zero_grad()
    output = GCN_model(features, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train]) # output_shape: (N, num_class), labels_shape: (N,)
    acc_train = accuracy_score(labels[idx_train].detach().cpu().numpy(), output[idx_train].detach().cpu().numpy().argmax(1))
    loss_train.backward()
    optimizer.step()
    print('Epoch: %.2f | loss train: %.4f | acc train: %.4f' %(epoch+1, loss_train.item(), acc_train.item())) if epoch % 10 == 0 else None
    return loss_train.data.item()

def test(features, adj, labels, idx_test):
    GCN_model.eval()
    output = GCN_model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test]) # output_shape: (N, num_class), labels_shape: (N,)
    ot = output[idx_test].detach().cpu().numpy().argmax(1)
    lb = labels[idx_test].detach().cpu().numpy()
    acc_test = accuracy_score(lb, ot)
    f = f1_score(ot, lb, average='weighted')
    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item(), f
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/BRCA/', help='The output dir.')
    parser.add_argument('-m', '--mode', type=int, choices=[0,1], default=1, help='mode 0: 10-fold cross validation; mode 1: train and test a model.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='Training epochs, default: 150.')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('-w', '--weight_decay', type=float, default=0.01, help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.01')
    parser.add_argument('-hd', '--hidden', type=int, default=64, help='Hidden layer dimension, default: 64.')
    parser.add_argument('-dp', '--dropout', type=float, default=0.5, help='Dropout rate, methods to avoid overfitting, default: 0.5.')
    parser.add_argument('-t', '--threshold', type=float, default=0.005, help='Threshold to filter edges, default: 0.005')
    parser.add_argument('-nc', '--nclass', type=int, default=5, help='Number of classes, default: 4')
    parser.add_argument('-p', '--patience', type=int, default=20, help='Patience')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    adj, data, label = load_data(args.data_dir, args.threshold)
    print('adj shape:', adj.shape)
    print('data shape:', data.shape)
    print('label shape:', label.shape)
    adj = torch.tensor(adj, dtype=torch.float, device=device)
    features = torch.tensor(data, dtype=torch.float, device=device)
    labels = torch.tensor(label.squeeze(), dtype=torch.long, device=device)
    if args.mode == 0:
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        acc_res, f1_res = [], []
        for idx_train, idx_test in skf.split(data, label):
            GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)
            GCN_model.to(device)
            optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)
            idx_train, idx_test= torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test, dtype=torch.long, device=device)
            for epoch in range(args.epochs):
                train(epoch, optimizer, features, adj, labels, idx_train)
            ac, f1= test(features, adj, labels, idx_test)
            acc_res.append(ac); f1_res.append(f1)
        print('10-fold  Acc(%.4f, %.4f)  F1(%.4f, %.4f)' % (np.mean(acc_res), np.std(acc_res), np.mean(f1_res), np.std(f1_res)))
    elif args.mode == 1:
        GCN_model = GCN(n_in=features.shape[1], n_hid=args.hidden, n_out=args.nclass, dropout=args.dropout)
        GCN_model.to(device)
        # print the FLOPs and params
        from thop import profile, clever_format
        params = sum([sum(p.numel() for p in GCN_model.parameters())])
        flops, params = profile(GCN_model.to(device), inputs=(features, adj), verbose=False, custom_ops={GraphConvolution: count_gcn_flops}) # return MACs and params; flops = 2 * MACs
        flops, params = clever_format([flops*2, params], "%.3f")
        print(f'Input data shape: {features.shape}')
        print(f'Input graph shape: {adj.shape}')
        print(f"FLOPs: {flops}, Params: {params}")
        print(f"Params by manual calculation: {sum(p.numel() for p in GCN_model.parameters())}")
        print('Begin training model...')
        optimizer = torch.optim.Adam(GCN_model.parameters(), lr=args.learningrate, weight_decay=args.weight_decay)
        idx_train = list(range(612)); idx_test = list(range(612, 812))
        idx_train = torch.tensor(idx_train, dtype=torch.long, device=device)
        idx_test = torch.tensor(idx_test, dtype=torch.long, device=device)
        loss_values = []
        bad_counter, best_epoch = 0, 0
        best = 1000
        for epoch in range(args.epochs):
            loss_values.append(train(epoch, optimizer, features, adj, labels, idx_train))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1
            if bad_counter == args.patience:
                break
            torch.save(GCN_model.state_dict(), os.path.join(args.output_dir, 'GCN_{}.pth'.format(epoch)))
            files = glob.glob(os.path.join(args.output_dir, 'GCN_*.pth'))
            for file in files:
                name = file.split('/')[-1]
                epoch_nb = int(name.split('_')[1].split('.')[0])
                if epoch_nb != best_epoch:
                    os.remove(file)
        print('Training finished.')
        print('The best epoch model is ',best_epoch)
        GCN_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'GCN_{}.pth'.format(best_epoch))))
        test(features, adj, labels, idx_test)
        

