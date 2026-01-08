import os, h5py, argparse
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from main import gene_selection, read_dataset, normalize, scMultiCluster

class ClustDistLayer(nn.Module):
    def __init__(self, centroids, n_clusters, cluster_list, device):
        super(ClustDistLayer, self).__init__()
        self.centroids = torch.tensor(centroids, dtype=torch.float32).to(device)
        self.n_clusters = n_clusters; self.cluster_list = cluster_list

    def forward(self, x, curr_clust_id):
        output = []
        for i in self.cluster_list:
            if i==curr_clust_id: continue
            weight = 2 * (self.centroids[self.cluster_list.index(curr_clust_id)] - self.centroids[self.cluster_list.index(i)])
            bias = torch.norm(self.centroids[self.cluster_list.index(curr_clust_id)], p=2) - torch.norm(self.centroids[self.cluster_list.index(i)], p=2)
            h = torch.matmul(x, weight.T) + bias
            output.append(h.unsqueeze(1))
        return torch.cat(output, dim=1) # (n, n_clusters-1)

class ClustMinPoolLayer(nn.Module):
    def __init__(self, beta):
        super(ClustMinPoolLayer, self).__init__()
        self.beta = beta
        self.eps = 1e-10

    def forward(self, inputs): # inputs: (n, n_clusters-1)
        return - torch.log(torch.sum(torch.exp(inputs * -self.beta), dim=1) + self.eps) # (n,)

class LRP(nn.Module):
    """ Adversarial clustering explanation
    Reference: https://proceedings.mlr.press/v139/lu21e/lu21e.pdf
    """
    def __init__(self, model, X1, X2, Z, cluster_ids, n_clusters, beta=1.0):
        super(LRP, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # 'cuda' or 'cpu'
        self.model = model; self.freeze_model() # freeze the model parameters
        self.cluster_ids = cluster_ids; self.n_clusters = n_clusters; self.cluster_list = np.unique(cluster_ids).astype(int).tolist()
        self.centroids_ = torch.tensor(self.set_centroids(Z), dtype=torch.float32)
        self.X1_ = torch.tensor(X1, dtype=torch.float32); self.X2_ = torch.tensor(X2, dtype=torch.float32); self.Z_ = torch.tensor(Z, dtype=torch.float32)
        self.distLayer = ClustDistLayer(self.centroids_, n_clusters, self.cluster_list, self.device).to(self.device)
        self.clustMinPool = ClustMinPoolLayer(beta).to(self.device)
        
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_centroids(self, Z):
        centroids = []
        for i in self.cluster_list:
            clust_Z = Z[self.cluster_ids==i]
            curr_centroid = np.mean(clust_Z, axis=0)
            centroids.append(curr_centroid)
        return np.stack(centroids, axis=0) # shape: (n_clusters, d)

    def clust_minpoolAct(self, X1, X2, curr_clust_id):
        z, _, _, _, _, _, _, _, _ = self.model.forward(X1, X2)
        return self.clustMinPool(self.distLayer(z, curr_clust_id))

    def calc_carlini_wagner_one_vs_one(self, clust_c_id, clust_k_id, margin=1., lamda=1e2, max_iter=5000, lr=2e-3, use_abs=True):
        X1_0 = torch.tensor(self.X1_[self.cluster_ids==clust_c_id], dtype=torch.float32).to(self.device) # shape: (n_cluster_c, d1)
        curr_X1 = torch.tensor(X1_0 + 1e-6, dtype=torch.float32, requires_grad=True).to(self.device) # shape: (n_cluster_c, d1)
        X2_0 = torch.tensor(self.X2_[self.cluster_ids==clust_c_id], dtype=torch.float32).to(self.device) # shape: (n_cluster_c, d2)
        curr_X2 = torch.tensor(X2_0 + 1e-6, dtype=torch.float32, requires_grad=True).to(self.device) # shape: (n_cluster_c, d2)
        optimizer = optim.SGD([curr_X1, curr_X2], lr=lr)
        for iter in range(max_iter):
            clust_c_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_c_id)
            clust_k_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_k_id)
            clust_loss_tensor = margin + clust_c_minpoolAct_tensor - clust_k_minpoolAct_tensor
            clust_loss_tensor = torch.maximum(clust_loss_tensor, torch.zeros_like(clust_loss_tensor))
            clust_loss = torch.sum(clust_loss_tensor)
            norm_loss = torch.norm(curr_X1 - X1_0, p=1) + torch.norm(curr_X2 - X2_0, p=1)
            loss = clust_loss * lamda + norm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iteration {}, Total loss:{:.8f}, clust loss:{:.8f}, L1 penalty:{:.8f}'.format(iter, loss.item(), clust_loss.item(), norm_loss.item())) if (iter+1) % 50 == 0 else None
        rel_score1 = torch.mean(torch.abs(curr_X1 - X1_0), dim=0) if use_abs else torch.mean(curr_X1 - X1_0, dim=0)
        rel_score2 = torch.mean(torch.abs(curr_X2 - X2_0), dim=0) if use_abs else torch.mean(curr_X2 - X2_0, dim=0)
        return rel_score1.data.cpu().numpy(), rel_score2.data.cpu().numpy()

    def calc_carlini_wagner_one_vs_rest(self, clust_c_id, margin=1., lamda=1e2, max_iter=5000, lr=2e-3, use_abs=True):
        X1_0 = torch.tensor(self.X1_[self.cluster_ids==clust_c_id], dtype=torch.float32).to(self.device) # shape: (n_cluster_c, d1)
        curr_X1 = torch.tensor(X1_0 + 1e-6, dtype=torch.float32, requires_grad=True).to(self.device) # shape: (n_cluster_c, d1)
        X2_0 = torch.tensor(self.X2_[self.cluster_ids==clust_c_id], dtype=torch.float32).to(self.device) # shape: (n_cluster_c, d2)
        curr_X2 = torch.tensor(X2_0 + 1e-6, dtype=torch.float32, requires_grad=True).to(self.device) # shape: (n_cluster_c, d2)
        optimizer = optim.SGD([curr_X1, curr_X2], lr=lr) # shape: (n, d1), (n, d2)
        for iter in range(max_iter):
            clust_rest_minpoolAct_tensor_list = []
            for clust_k_id in self.cluster_list:
                if clust_k_id == clust_c_id: continue
                clust_k_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_k_id) # shape: (n_cluster_c,)
                clust_rest_minpoolAct_tensor_list.append(clust_k_minpoolAct_tensor)
            clust_rest_minpoolAct_tensor = clust_rest_minpoolAct_tensor_list[0]
            for clust_k_id in range(1, len(clust_rest_minpoolAct_tensor_list)):
                clust_rest_minpoolAct_tensor = torch.maximum(clust_rest_minpoolAct_tensor, clust_rest_minpoolAct_tensor_list[clust_k_id])
            clust_c_minpoolAct_tensor = self.clust_minpoolAct(curr_X1, curr_X2, clust_c_id)
            clust_loss_tensor = margin + clust_c_minpoolAct_tensor - clust_rest_minpoolAct_tensor
            clust_loss_tensor = torch.maximum(clust_loss_tensor, torch.zeros_like(clust_loss_tensor))
            clust_loss = torch.sum(clust_loss_tensor)
            norm_loss = torch.norm(curr_X1 - X1_0, p=1) + torch.norm(curr_X2 - X2_0, p=1)
            loss = clust_loss * lamda + norm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iteration {}, Total loss:{:.8f}, clust loss:{:.8f}, L1 penalty:{:.8f}'.format(iter, loss.item(), clust_loss.item(), norm_loss.item())) if (iter+1) % 50 == 0 else None
        rel_score1 = torch.mean(torch.abs(curr_X1 - X1_0), dim=0) if use_abs else torch.mean(curr_X1 - X1_0, dim=0)
        rel_score2 = torch.mean(torch.abs(curr_X2 - X2_0), dim=0) if use_abs else torch.mean(curr_X2 - X2_0, dim=0)
        return rel_score1.data.cpu().numpy(), rel_score2.data.cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='./data/CITESeq_GSE128639_BMNC_anno.h5')
    parser.add_argument('--pretrain_weight_file', default='None', type=str)
    parser.add_argument('--save_dir', default='./result/')
    parser.add_argument('--run', default=1, type=int) # the number of runs
    parser.add_argument('-el','--encode_layer', nargs='+', default=[256,64,32,16])
    parser.add_argument('-dl1','--decode_layer_1', nargs='+', default=[16,64,256])
    parser.add_argument('-dl2','--decode_layer_2', nargs='+', default=[16,20])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--gamma', default=0.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float, help='fuzziness of clustering loss')          
    parser.add_argument('--sigma_1', default=2.5, type=float)
    parser.add_argument('--sigma_2', default=1.5, type=float)          
    parser.add_argument('--phi_1', default=0.001, type=float, help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi_2', default=0.001, type=float, help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--no_labels', action='store_true', default=False)
    parser.add_argument('--filter_1', action='store_true', default=False, help='Do mRNA selection')
    parser.add_argument('--filter_2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--filter_1_num', default=1000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--filter_2_num', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    # LRP parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--beta', default=1.0, type=float, help='coefficient of the clustering fuzziness')
    parser.add_argument('--margin', default=1.0, type=float, help='margin of difference between logits')
    parser.add_argument('--lamda', default=100.0, type=float, help='coefficient of the clustering perturbation loss')
    parser.add_argument('--cluster_index_file', default='label.txt')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1\ preprocessing scRNA-seq read counts matrix
    with h5py.File(args.data_file, 'r') as f:
        x1 = np.array(f['X1']); x2 = np.array(f['X2']); y = np.array(f['Y']) if not args.no_labels else None # y = np.array(data_mat['Y']) - 1
    x1 = x1[:, gene_selection(x1, n=args.filter_1_num, plot=False)] if args.filter_1 else x1
    x2 = x2[:, gene_selection(x2, n=args.filter_2_num, plot=False)] if args.filter_2 else x2
    cluster_ids = np.loadtxt(args.cluster_index_file, delimiter=",").astype(int)
    adata1 = sc.AnnData(x1); # adata1.obs['Group'] = y
    adata1 = read_dataset(adata1, transpose=False, test_split=False, copy=True)
    adata1 = normalize(adata1, size_factors=True, normalize_input=True, logtrans_input=True)
    adata2 = sc.AnnData(x2); # adata2.obs['Group'] = y
    adata2 = read_dataset(adata2, transpose=False, test_split=False, copy=True)
    adata2 = normalize(adata2, size_factors=True, normalize_input=True, logtrans_input=True)

    # 2\ load pre-trained model
    model = scMultiCluster(input_dim_1=adata1.n_vars, input_dim_2=adata2.n_vars, encode_layer=args.encode_layer, decode_layer_1=args.decode_layer_1, decode_layer_2=args.decode_layer_2,
                           activation='elu', tau=args.tau, sigma_1=args.sigma_1, sigma_2=args.sigma_2, gamma=args.gamma, phi_1=args.phi_1, phi_2=args.phi_2, cutoff = args.cutoff, save_dir=args.save_dir)
    assert os.path.exists(args.pretrain_weight_file), "==> no checkpoint found at '{}'".format(args.pretrain_weight_file)
    print("==> loading checkpoint '{}'".format(args.pretrain_weight_file))
    model.load_state_dict(torch.load(args.pretrain_weight_file)['model_state_dict'])

    # 3\ LRP (Layer-wise Relevance Propagation)
    n_clusters = len(np.unique(cluster_ids))
    cluster_list = np.unique(cluster_ids).astype(int).tolist()
    print('n cluster is: ' + str(n_clusters))
    Z = model.encode(torch.tensor(adata1.X, dtype=torch.float32).to(args.device), torch.tensor(adata2.X, dtype=torch.float32).to(args.device)).data.cpu().numpy()
    model_explainer = LRP(model, X1=adata1.X, X2=adata2.X, Z=Z, cluster_ids=cluster_ids, n_clusters=n_clusters, beta=args.beta).to(args.device)
    for clust_c in cluster_list:
        print('Cluster' + str(clust_c) + 'vs Rest')
        rel_score1, rel_score2 = model_explainer.calc_carlini_wagner_one_vs_rest(clust_c, margin=args.margin, lamda=args.lamda, max_iter=args.max_iter, lr=args.lr)
        np.savetxt(args.save_dir + '/' + str(clust_c) + '_vs_rest_rel_mRNA_scores.csv', rel_score1, delimiter=',')
        np.savetxt(args.save_dir + '/' + str(clust_c) + '_vs_rest_rel_ADT_scores.csv', rel_score2, delimiter=',')
        