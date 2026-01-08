import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist, pdist

def construct_hyperedge_with_KNN(X, k_list=[10], is_prob=True, m_prob=1, edge_type='euclid'):
    hyperedge_mat_list = []
    for k in k_list:
        if edge_type == 'euclid':
            distance_mat = cdist(X, X, 'euclid')
            hyperedge_mat = np.zeros(distance_mat.shape)
            for center_idx in range(distance_mat.shape[0]):
                distance_mat[center_idx, center_idx] = 0
                distance_vec = distance_mat[center_idx]
                distance_vec_avg = np.average(distance_vec)
                nearest_idx = np.array(np.argsort(distance_vec)).squeeze()
                nearest_idx[k - 1] = center_idx if not np.any(nearest_idx[:k] == center_idx) else nearest_idx[k - 1] # add the center node to the nearest neighbors if it is not in the top k
                for node_idx in nearest_idx[:k]:
                    hyperedge_mat[node_idx, center_idx] = np.exp(-distance_vec[node_idx] ** 2 / (m_prob * distance_vec_avg) ** 2) if is_prob else 1.0
        elif edge_type == 'pearson':
            distance_mat = pd.DataFrame(X.T).corr(method='pearson').to_numpy()
            hyperedge_mat = np.zeros(distance_mat.shape)
            for center_idx in range(distance_mat.shape[0]):
                distance_mat[center_idx, center_idx] = -999
                distance_vec = distance_mat[center_idx]
                distance_vec_avg = np.average(distance_vec) 
                nearest_idx = np.array(np.argsort(distance_vec, order='desc')).squeeze()
                nearest_idx[k - 1] = center_idx if not np.any(nearest_idx[:k] == center_idx) else nearest_idx[k - 1] # add the center node to the nearest neighbors if it is not in the top k
                for node_idx in nearest_idx[:k]:
                    hyperedge_mat[node_idx, center_idx] = 1-np.exp(-(distance_vec[node_idx] + 1.0) ** 2 ) if is_prob else 1.0
        hyperedge_mat_list.append(hyperedge_mat)
    return np.hstack(hyperedge_mat_list)

def load_feature_and_hyperedge(data_dir_rna, data_dir_adt, data_dir_atac, label_dir, use_rna=True, use_adt=True, use_atac=True, k_list=[10], is_prob=True, m_prob=1, edge_type='pearson'):
    # Load the multi-omics data and concatenate the modality-specific features.
    feature_rna = np.load(data_dir_rna) if use_rna else None
    feature_adt = np.load(data_dir_adt) if use_adt else None
    feature_atac = np.load(data_dir_atac) if use_atac else None
    label = np.load(label_dir)
    feature_multi_omics = None
    for feature in [feature_rna, feature_adt, feature_atac]:
        if feature is not None:
            feature_multi_omics = feature if feature_multi_omics is None else np.hstack((feature_multi_omics, feature))
    # Construct the multi-omics hypergraph incidence matrix and concatenate the modality-specific hyperedges.
    hyperedge_rna = construct_hyperedge_with_KNN(feature_rna, k_list=k_list, is_prob=is_prob, m_prob=m_prob, edge_type=edge_type) if use_rna else None
    hyperedge_adt = construct_hyperedge_with_KNN(feature_adt, k_list=k_list, is_prob=is_prob, m_prob=m_prob, edge_type=edge_type) if use_adt else None
    hyperedge_atac = construct_hyperedge_with_KNN(feature_atac, k_list=k_list, is_prob=is_prob, m_prob=m_prob, edge_type=edge_type) if use_atac else None
    hyperedge_multi_omics = None
    for hyperedge in [hyperedge_rna, hyperedge_adt, hyperedge_atac]:
        if hyperedge is not None:
            hyperedge_multi_omics = hyperedge if hyperedge_multi_omics is None else np.hstack((hyperedge_multi_omics, hyperedge))
            
    print('multi-omics feature shape:', feature_multi_omics.shape)
    print('multi-omics hyperedge shape:', hyperedge_multi_omics.shape)
    print('label shape:', label.shape)
    return feature_multi_omics, label, hyperedge_multi_omics, hyperedge_rna, hyperedge_adt, hyperedge_atac

def generate_G_from_H(H, variable_weight=False):
    # Calculate G from hypgerraph incidence matrix H, where G = DV2 * H * W * invDE * HT * DV2
    H = np.array(H) # shape: N X M, N is the number of nodes, M is the number of hyperedges
    W = np.ones(H.shape[1]) # the weight of the hyperedge
    DV = np.sum(H * W, axis=1) # the degree of the node
    DE = np.sum(H, axis=0) # the degree of the hyperedge
    invDE = np.mat(np.diag(np.power(DE, -1))) # shape: M X M
    invDV2 = np.mat(np.diag(np.power(DV, -0.5))) # shape: N X N
    W = np.mat(np.diag(W)) # shape: M X M
    H = np.mat(H) # shape: N X M
    if variable_weight:
        return invDV2 * H, W, invDE * H.T * invDV2
    else:
        return invDV2 * H * W * invDE * H.T * invDV2 # shape: N X N
    
def neighbor_sampling(H, positive_neighbor_num, p):
    # Given a dense incidence matrix and a sample num (positive_neighbor_num * p), return a sampled coordinate array
    coor = np.vstack((np.nonzero(H)))
    indices = list(range(coor.shape[1]))
    random.shuffle(indices)
    return coor[:,indices[:int(positive_neighbor_num * p)]]

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.register_parameter('bias', None) if not bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # x.shape: N X in_ft, G.shape: N X N
        x = x.matmul(self.weight) # shape: N X out_ft
        x = x + self.bias if self.bias is not None else x
        x = G.matmul(x) # shape: N X out_ft
        return x

class HGNN_unsupervised(nn.Module):
    def __init__(self, in_ch, dim_hid, dropout):
        super(HGNN_unsupervised, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, dim_hid)
        self.hgc2 = HGNN_conv(dim_hid, dim_hid)
        self.mlp1 = nn.Linear(in_ch, dim_hid)
        self.mlp2 = nn.Linear(dim_hid, dim_hid)

    def forward(self, x, G):
        x_pos = self.mlp2(F.dropout(F.relu(self.mlp1(x)), self.dropout)) # not consider the hyperedge information
        x_ach = self.hgc2(F.dropout(F.relu(self.hgc1(x, G)), self.dropout), G) # consider the hyperedge information
        x_neg = x_pos[torch.randperm(x_pos.size()[0])] # shuffle the positive samples
        return x_ach, x_pos, x_neg
    
    @staticmethod
    def intra_cell_loss(x_ach, x_pos, x_neg, beta, size_average=True): 
        # contrastive loss 1: intra-cell contrastive loss, takes embeddings of an anchor sample, a positive sample and a negative sample
        # Reference: Triplet Loss, TFeat, BMVC 2016
        distance_positive = (x_ach - x_pos).pow(2).sum(1)
        distance_negative = (x_ach - x_neg).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + beta) # The target is to make distance_positive <= distance_negative - beta, beta is the margin.
        return losses.mean() if size_average else losses.sum()
    
    @staticmethod
    def inter_cell_loss(x_ach, H_union, H_none, tau):
        # contrastive loss 2: inter-cell contrastive loss, pull similar node pairs closer and push dissimilar node pairs apart
        f = lambda x: torch.exp(x / tau)
        x_ach = F.normalize(x_ach)
        sim_mat = f(x_ach @ x_ach.t())
        neighbor_sim_mat = H_union * sim_mat
        none_neighbor_sim_mat = H_none * sim_mat
        loss = -torch.log(neighbor_sim_mat.sum() / none_neighbor_sim_mat.sum())
        return loss

class HGNN_supervised(nn.Module):
    def __init__(self, in_ch, n_class, dim_hid, dropout=0.5):
        super(HGNN_supervised, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, dim_hid)
        self.hgc2 = HGNN_conv(dim_hid, dim_hid)
        self.hgc3 = HGNN_conv(dim_hid, n_class)

    def forward(self, x, G):
        x = F.dropout(F.relu(self.hgc1(x, G)), self.dropout)
        x = F.dropout(F.relu(self.hgc2(x, G)), self.dropout)
        x = self.hgc3(x, G)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='sim1', help='dataset name')
    parser.add_argument('--data_dir_rna', default='./data/simulation_dataset_1/rna.npy', help='path of RNA data')
    parser.add_argument('--data_dir_adt', default='./data/simulation_dataset_1/adt.npy', help='path of ADT data')
    parser.add_argument('--data_dir_atac', default='./data/simulation_dataset_1/atac.npy', help='path of ATAC data')
    parser.add_argument('--label_dir',  default='./data/simulation_dataset_1/lbls.npy', help='path of cell label data')
    parser.add_argument("--use_rna", type=bool, default=1, help='use rna modality')
    parser.add_argument("--use_adt", type=bool, default=1, help='use adt modality')
    parser.add_argument("--use_atac", type=bool, default=1, help='use atac modality')
    parser.add_argument("--k_neighbors", type=int, default=70, help='k_neighbors for hypergraph construction')  
    parser.add_argument("--is_prob", type=bool, default=False, help='prob edge True or False')
    parser.add_argument("--m_prob", type=float, default=1.0, help='m_prob')
    parser.add_argument("--edge_type", type=str, default='euclid', help='euclid or pearson')
    parser.add_argument("--p_trible", type=float, default=0.8, help='sample probability for trible-neighbor set')
    parser.add_argument("--p_double", type=float, default=0.15, help='sample probability for double-neighbor set')
    parser.add_argument("--p_single", type=float, default=0.05, help='sample probability for single-neighbor set')
    parser.add_argument("--positive_neighbor_num", type=int, default=1000, help='num of node pairs in positive neighbors set')
    parser.add_argument("--supervised", type=bool, default=False, help='True for stage 2 (cell type annotation), False for stage1 (unsupervised cell representation learning)')
    parser.add_argument("--dim_hid", type=int, default=128, help='dimension of hidden layer')   
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout rate')
    parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
    parser.add_argument("--milestones", type=int, default=[100], help='milestones')
    parser.add_argument("--gamma", type=float, default=0.9, help='gamma')
    parser.add_argument("--weight_decay", type=float, default=0.0005, help='weight_decay')
    parser.add_argument("--max_epoch", type=int, default=200, help='max_epoch')
    parser.add_argument("--tau", type=float, default=0.5, help='temperature Coefficient')
    parser.add_argument("--beta", type=float, default=100., help='non-negative control parameter for intra_cell_loss')
    parser.add_argument("--alpha", type=float, default=0.05, help='balanced factor for dual contrastive loss')
    parser.add_argument("--labeled_cell_ratio", type=float, default=0.02, help='labeled cell ratio for cell type annotation')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_multi_omics, label_multi_omics, H, H_rna, H_adt, H_atac = load_feature_and_hyperedge(
        args.data_dir_rna, args.data_dir_adt, args.data_dir_atac, args.label_dir, args.use_rna, args.use_adt, args.use_atac, 
        k_list=[args.k_neighbors], is_prob=args.is_prob, m_prob=args.m_prob, edge_type=args.edge_type)
    np.fill_diagonal(H_rna, 0); np.fill_diagonal(H_adt, 0); np.fill_diagonal(H_atac, 0) # remove self-loop
    H_rna = np.where(H_rna, 1, 0); H_adt = np.where(H_adt, 1, 0); H_atac = np.where(H_atac, 1, 0) # binarize the hyperedge
    H_all = H_rna + H_adt + H_atac; H_trible = np.where(H_all==3, 1, 0); H_double = np.where(H_all==2, 1, 0); H_single = np.where(H_all==1, 1, 0); H_none = np.where(H_all==0, 1,0)
    G = generate_G_from_H(H) # G = DV2 * H * W * invDE * HT * DV2, shape: N X N (H.shape: N X M, N is the number of nodes, M is the number of hyperedges)
    N = feature_multi_omics.shape[0] # cell number
    N_class = int(label_multi_omics.max() + 1) # class number
    
    feature_multi_omics = torch.Tensor(feature_multi_omics).to(device)
    label_multi_omics = torch.Tensor(label_multi_omics).squeeze().long().to(device)
    G = torch.Tensor(G).to(device) # shape: N X N
    if not args.supervised:
        print('Stage 1: unsupervised cell representation learning')
        model = HGNN_unsupervised(in_ch=feature_multi_omics.shape[1], dim_hid=args.dim_hid, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.milestones,gamma=args.gamma)
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            assert args.p_trible + args.p_double + args.p_single == 1, 'the sum of p_trible, p_double and p_single should be 1'
            coor_sampled_trible = neighbor_sampling(H_trible, args.positive_neighbor_num, args.p_trible)
            coor_sampled_double = neighbor_sampling(H_double, args.positive_neighbor_num, args.p_double)
            coor_sampled_single = neighbor_sampling(H_single, args.positive_neighbor_num, args.p_single)
            coor_sampled = np.hstack((coor_sampled_trible, coor_sampled_double, coor_sampled_single))
            H_union_sampled = torch.from_numpy(coo_matrix((np.ones(coor_sampled.shape[1]), (coor_sampled[0,:], coor_sampled[1,:])), shape=(N, N)).toarray()).to(device) # shape: N X N
            H_none_all = torch.from_numpy(H_none).to(device)
            x_ach, x_pos, x_neg = model(feature_multi_omics, G)
            loss_intra_cell = model.intra_cell_loss(x_ach, x_pos, x_neg, args.beta)
            loss_inter_cell = model.inter_cell_loss(x_ach, H_union_sampled, H_none_all, args.tau)
            loss = loss_intra_cell + args.alpha * loss_inter_cell 
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.save(model, './result/pretrained.pt')
            print('Epoch {}/{}: Loss: {:.4f} Loss_intra_cell: {:.4f} Loss_inter_cell: {:.4f}'.format(epoch, args.max_epoch - 1, loss.item(), loss_intra_cell.item(), loss_inter_cell.item())) if epoch % 5 == 0 else None
        model.eval()
        embedding, _, _ = model(feature_multi_omics, G)
        np.save('./result/embedding.npy', embedding.cpu().detach().numpy())
    else:
        print('Stage 2: supervised training for cell type annotation')
        model = HGNN_supervised(in_ch=feature_multi_omics.shape[1], n_class=N_class, dim_hid=args.dim_hid, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        criterion = torch.nn.CrossEntropyLoss()
        # the index of the samples that must be included in the labeled set, to ensure at least one sample from each class
        must_include_idx = [np.random.choice(np.where(label_multi_omics.cpu().numpy()==i)[0]) for i in range(N_class)]
        indices_subset = list(range(N))
        random.shuffle(indices_subset)
        idx_train = torch.Tensor(list(set(indices_subset[:int(N*args.labeled_cell_ratio-N_class)]).union(set(must_include_idx)))).long().to(device)
        idx_test = torch.Tensor(list(set(indices_subset[int(N*args.labeled_cell_ratio-N_class):]).difference(set(must_include_idx)))).long().to(device)
        assert os.path.exists('./result/pretrained.pt'), 'should first pretrained model in unsupervised manner with args.supervised 0'
        pretrained_dict = torch.load('./result/pretrained.pt').state_dict() 
        model_dict = model.state_dict()
        model_dict.update({k: v for k, v in pretrained_dict.items() if k in model_dict})
        model.load_state_dict(model_dict)
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            output = model(feature_multi_omics, G)
            loss = criterion(output[idx_train], label_multi_omics[idx_train])
            pred = torch.argmax(output, 1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss = loss.item()
            epoch_acc = torch.sum(pred[idx_train] == label_multi_omics[idx_train]).item() / len(idx_train)
            print('Epoch {}/{}: Loss: {:.4f} Acc: {:.4f}'.format(epoch, args.max_epoch - 1, epoch_loss, epoch_acc)) if epoch % 5 == 0 else None
        model.eval()
        output = model(feature_multi_omics, G)
        pred = torch.argmax(output, 1)
        np.savez('./result/result_test.npz', label=label_multi_omics[idx_test].detach().cpu().numpy(), pred=pred[idx_test].detach().cpu().numpy())
