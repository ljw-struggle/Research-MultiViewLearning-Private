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
from _utils import MMDataset, overall_performance_report

def load_feature_and_hyperedge(data, label, k_list=[10], is_prob=True, m_prob=1):
    def construct_hyperedge_with_KNN(X, k_list=[10], is_prob=True, m_prob=1):
        hyperedge_mat_list = []
        for k in k_list:
            distance_mat = cdist(X, X, 'euclid')
            hyperedge_mat = np.zeros(distance_mat.shape)
            for center_idx in range(distance_mat.shape[0]):
                distance_mat[center_idx, center_idx] = 0
                distance_vec = distance_mat[center_idx]
                distance_vec_avg = np.average(distance_vec)
                nearest_idx = np.array(np.argsort(distance_vec)).squeeze()
                nearest_idx[k - 1] = center_idx if not np.any(nearest_idx[:k] == center_idx) else nearest_idx[k - 1] # add the center node to the nearest neighbors if it is not in the top k
                for node_idx in nearest_idx[:k]:
                    hyperedge_mat[node_idx, center_idx] = np.exp(-distance_vec[node_idx] ** 2 / (m_prob * distance_vec_avg) ** 2) if is_prob else 1.0 # Gaussian kernel for transforming euclidean distance to probability
            hyperedge_mat_list.append(hyperedge_mat)
        return np.hstack(hyperedge_mat_list)
    feature_multi_omics = np.hstack(data)
    hyperedge_omics_1 = construct_hyperedge_with_KNN(data[0], k_list=k_list, is_prob=is_prob, m_prob=m_prob)
    hyperedge_omics_2 = construct_hyperedge_with_KNN(data[1], k_list=k_list, is_prob=is_prob, m_prob=m_prob)
    hyperedge_omics_3 = construct_hyperedge_with_KNN(data[2], k_list=k_list, is_prob=is_prob, m_prob=m_prob)
    hyperedge_multi_omics = np.hstack((hyperedge_omics_1, hyperedge_omics_2, hyperedge_omics_3))
    return feature_multi_omics, label, hyperedge_multi_omics, hyperedge_omics_1, hyperedge_omics_2, hyperedge_omics_3

def generate_G_from_H(H):
    # Calculate G from hypgerraph incidence matrix H, where G = DV2 * H * W * invDE * HT * DV2
    H = np.array(H) # shape: N X M, N is the number of nodes, M is the number of hyperedges
    W = np.ones(H.shape[1]) # the weight of the hyperedge
    DV = np.sum(H * W, axis=1) # the degree of the node
    DE = np.sum(H, axis=0) # the degree of the hyperedge
    invDE = np.mat(np.diag(np.power(DE, -1))) # shape: M X M
    invDV2 = np.mat(np.diag(np.power(DV, -0.5))) # shape: N X N
    W = np.mat(np.diag(W)) # shape: M X M
    H = np.mat(H) # shape: N X M
    return invDV2 * H * W * invDE * H.T * invDV2 # shape: N X N
    
def neighbor_sampling(H, positive_neighbor_num, p):
    # Given a dense incidence matrix and a sample num (positive_neighbor_num * p), return a sampled coordinate array
    coor = np.vstack((np.nonzero(H))) # shape: 2 X M, M is the number of hyperedges
    indices = list(range(coor.shape[1])) # shape: M
    random.shuffle(indices) # shuffle the indices
    return coor[:,indices[:int(positive_neighbor_num * p)]] # shape: 2 X num_sampled, num_sampled is the number of sampled hyperedges

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # x.shape: N X in_ft, G.shape: N X N
        x = x.matmul(self.weight) # shape: N X out_ft
        x = x + self.bias if self.bias is not None else x
        x = G.matmul(x) # shape: N X out_ft
        return x

class scMHNN(nn.Module):
    def __init__(self, embed_dim=200, feature_dims=[1000, 1000, 500], dropout=0.1):
        super(scMHNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(sum(feature_dims), embed_dim)
        self.hgc2 = HGNN_conv(embed_dim, embed_dim)
        self.mlp1 = nn.Linear(sum(feature_dims), embed_dim)
        self.mlp2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, G):
        x_pos = self.mlp2(F.dropout(F.relu(self.mlp1(x)), self.dropout))
        x_ach = self.hgc2(F.dropout(F.relu(self.hgc1(x, G)), self.dropout), G)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_sc_multiomics/TEA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_sc_multiomics/DUCMME/TEA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', default=128, type=int, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', default=2000, type=int, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', default=[200], type=int, help='number of epochs to train [default: [200, 100, 100]]')
    parser.add_argument('--learning_rate', default=[1e-3], type=float, help='learning rate [default: [5e-3, 1e-3, 1e-3]]')
    parser.add_argument('--log_interval', default=10, type=int, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--update_interval', default=10, type=int, help='how many epochs to wait before updating cluster centers [default: 10]')
    parser.add_argument('--tol', default=1e-3, type=float, help='tolerance for convergence [default: 1e-3]')
    parser.add_argument('--times', default=5, type=int, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    for _ in range(args.times):
        dataset = MMDataset(args.data_dir, concat_data=False); data = [x.clone().numpy() for x in dataset.X]; label = dataset.Y.clone().numpy()
        data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
        feature_multi_omics, label_multi_omics, H, H_omics_1, H_omics_2, H_omics_3 = load_feature_and_hyperedge(data, label, k_list=[70], is_prob=False, m_prob=1.0)
        
        feature_multi_omics = torch.Tensor(feature_multi_omics).to(device)
        G = torch.Tensor(generate_G_from_H(H)).to(device) # shape: N X N
        np.fill_diagonal(H_omics_1, 0); np.fill_diagonal(H_omics_2, 0); np.fill_diagonal(H_omics_3, 0) # remove self-loop
        H_omics_1 = np.where(H_omics_1, 1, 0); H_omics_2 = np.where(H_omics_2, 1, 0); H_omics_3 = np.where(H_omics_3, 1, 0) # binarize the hyperedge
        H_all = H_omics_1 + H_omics_2 + H_omics_3; H_trible = np.where(H_all==3, 1, 0); H_double = np.where(H_all==2, 1, 0); H_single = np.where(H_all==1, 1, 0); H_none = np.where(H_all==0, 1,0)
        
        model = scMHNN(embed_dim=args.latent_dim, feature_dims=data_features, dropout=0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate[0], weight_decay=0.0005)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.9)
        for epoch in range(args.epoch_num[0]):
            model.train()
            optimizer.zero_grad()
            coor_sampled_trible = neighbor_sampling(H_trible, 1000, 0.8) # shape: 2 X num_sampled, num_sampled is the number of sampled hyperedges
            coor_sampled_double = neighbor_sampling(H_double, 1000, 0.15) # shape: 2 X num_sampled, num_sampled is the number of sampled hyperedges
            coor_sampled_single = neighbor_sampling(H_single, 1000, 0.05) # shape: 2 X num_sampled, num_sampled is the number of sampled hyperedges
            coor_sampled = np.hstack((coor_sampled_trible, coor_sampled_double, coor_sampled_single)) # shape: 2 X num_sampled, num_sampled is the number of sampled hyperedges
            H_union_sampled = torch.from_numpy(coo_matrix((np.ones(coor_sampled.shape[1]), (coor_sampled[0,:], coor_sampled[1,:])), shape=(data_samples, data_samples)).toarray()).to(device) # shape: N X N
            H_none_all = torch.from_numpy(H_none).to(device) # shape: N X N
            x_ach, x_pos, x_neg = model(feature_multi_omics, G)
            loss_intra_cell = model.intra_cell_loss(x_ach, x_pos, x_neg, 100.)
            loss_inter_cell = model.inter_cell_loss(x_ach, H_union_sampled, H_none_all, 0.5)
            loss = loss_intra_cell + 0.05 * loss_inter_cell 
            loss.backward()
            optimizer.step()
            scheduler.step()
            print('Epoch {}/{}: Loss: {:.4f} Loss_intra_cell: {:.4f} Loss_inter_cell: {:.4f}'.format(epoch, args.epoch_num[0] - 1, loss.item(), loss_intra_cell.item(), loss_inter_cell.item())) if epoch % 5 == 0 else None
        model.eval()
        embedding, _, _ = model(feature_multi_omics, G)
        multi_times_embedding_list.append(embedding.cpu().detach().numpy())
        
    overall_performance_report(multi_times_embedding_list, None, label, args.output_dir)
