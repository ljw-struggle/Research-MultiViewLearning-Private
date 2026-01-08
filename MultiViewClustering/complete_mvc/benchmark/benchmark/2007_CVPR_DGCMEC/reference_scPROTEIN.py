# -*- coding: utf-8 -*-
import os
import re
import random
import argparse
import numpy as np 
import pandas as pd
import scipy.sparse as sp
from time import perf_counter
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import dropout_adj
    

class Model(torch.nn.Module):
    """
    Model for unsupervised training.
    Reference: https://github.com/CRIPAC-DIG/GRACE
    Reference: https://github.com/BRGCL-code/BRGCL-code
    """
    def __init__(self, in_channels=1490, out_channels=400, base_model='GCNConv', activation='relu', 
                 num_layers = 2, num_hidden_1 = 200, num_hidden_2 = 400):
        super(Model, self).__init__()
        assert base_model in ['GCNConv', 'GATConv'], 'base_model must be GCNConv or GATConv'
        assert activation in ['relu', 'prelu'], 'activation must be relu or prelu'
        assert num_layers >= 2, 'k must be greater than 2'

        base_model = {'GCNConv': GCNConv, 'GATConv': GATConv}[base_model]
        self.conv = []
        self.activation = []
        for i in range(num_layers):
            self.conv.append(base_model(in_channels, 2 * out_channels)) if i == 0 else None
            self.conv.append(base_model(2 * out_channels, 2 * out_channels)) if i != 0 and i != num_layers-1 else None
            self.conv.append(base_model(2 * out_channels, out_channels)) if i == num_layers-1 else None
            self.activation.append(nn.ReLU() if activation == 'relu' else nn.PReLU())
        self.conv = nn.ModuleList(self.conv)
        self.activation = nn.ModuleList(self.activation)

        self.fc_1 = torch.nn.Linear(out_channels, num_hidden_1)
        self.ac_1 = nn.ReLU() if activation == 'relu' else nn.PReLU()
        self.fc_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)

    def forward(self, x, edge_index):
        for i in range(len(self.conv)):
            x = self.activation[i](self.conv[i](x, edge_index))
        return x

    def projection(self, z):
        h = self.fc_2(self.ac_1(self.fc_1(z)))
        return h
    
    @staticmethod
    def drop_feature(x, drop_prob):
        # x.shape: (num_cells, num_proteins)
        drop_mask = torch.zeros((x.size(1))).uniform_(0, 1) < drop_prob # shape: (num_proteins)
        return x * drop_mask.unsqueeze(0) # shape: (num_cells, num_proteins)
    
    @staticmethod
    def drop_edge(edge_index, drop_prob):
        # edge_index.shape: (2, num_edges)
        edge_index_drop, _ = dropout_adj(edge_index, p=drop_prob)
        return edge_index_drop # shape: (2, num_edges * (1 - drop_prob))

    @staticmethod
    def loss_contrast_node(h_1, h_2, tau=0.4):
        # h_1: shape: (num_cells, num_hidden_2)
        # h_2: shape: (num_cells, num_hidden_2)
        h_1 = F.normalize(h_1)
        h_2 = F.normalize(h_2)
        intra_sim_h1_h1 = torch.exp(torch.mm(h_1, h_1.t()) / tau) # shape: (num_cells, num_cells)
        inter_sim_h1_h2 = torch.exp(torch.mm(h_1, h_2.t()) / tau) # shape: (num_cells, num_cells)
        intra_sim_h2_h2 = torch.exp(torch.mm(h_2, h_2.t()) / tau) # shape: (num_cells, num_cells)
        inter_sim_h2_h1 = torch.exp(torch.mm(h_2, h_1.t()) / tau) # shape: (num_cells, num_cells)
        l_1 = -torch.log(inter_sim_h1_h2.diag() / (intra_sim_h1_h1.sum(1) + inter_sim_h1_h2.sum(1) - intra_sim_h1_h1.diag())) # shape: (num_cells)
        l_2 = -torch.log(inter_sim_h2_h1.diag() / (intra_sim_h2_h2.sum(1) + inter_sim_h2_h1.sum(1) - intra_sim_h2_h2.diag())) # shape: (num_cells)
        loss = (l_1 + l_2) * 0.5 # shape: (num_cells)
        return loss.mean() # shape: (1)

    @staticmethod
    def loss_contrast_proto(cell_embeddings, cell_cluster_centers, cell_cluster_labels, tau=0.4):
        # cell_embeddings.shape: (num_cell, num_hidden)
        # cell_cluster_centers.shape: (num_protos, num_hidden)
        # cell_cluster_labels.shape: (num_cell)
        cell_embeddings = F.normalize(cell_embeddings, dim=1) # shape: (num_cell, num_hidden)
        cell_cluster_centers = F.normalize(cell_cluster_centers, dim=1) # shape: (num_protos, num_hidden)
        cell_cluster_labels = cell_cluster_labels.unsqueeze(1) # shape: (num_cell, 1)

        sim_cell_proto = torch.exp(torch.mm(cell_embeddings, cell_cluster_centers.t()) / tau) # shape: (num_cell, num_protos)
        sim_cell_corresponding_proto = torch.gather(sim_cell_proto, -1, cell_cluster_labels) # shape: (num_cell, 1)
        sim_cell_all_proto = torch.sum(sim_cell_proto, -1, keepdim=True) # shape: (num_cell, 1)
        sim_cell_corresponding_proto = torch.div(sim_cell_corresponding_proto, sim_cell_all_proto) # shape: (num_cell, 1)
        loss = -torch.log(sim_cell_corresponding_proto) # shape: (num_cell, 1)
        return loss.mean() # shape: (1)


if __name__ == '__main__':
    # Parse arguments and create directory to save results
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../data/SCoPE2_Specht/', help='directory to load data.')
    parser.add_argument("--result_dir", type=str, default='../result/SCoPE2_Specht/graph_contrastive_learning/seed_666_epoch_100_patience_15/', help='directory to save results.')
    parser.add_argument("--uncertainty_dir", type=str, default='../result/SCoPE2_Specht/uncertainty_estimation/seed_666_epoch_100_patience_15/', help='directory to load peptide uncertainty.')

    parser.add_argument("--seed", type=int, default=666, help='random seed')
    parser.add_argument("--learning_rate", type=float, default=1e-2, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=1e-5, help='weight_decay')
    parser.add_argument("--num_epochs", type=int, default=1200, help='number of epochs')
    parser.add_argument("--patience", type=int, default=30, help='Hidden dimension.')

    parser.add_argument("--uncertainty_preprocess", type=bool, default=True, help='if scPROTEIN starts from stage_1')
    parser.add_argument("--feature_preprocess", type=bool, default=True, help='feature preprocess')
    parser.add_argument("--threshold", type=float, default=0.15, help='threshold of graph construct')

    parser.add_argument("--base_model", type=str, default='GCNConv', help='base encoding model')
    parser.add_argument("--activation", type=str, default='prelu', help='activation function')
    parser.add_argument("--num_layers", type=int, default=2, help='num of GCN layers')
    parser.add_argument("--num_hidden_1", type=int, default=200, help='hidden dimension 1')
    parser.add_argument("--num_hidden_2", type=int, default=400, help='hidden dimension 2')

    parser.add_argument("--drop_edge_rate_1", type=float, default=0.2, help='dropedge rate for view1')
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.4, help='dropedge rate for view2')
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.4, help='mask_feature rate for view1')
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.2, help='mask_feature rate for view2')

    parser.add_argument("--tau", type=float, default=0.4, help='temperature coefficient')
    parser.add_argument("--num_protos", type=int, default=2, help='num of prototypes')
    parser.add_argument("--alpha", type=float, default=0.05, help='balance factor')
    
    parser.add_argument("--topology_denoising", type=bool, default=True, help='if scPROTEIN uses topology denoising')
    parser.add_argument("--num_changed_edges", type=int, default=50, help='num of added/removed edges')
    args = parser.parse_args()

    data_dir = args.data_dir
    result_dir = args.result_dir
    uncertainty_dir = args.uncertainty_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load data
    peptide_dataframe = pd.read_csv(os.path.join(data_dir, 'Peptides-raw.csv'), index_col=None, header=0).fillna(0.)
    protein_list = np.array(peptide_dataframe['protein'].tolist())
    peptide_list = np.array(peptide_dataframe['peptide'].tolist())
    peptide_list = [re.sub(r'\(.*?\)', '', peptide).replace(')', '').split('_')[1] for peptide in peptide_list]
    cell_list = peptide_dataframe.iloc[:,2:].columns.tolist()
    peptide_expression_data = peptide_dataframe.iloc[:,2:].values.astype(np.float32) # shape: (peptide_num, cell_num)

    protein_set = np.unique(protein_list)
    if args.uncertainty_preprocess:
        # uncertainty-guided protein-level data
        uncertainty = np.load(os.path.join(uncertainty_dir, 'uncertainty.npz'))['uncertainty'] # shape: (peptide_num, cell_num)
        uncertainty_score = 1. / uncertainty
        peptide_expression_data = np.multiply(peptide_expression_data, uncertainty_score) # shape: (peptide_num, cell_num)
        protein_expression_data = [] # obtain the uncertainty-guided protein-level data
        for protein in protein_set:
            indices = (protein_list == protein)
            expression_data = peptide_expression_data[indices, :] # shape: (peptide_num_protein, cell_num)
            expression_data = np.sum(expression_data, axis=0) # shape: (cell_num)
            protein_expression_data.append(expression_data)
        protein_expression_data = np.array(protein_expression_data).astype(np.float32) # shape: (protein_num, cell_num)
    else:
        # non-uncertainty-guided protein-level data
        protein_dataframe = peptide_dataframe.groupby('protein').sum()
        protein_expression_data = [] # obtain the non-uncertainty-guided protein-level data
        for protein in protein_set:
            protein_expression_data.append(protein_dataframe.loc[protein].values)
        protein_expression_data = np.array(protein_expression_data).astype(np.float32) # shape: (protein_num, cell_num)

    protein_expression_data = pd.DataFrame(protein_expression_data) # shape: (num_protein, num_cell)
    adj_matrix = protein_expression_data.corr() # shape: (num_cell, num_cell), Pearson correlation coefficient
    adj_matrix = np.where(adj_matrix > args.threshold, 1, 0) # shape: (num_cell, num_cell)
    protein_expression_data = protein_expression_data.values.T # shape: (num_cell, num_protein)
    print('num_cell: {}, num_protein: {}'.format(protein_expression_data.shape[0], protein_expression_data.shape[1]))
    print('num_edges: {}'.format(np.sum(adj_matrix)))

    if args.feature_preprocess:
        # row-normalize protein expression data
        rowsum = np.array(protein_expression_data.sum(1))
        rowsum[np.where(rowsum == 0)] = 1 # shape: (num_cell)
        protein_expression_data = protein_expression_data / rowsum[:, np.newaxis]  # shape: (num_cell, num_protein)

    sp_adj_matrix = sp.coo_matrix(adj_matrix)
    edge_index = torch.tensor(np.vstack((sp_adj_matrix.row, sp_adj_matrix.col)), dtype=torch.long) # shape: (2, num_edges)
    protein_expression_data = torch.tensor(protein_expression_data, dtype=torch.float32) # shape: (num_cell, num_protein)
    data = Data(x=protein_expression_data, edge_index=edge_index)

    # Define model and optimizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(in_channels = data.num_features, out_channels = args.num_hidden_2, base_model = args.base_model, activation = args.activation,
                  num_layers = args.num_layers, num_hidden_1 = args.num_hidden_1, num_hidden_2 = args.num_hidden_2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)


    # Train model
    best_loss = np.inf
    not_improved_count = 0
    for epoch in range(0, args.num_epochs):
        start = perf_counter()

        model.train()
        optimizer.zero_grad()

        # loss_node: node-level contrastive learning
        x_1 = model.drop_feature(data.x, args.drop_feature_rate_1).to(device)
        edge_index_1 = model.drop_edge(data.edge_index, args.drop_edge_rate_1).to(device)
        z_1 = model(x_1, edge_index_1)
        h_1 = model.projection(z_1)
        x_2 = model.drop_feature(data.x, args.drop_feature_rate_2).to(device)
        edge_index_2 = model.drop_edge(data.edge_index, args.drop_edge_rate_2).to(device)
        z_2 = model(x_2, edge_index_2)
        h_2 = model.projection(z_2)
        loss_node = model.loss_contrast_node(h_1, h_2, args.tau)
        
        # loss_proto: prototype-level contrastive learning, prototype-based denoising
        cell_embeddings = model(data.x.to(device), data.edge_index.to(device)) # shape: (num_cell, num_hidden)
        cell_embeddings_numpy = cell_embeddings.cpu().detach().numpy() # shape: (num_cell, num_hidden)
        kmeans = KMeans(n_clusters=args.num_protos, n_init=20).fit(cell_embeddings_numpy)
        cell_cluster_labels = kmeans.labels_ # shape: (num_cell)
        cell_cluster_centers = kmeans.cluster_centers_ # shape: (num_protos, num_hidden)
        cell_cluster_labels = torch.tensor(cell_cluster_labels, dtype=torch.long).to(device)
        cell_cluster_centers = torch.tensor(cell_cluster_centers, dtype=torch.float32).to(device)
        loss_proto = model.loss_contrast_proto(cell_embeddings, cell_cluster_centers, cell_cluster_labels)
        
        # topology denoising
        if args.topology_denoising:
            with torch.no_grad():
                model.eval()
                cell_embeddings = model(data.x.to(device), data.edge_index.to(device)) # shape: (num_cell, num_hidden)
                cell_embeddings = cell_embeddings.cpu().detach().numpy() # shape: (num_cell, num_hidden)

                # calculate the pcc similarity matrix
                similarity_matrix = np.corrcoef(cell_embeddings) # shape: (num_cell, num_cell)
                similarity_matrix = sp.coo_matrix(similarity_matrix) # shape: (num_cell, num_cell)
                similarity_data = list(similarity_matrix.data) # shape: (num_cell * num_cell)
                coords = list(np.vstack((similarity_matrix.row, similarity_matrix.col)).transpose()) # shape: (num_cell * num_cell, 2)
                coord_value_dict = {tuple(coords[i]): similarity_data[i] for i in range(len(coords))}
                for i in range(len(cell_embeddings)): # remove the diagonal elements
                    coord_value_dict.pop((i, i))

                # select the top-k high probability edges and low probability edges
                coords = np.array(list(coord_value_dict.keys())) # shape: (num_cell * num_cell - num_cell, 2)
                values = np.array(list(coord_value_dict.values())) # shape: (num_cell * num_cell - num_cell)
                indices_sort = np.argsort(values)  # sort by ascending order
                high_prob_coords = coords[indices_sort[-args.num_changed_edges:]].tolist() # shape: (num_changed_edges, 2)
                low_prob_coords = coords[indices_sort[:args.num_changed_edges]].tolist() # shape: (num_changed_edges, 2)
                
                # add/remove edges
                edge_index_now = data.edge_index.cpu().detach().numpy().T.tolist() # shape: (num_edges, 2)
                count_add = 0
                count_remove = 0
                for i in high_prob_coords:
                    if i not in edge_index_now:
                        edge_index_now.append(i)
                        count_add += 1
                for i in low_prob_coords:
                    if i in edge_index_now:
                        edge_index_now.remove(i)
                        count_remove += 1
                
                data.edge_index = torch.tensor(np.array(edge_index_now).T, dtype=torch.long)
                print('high_prob_coords: {}, low_prob_coords: {}'.format(len(high_prob_coords), len(low_prob_coords)))
                print('count_add: {}, count_remove: {}'.format(count_add, count_remove))

        loss = loss_node # + args.alpha * loss_proto
        loss.backward()
        optimizer.step()

        loss = loss.item()
        loss_node = loss_node.item()
        loss_proto = loss_proto.item()

        if loss < best_loss:
            best_loss = loss
            not_improved_count = 0
            torch.save(model.state_dict(), os.path.join(result_dir, 'model.pth'))
        else:
            not_improved_count += 1
            if not_improved_count == args.patience:
                break

        end = perf_counter()
        print(f'Epoch: {epoch:03d}, loss: {loss:.4f}, loss_node: {loss_node:.4f}, loss_proto: {loss_proto:.4f}, time: {end-start:.4f}')

    # Save embedding of the best model
    model.load_state_dict(torch.load(os.path.join(result_dir, 'model.pth')))
    model.eval()
    embedding = model(data.x.to(device), data.edge_index.to(device))
    np.savez(os.path.join(result_dir, 'embedding.npz'), embedding=embedding.cpu().detach().numpy(), cell_list=cell_list)
