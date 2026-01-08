# -*- coding: utf-8 -*-
import os
import time
import random
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

from model import GCNModelVAE, loss_function
from utils import calculate_auroc, calculate_aupr, calculate_ep, profile


def main(data_path, result_path, cell, size):
    exp_file = os.path.join(data_path, 'expression.csv')
    train_file = os.path.join(data_path, 'train_set.csv')
    val_file = os.path.join(data_path, 'val_set.csv')
    test_file = os.path.join(data_path, 'test_set.csv')

    # Load the node vector and the gene set
    node_vector = pd.read_csv(exp_file, index_col=0, header=0).values # shape: (n_gene, n_feature)
    node_vector = StandardScaler().fit_transform(node_vector.T).T # shape: (n_gene, n_feature)
    node_vector = node_vector / node_vector.shape[0] # shape: (n_gene, n_feature) # why divide by n_gene? For the sake of the loss function
    node_vector = node_vector.astype(np.float32) # shape: (n_gene, n_feature)
    if cell < node_vector.shape[1]:
        np.random.seed(0)
        index = np.random.choice(node_vector.shape[1], int(cell), replace=False)
        index = np.sort(index)
        node_vector = node_vector[:, index] # shape: (n_gene, data_cell)

    gene_set = pd.read_csv(exp_file, index_col=0, header=0).index.values
    gene_index_dict = {gene: i for i, gene in enumerate(gene_set)}
    index_gene_dict = {i: gene for i, gene in enumerate(gene_set)}


    # Load the training, validation and test set
    train_file = pd.read_csv(train_file, index_col=None, header=0).values
    if size < 1:
        np.random.seed(0)
        index = np.random.choice(train_file.shape[0], int(train_file.shape[0] * size), replace=False)
        train_file = train_file[index, :]
    train_data = np.array([[gene_index_dict[row[0]], gene_index_dict[row[1]]] for row in train_file])
    train_label = np.array([row[-1] for row in train_file]) # shape: (n_edge, ) for normal inference

    val_file = pd.read_csv(val_file, index_col=None, header=0).values
    val_data = np.array([[gene_index_dict[row[0]], gene_index_dict[row[1]]] for row in val_file])
    val_label = np.array([row[-1] for row in val_file])

    test_file = pd.read_csv(test_file, index_col=None, header=0).values
    test_data = np.array([[gene_index_dict[row[0]], gene_index_dict[row[1]]] for row in test_file])
    test_label = np.array([row[-1] for row in test_file])


    # Load the prior graph topology
    adj = np.zeros((len(gene_set), len(gene_set)))
    for row in train_file:
        if row[-1] == 1:
            adj[gene_index_dict[row[0]], gene_index_dict[row[1]]] = 1.0
            adj[gene_index_dict[row[1]], gene_index_dict[row[0]]] = 1.0
    adj = adj + np.identity(len(gene_set)) # Add self-loop
    pos_weight = torch.tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()]) # shape: (1, )
    adj = torch.sparse_coo_tensor(indices=torch.from_numpy(np.array(adj.nonzero())).contiguous(), values=torch.from_numpy(adj[adj.nonzero()]), size=adj.shape, dtype=torch.float32).to_dense() # shape: (n_gene, n_gene)

    # Define model, optimizer and scheduler
    model = GCNModelVAE(input_dim=node_vector.shape[1], hidden_dim_1=32, hidden_dim_2=16, dropout=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    node_vector = torch.from_numpy(node_vector).to(device)
    adj = adj.to(device)
    model = model.to(device)
    pos_weight = pos_weight.to(device)
    train_data = torch.from_numpy(train_data).to(device)
    train_label = torch.from_numpy(train_label).to(device).float()
    val_data = torch.from_numpy(val_data).to(device)
    val_label = torch.from_numpy(val_label).to(device).float()
    test_data = torch.from_numpy(test_data).to(device)
    test_label = torch.from_numpy(test_label).to(device).float()


    # For Training and Validation
    print('Training and Validation')
    patience = 100
    patience_count = 0
    for epoch in range(100): # 5000
        # Training
        model.train()
        time_start = time.time()
        optimizer.zero_grad()
        adj_pred, mu, logvar = model(node_vector, adj)
        train_pred = adj_pred[train_data[:, 0], train_data[:, 1]]
        loss = loss_function(train_pred, train_label, mu, logvar, n_nodes=len(gene_set), norm=1.0, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        time_end = time.time()
        duration = time_end - time_start

        # Validation
        model.eval()
        adj_pred, mu, logvar = model(node_vector, adj)
        val_pred = adj_pred[val_data[:, 0], val_data[:, 1]]
        val_loss = loss_function(val_pred, val_label, mu, logvar, n_nodes=len(gene_set), norm=1.0, pos_weight=pos_weight)

        # Early stopping
        if epoch == 0:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(result_path, 'model.pkl'))
        else:  
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(result_path, 'model.pkl'))
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    break

        scheduler.step(val_loss)

        # Print the results
        val_pred = torch.sigmoid(val_pred)
        AUC = roc_auc_score(y_true=val_label.detach().cpu().numpy(), y_score=val_pred.detach().cpu().numpy())
        AUPR = average_precision_score(y_true=val_label.detach().cpu().numpy(), y_score=val_pred.detach().cpu().numpy())
        AUPR_norm = AUPR/np.mean(val_label.cpu().numpy())
        EP, EPR = calculate_ep(val_pred.detach().cpu().numpy(), val_label.cpu().numpy())
        print('Epoch: {}'.format(epoch + 1), 'Valid AUC: {:.3F}'.format(AUC), 'Valid AUPR: {:.3F}'.format(AUPR), 'Valid AUPR_norm: {:.3F}'.format(AUPR_norm), 'Valid EP: {:.3F}'.format(EP), 'Valid EPR: {:.3F}'.format(EPR), 'Time: {:.3F}'.format(duration), flush=True)


    # For Test
    print('Test')
    model.load_state_dict(torch.load(os.path.join(result_path, 'model.pkl')))
    model.eval()

    adj_pred, mu, logvar = model(node_vector, adj)
    train_pred = adj_pred[train_data[:, 0], train_data[:, 1]]
    train_pred = torch.sigmoid(train_pred)
    val_pred = adj_pred[val_data[:, 0], val_data[:, 1]]
    val_pred = torch.sigmoid(val_pred)
    test_pred = adj_pred[test_data[:, 0], test_data[:, 1]]
    test_pred = torch.sigmoid(test_pred)

    train_pred = train_pred.cpu().detach().numpy().flatten()
    train_label = train_label.cpu().detach().numpy().flatten().astype(int)
    val_pred = val_pred.cpu().detach().numpy().flatten()
    val_label = val_label.cpu().detach().numpy().flatten().astype(int)
    test_pred = test_pred.cpu().detach().numpy().flatten()
    test_label = test_label.cpu().detach().numpy().flatten().astype(int)
    AUC = roc_auc_score(y_true=test_label, y_score=test_pred)
    AUPR = average_precision_score(y_true=test_label, y_score=test_pred)
    AUPR_norm = AUPR/np.mean(test_label)
    EP, EPR = calculate_ep(test_pred, test_label)
    # ACC = balanced_accuracy_score(y_true=test_label, y_pred=(test_pred>0.5).astype(int))
    # RECALL = recall_score(y_true=test_label, y_pred=(test_pred>0.5).astype(int))
    # PRECISION = precision_score(y_true=test_label, y_pred=(test_pred>0.5).astype(int))
    # F1 = f1_score(y_true=test_label, y_pred=(test_pred>0.5).astype(int))
    # MCC = matthews_corrcoef(y_true=test_label, y_pred=(test_pred>0.5).astype(int))
    print('Test AUC: {:.3F}'.format(AUC), 'AUPR: {:.3F}'.format(AUPR), 'AUPR_norm: {:.3F}'.format(AUPR_norm), 'EP: {:.3F}'.format(EP), 'EPR: {:.3F}'.format(EPR), flush=True)


    # Calculate MACs, Params and Memory Usage
    macs, params = profile(model, inputs=(node_vector, adj))
    print('MACs: {:.3F} G'.format(macs/1024/1024/1024), flush=True)
    print('Params: {:.3F} M'.format(params/1024/1024), flush=True)
    print('Memory Usage: {:.3F} GB'.format(torch.cuda.memory_allocated(device)/1024/1024/1024), flush=True)


    # Save the results
    data_type_list = []
    TF_list = []
    GENE_list = []
    output_list = []
    label_list = []

    for i in range(len(train_pred)):
        data_type_list.append('train')
        TF_list.append(index_gene_dict[train_data[i, 0].item()])
        GENE_list.append(index_gene_dict[train_data[i, 1].item()])
        output_list.append(train_pred[i])
        label_list.append(train_label[i])
    
    for i in range(len(val_pred)):
        data_type_list.append('val')
        TF_list.append(index_gene_dict[val_data[i, 0].item()])
        GENE_list.append(index_gene_dict[val_data[i, 1].item()])
        output_list.append(val_pred[i])
        label_list.append(val_label[i])

    for i in range(len(test_pred)):
        data_type_list.append('test')
        TF_list.append(index_gene_dict[test_data[i, 0].item()])
        GENE_list.append(index_gene_dict[test_data[i, 1].item()])
        output_list.append(test_pred[i])
        label_list.append(test_label[i])

    df = pd.DataFrame({'data_type': data_type_list, 'TF': TF_list, 'GENE': GENE_list, 'output': output_list, 'label': label_list})
    df.to_csv(os.path.join(result_path, 'result.csv'), index=False)


if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Human-Blood', help='dataset path')
    parser.add_argument('--result_path', type=str, default='../result/temp', help='result path')
    parser.add_argument('--seed', type=int, default=8, help='random seed')
    parser.add_argument('--cell', type=float, default=5000, help='the cell number')
    parser.add_argument('--size', type=float, default=1.0, help='the size of training data')
    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Main
    os.makedirs(args.result_path, exist_ok=True)
    main(data_path=args.data_path, result_path=args.result_path, cell=args.cell, size=args.size)
    