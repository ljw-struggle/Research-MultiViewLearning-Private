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

from model import DeepSEM
from utils import calculate_auroc, calculate_aupr, calculate_ep, profile
from sklearn.metrics import roc_auc_score,average_precision_score


def main(data_path, result_path, cell, size, alpha, beta):
    exp_file = os.path.join(data_path, 'expression.csv')
    train_file = os.path.join(data_path, 'train_set.csv')
    val_file = os.path.join(data_path, 'val_set.csv')
    test_file = os.path.join(data_path, 'test_set.csv')

    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    
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


    # Define model, optimizer and scheduler
    model = DeepSEM(len(gene_set), alpha=alpha, beta=beta)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
    optimizer_1 = torch.optim.RMSprop([model.adj_parameter], lr=0.0001 * 0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    data = TensorDataset(torch.from_numpy(node_vector.transpose()))
    data_loader = DataLoader(data, batch_size=64, shuffle=True, num_workers=1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
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
    best_val_ep = 0
    for epoch in range(20): # 5000
        # Training
        model.train()
        time_start = time.time()

        for batch_idx, batch in enumerate(data_loader):
            optimizer.zero_grad()
            dropout_mask = torch.as_tensor(batch[0] != 0, dtype=torch.float32) 
            loss, x_recon, mean, logvar = model(batch[0].to(device), dropout_mask.to(device))
            loss.backward(retain_graph=True)
            if epoch % 2 == 0:
                optimizer.step()
            else:
                optimizer_1.step()
            
            # print the gradient
            # print(model.adj_parameter)
            # print(model.adj_parameter.grad)
            
        scheduler.step()
        time_end = time.time()
        duration = time_end - time_start

        # For Validation
        model.eval()
        adj = model.adj_parameter.cpu().detach().numpy()
        val_data_ = val_data.cpu().detach().numpy()
        val_pred = adj[val_data_[:, 0], val_data_[:, 1]]
        val_pred = - np.abs(val_pred)

        # Print the results
        val_pred = torch.sigmoid(torch.from_numpy(val_pred)).to(device)
        AUC = roc_auc_score(y_true=val_label.detach().cpu().numpy(), y_score=val_pred.detach().cpu().numpy())
        AUPR = average_precision_score(y_true=val_label.detach().cpu().numpy(), y_score=val_pred.detach().cpu().numpy())
        AUPR_norm = AUPR/np.mean(val_label.cpu().numpy())
        EP, EPR = calculate_ep(val_pred.detach().cpu().numpy(), val_label.cpu().numpy())
        print('Epoch: {}'.format(epoch + 1), 'Valid AUC: {:.3F}'.format(AUC), 'Valid AUPR: {:.3F}'.format(AUPR), 'Valid AUPR_norm: {:.3F}'.format(AUPR_norm), 'Valid EP: {:.3F}'.format(EP), 'Valid EPR: {:.3F}'.format(EPR), 'Time: {:.3F}'.format(duration), flush=True)

        # For Test
        model.eval()
        adj = model.adj_parameter.cpu().detach().numpy()
        test_data_ = test_data.cpu().detach().numpy()
        test_pred = adj[test_data_[:, 0], test_data_[:, 1]]
        test_pred = - np.abs(test_pred)

        # Print the results
        test_pred = torch.sigmoid(torch.from_numpy(test_pred)).to(device)
        test_AUC = roc_auc_score(y_true=test_label.detach().cpu().numpy(), y_score=test_pred.detach().cpu().numpy())
        test_AUPR = average_precision_score(y_true=test_label.detach().cpu().numpy(), y_score=test_pred.detach().cpu().numpy())
        test_AUPR_norm = AUPR/np.mean(test_label.cpu().numpy())
        test_EP, test_EPR = calculate_ep(test_pred.detach().cpu().numpy(), test_label.cpu().numpy())

        if EP > best_val_ep:
            torch.save(model.state_dict(), os.path.join(result_path, 'model.pkl'))
            best_val_ep = EP
            best_test_AUC = test_AUC
            best_test_AUPR = test_AUPR
            best_test_AUPR_norm = test_AUPR_norm
            best_test_EP = test_EP
            best_test_EPR = test_EPR
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= patience:
            break    

    print('Test')
    print('Test AUC: {:.3F}'.format(best_test_AUC), 'AUPR: {:.3F}'.format(best_test_AUPR), 'AUPR_norm: {:.3F}'.format(best_test_AUPR_norm), 'EP: {:.3F}'.format(best_test_EP), 'EPR: {:.3F}'.format(best_test_EPR), flush=True)

    # Calculate MACs, Params and Memory Usage
    macs, params = profile(model, inputs=(batch[0].to(device), dropout_mask.to(device))) # batch_size = 64
    print('MACs: {:.3F} G'.format(macs/1024/1024/1024), flush=True)
    print('Params: {:.3F} M'.format(params/1024/1024), flush=True)
    print('Memory Usage: {:.3F} GB'.format(torch.cuda.memory_allocated(device)/1024/1024/1024), flush=True)


if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Human-Blood', help='dataset path')
    parser.add_argument('--result_path', type=str, default='../result/temp', help='result path')
    parser.add_argument('--seed', type=int, default=8, help='random seed')
    parser.add_argument('--cell', type=float, default=5000, help='the cell number')
    parser.add_argument('--size', type=float, default=1.0, help='the size of training data')
    parser.add_argument('--alpha', type=float, default=100, help='the weight of constraint_loss')
    parser.add_argument('--beta', type=float, default=1, help='the weight of kl_loss')
    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Main
    main(data_path=args.data_path, result_path=args.result_path, cell=args.cell, size=args.size, alpha=args.alpha, beta=args.beta)
    