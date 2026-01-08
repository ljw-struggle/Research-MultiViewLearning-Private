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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

from model import STGRNS
from loader import STGRNS_Data
from utils import calculate_auroc, calculate_aupr, calculate_ep, profile


def main(data_path, result_path, cell, size):
    # Load the training, validation and test set
    exp_file = os.path.join(data_path, 'expression.csv')
    train_file = os.path.join(data_path, 'train_set.csv')
    val_file = os.path.join(data_path, 'val_set.csv')
    test_file = os.path.join(data_path, 'test_set.csv')
    train_dataset = STGRNS_Data(exp_file=exp_file, data_file=train_file, data_cell=cell, data_size=size)
    val_dataset = STGRNS_Data(exp_file=exp_file, data_file=val_file, data_cell=cell, data_size=1)
    test_dataset = STGRNS_Data(exp_file=exp_file, data_file=test_file, data_cell=cell, data_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    # Define model, optimizer and scheduler
    model = STGRNS()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    # For Training and Validation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print('Training and Validation')
    patience = 0
    best_val_loss = np.inf
    for epoch in range(1, 201):
        model.train()
        time_start = time.time()
        train_loss_total = 0
        train_sample_total = 0
        for train_data, train_label, _, _ in train_dataloader:
            train_data = train_data.to(device) # (batch_size, seq_len, input_dim)
            train_label = train_label.to(device) # (batch_size, 1)
            train_pred = model(train_data)
            train_loss = criterion(train_pred, train_label)
            optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) # Gradient Clipping
            optimizer.step()
            scheduler.step()

            train_loss_total = train_loss_total + train_loss.item() * train_data.size(0)
            train_sample_total = train_sample_total + train_data.size(0)

        train_loss = train_loss_total / train_sample_total

        model.eval()
        val_loss_total = 0
        val_sample_total = 0
        with torch.no_grad():
            for val_data, val_label, _, _ in val_dataloader:
                val_data = val_data.to(device)
                val_label = val_label.to(device)
                val_pred = model(val_data)
                val_loss = criterion(val_pred, val_label)

                val_loss_total = val_loss_total + val_loss.item() * val_data.size(0)
                val_sample_total = val_sample_total + val_data.size(0)

        val_loss = val_loss_total / val_sample_total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(result_path, 'model.pkl'))
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
        
        time_end = time.time()
        duration = time_end - time_start
        print('Epoch: {:03d}'.format(epoch), 'Train Loss: {:.3F}'.format(train_loss), 'Val Loss: {:.3F}'.format(val_loss), 'Duration: {:.3F}'.format(duration), flush=True)
        

    # For Test PBMC1
    print('Test')
    model.load_state_dict(torch.load(os.path.join(result_path, 'model.pkl')))

    exp_file = os.path.join(data_path + '/PBMC1', 'expression.csv')
    train_file = os.path.join(data_path + '/PBMC1', 'train_set.csv')
    val_file = os.path.join(data_path + '/PBMC1', 'val_set.csv')
    test_file = os.path.join(data_path + '/PBMC1', 'test_set.csv')
    train_dataset = STGRNS_Data(exp_file=exp_file, data_file=train_file, data_cell=cell, data_size=size)
    val_dataset = STGRNS_Data(exp_file=exp_file, data_file=val_file, data_cell=cell, data_size=1)
    test_dataset = STGRNS_Data(exp_file=exp_file, data_file=test_file, data_cell=cell, data_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    model.eval()
    train_tf_list = []
    train_gene_list = []
    train_pred_list = []
    train_label_list = []
    for train_data, train_label, node_1_name, node_2_name in train_dataloader:
        train_tf_list.append(node_1_name)
        train_gene_list.append(node_2_name)

        train_data = train_data.to(device)
        train_label = train_label.to(device)
        train_pred = model(train_data)
        train_pred = F.softmax(train_pred, dim=1)
        train_pred_list.append(train_pred[:, 1].cpu().detach().numpy())
        train_label_list.append(train_label.cpu().detach().numpy().astype(int))

    val_tf_list = []
    val_gene_list = []
    val_pred_list = []
    val_label_list = []
    for val_data, val_label, node_1_name, node_2_name in val_dataloader:
        val_tf_list.append(node_1_name)
        val_gene_list.append(node_2_name)

        val_data = val_data.to(device)
        val_label = val_label.to(device)
        val_pred = model(val_data)
        val_pred = F.softmax(val_pred, dim=1)
        val_pred_list.append(val_pred[:, 1].cpu().detach().numpy())
        val_label_list.append(val_label.cpu().detach().numpy().astype(int))

    test_tf_list = []
    test_gene_list = []
    test_pred_list = []
    test_label_list = []
    for test_data, test_label, node_1_name, node_2_name in test_dataloader:
        test_tf_list.append(node_1_name)
        test_gene_list.append(node_2_name)

        test_data = test_data.to(device)
        test_label = test_label.to(device)
        test_pred = model(test_data)
        test_pred = F.softmax(test_pred, dim=1)
        test_pred_list.append(test_pred[:, 1].cpu().detach().numpy())
        test_label_list.append(test_label.cpu().detach().numpy().astype(int))

    train_tf = np.concatenate(train_tf_list)
    train_gene = np.concatenate(train_gene_list)
    train_pred = np.concatenate(train_pred_list)
    train_label = np.concatenate(train_label_list)

    val_tf = np.concatenate(val_tf_list)
    val_gene = np.concatenate(val_gene_list)
    val_pred = np.concatenate(val_pred_list)
    val_label = np.concatenate(val_label_list)
    
    test_tf = np.concatenate(test_tf_list)
    test_gene = np.concatenate(test_gene_list)
    test_pred = np.concatenate(test_pred_list)
    test_label = np.concatenate(test_label_list)


    # Save the results
    data_type_list = []
    TF_list = []
    GENE_list = []
    output_list = []
    label_list = []

    for i in range(len(train_pred)):
        data_type_list.append('train')
        TF_list.append(train_tf[i])
        GENE_list.append(train_gene[i])
        output_list.append(train_pred[i])
        label_list.append(train_label[i])
    
    for i in range(len(val_pred)):
        data_type_list.append('val')
        TF_list.append(val_tf[i])
        GENE_list.append(val_gene[i])
        output_list.append(val_pred[i])
        label_list.append(val_label[i])

    for i in range(len(test_pred)):
        data_type_list.append('test')
        TF_list.append(test_tf[i])
        GENE_list.append(test_gene[i])
        output_list.append(test_pred[i])
        label_list.append(test_label[i])

    df = pd.DataFrame({'data_type': data_type_list, 'TF': TF_list, 'GENE': GENE_list, 'output': output_list, 'label': label_list})
    df.to_csv(os.path.join(result_path, 'result_PBMC1.csv'), index=False)
    AUC = roc_auc_score(y_true=label_list, y_score=output_list)
    AUPR = average_precision_score(y_true=label_list, y_score=output_list)
    AUPR_norm = AUPR/np.mean(label_list)
    EP, EPR = calculate_ep(output_list, label_list)
    log = {'auroc': AUC, 'aupr': AUPR, 'aupr_norm': AUPR_norm, 'ep': EP, 'epr': EPR}
    # write log to txt file
    with open(os.path.join(result_path, 'log_PBMC1.txt'), 'w') as f:
        f.write(str(log))


    # For Test PBMC2
    print('Test')
    model.load_state_dict(torch.load(os.path.join(result_path, 'model.pkl')))

    exp_file = os.path.join(data_path + '/PBMC2', 'expression.csv')
    train_file = os.path.join(data_path + '/PBMC2', 'train_set.csv')
    val_file = os.path.join(data_path + '/PBMC2', 'val_set.csv')
    test_file = os.path.join(data_path + '/PBMC2', 'test_set.csv')
    train_dataset = STGRNS_Data(exp_file=exp_file, data_file=train_file, data_cell=cell, data_size=size)
    val_dataset = STGRNS_Data(exp_file=exp_file, data_file=val_file, data_cell=cell, data_size=1)
    test_dataset = STGRNS_Data(exp_file=exp_file, data_file=test_file, data_cell=cell, data_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    model.eval()
    train_tf_list = []
    train_gene_list = []
    train_pred_list = []
    train_label_list = []
    for train_data, train_label, node_1_name, node_2_name in train_dataloader:
        train_tf_list.append(node_1_name)
        train_gene_list.append(node_2_name)

        train_data = train_data.to(device)
        train_label = train_label.to(device)
        train_pred = model(train_data)
        train_pred = F.softmax(train_pred, dim=1)
        train_pred_list.append(train_pred[:, 1].cpu().detach().numpy())
        train_label_list.append(train_label.cpu().detach().numpy().astype(int))

    val_tf_list = []
    val_gene_list = []
    val_pred_list = []
    val_label_list = []
    for val_data, val_label, node_1_name, node_2_name in val_dataloader:
        val_tf_list.append(node_1_name)
        val_gene_list.append(node_2_name)

        val_data = val_data.to(device)
        val_label = val_label.to(device)
        val_pred = model(val_data)
        val_pred = F.softmax(val_pred, dim=1)
        val_pred_list.append(val_pred[:, 1].cpu().detach().numpy())
        val_label_list.append(val_label.cpu().detach().numpy().astype(int))

    test_tf_list = []
    test_gene_list = []
    test_pred_list = []
    test_label_list = []
    for test_data, test_label, node_1_name, node_2_name in test_dataloader:
        test_tf_list.append(node_1_name)
        test_gene_list.append(node_2_name)

        test_data = test_data.to(device)
        test_label = test_label.to(device)
        test_pred = model(test_data)
        test_pred = F.softmax(test_pred, dim=1)
        test_pred_list.append(test_pred[:, 1].cpu().detach().numpy())
        test_label_list.append(test_label.cpu().detach().numpy().astype(int))

    train_tf = np.concatenate(train_tf_list)
    train_gene = np.concatenate(train_gene_list)
    train_pred = np.concatenate(train_pred_list)
    train_label = np.concatenate(train_label_list)

    val_tf = np.concatenate(val_tf_list)
    val_gene = np.concatenate(val_gene_list)
    val_pred = np.concatenate(val_pred_list)
    val_label = np.concatenate(val_label_list)
    
    test_tf = np.concatenate(test_tf_list)
    test_gene = np.concatenate(test_gene_list)
    test_pred = np.concatenate(test_pred_list)
    test_label = np.concatenate(test_label_list)


    # Save the results
    data_type_list = []
    TF_list = []
    GENE_list = []
    output_list = []
    label_list = []

    for i in range(len(train_pred)):
        data_type_list.append('train')
        TF_list.append(train_tf[i])
        GENE_list.append(train_gene[i])
        output_list.append(train_pred[i])
        label_list.append(train_label[i])
    
    for i in range(len(val_pred)):
        data_type_list.append('val')
        TF_list.append(val_tf[i])
        GENE_list.append(val_gene[i])
        output_list.append(val_pred[i])
        label_list.append(val_label[i])

    for i in range(len(test_pred)):
        data_type_list.append('test')
        TF_list.append(test_tf[i])
        GENE_list.append(test_gene[i])
        output_list.append(test_pred[i])
        label_list.append(test_label[i])

    df = pd.DataFrame({'data_type': data_type_list, 'TF': TF_list, 'GENE': GENE_list, 'output': output_list, 'label': label_list})
    df.to_csv(os.path.join(result_path, 'result_PBMC2.csv'), index=False)
    AUC = roc_auc_score(y_true=label_list, y_score=output_list)
    AUPR = average_precision_score(y_true=label_list, y_score=output_list)
    AUPR_norm = AUPR/np.mean(label_list)
    EP, EPR = calculate_ep(output_list, label_list)
    log = {'auroc': AUC, 'aupr': AUPR, 'aupr_norm': AUPR_norm, 'ep': EP, 'epr': EPR}
    # write log to txt file
    with open(os.path.join(result_path, 'log_PBMC2.txt'), 'w') as f:
        f.write(str(log))

    
    # For Test PBMC3
    print('Test')
    model.load_state_dict(torch.load(os.path.join(result_path, 'model.pkl')))

    exp_file = os.path.join(data_path + '/PBMC3', 'expression.csv')
    train_file = os.path.join(data_path + '/PBMC3', 'train_set.csv')
    val_file = os.path.join(data_path + '/PBMC3', 'val_set.csv')
    test_file = os.path.join(data_path + '/PBMC3', 'test_set.csv')
    train_dataset = STGRNS_Data(exp_file=exp_file, data_file=train_file, data_cell=cell, data_size=size)
    val_dataset = STGRNS_Data(exp_file=exp_file, data_file=val_file, data_cell=cell, data_size=1)
    test_dataset = STGRNS_Data(exp_file=exp_file, data_file=test_file, data_cell=cell, data_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    model.eval()
    train_tf_list = []
    train_gene_list = []
    train_pred_list = []
    train_label_list = []
    for train_data, train_label, node_1_name, node_2_name in train_dataloader:
        train_tf_list.append(node_1_name)
        train_gene_list.append(node_2_name)

        train_data = train_data.to(device)
        train_label = train_label.to(device)
        train_pred = model(train_data)
        train_pred = F.softmax(train_pred, dim=1)
        train_pred_list.append(train_pred[:, 1].cpu().detach().numpy())
        train_label_list.append(train_label.cpu().detach().numpy().astype(int))

    val_tf_list = []
    val_gene_list = []
    val_pred_list = []
    val_label_list = []
    for val_data, val_label, node_1_name, node_2_name in val_dataloader:
        val_tf_list.append(node_1_name)
        val_gene_list.append(node_2_name)

        val_data = val_data.to(device)
        val_label = val_label.to(device)
        val_pred = model(val_data)
        val_pred = F.softmax(val_pred, dim=1)
        val_pred_list.append(val_pred[:, 1].cpu().detach().numpy())
        val_label_list.append(val_label.cpu().detach().numpy().astype(int))

    test_tf_list = []
    test_gene_list = []
    test_pred_list = []
    test_label_list = []
    for test_data, test_label, node_1_name, node_2_name in test_dataloader:
        test_tf_list.append(node_1_name)
        test_gene_list.append(node_2_name)

        test_data = test_data.to(device)
        test_label = test_label.to(device)
        test_pred = model(test_data)
        test_pred = F.softmax(test_pred, dim=1)
        test_pred_list.append(test_pred[:, 1].cpu().detach().numpy())
        test_label_list.append(test_label.cpu().detach().numpy().astype(int))

    train_tf = np.concatenate(train_tf_list)
    train_gene = np.concatenate(train_gene_list)
    train_pred = np.concatenate(train_pred_list)
    train_label = np.concatenate(train_label_list)

    val_tf = np.concatenate(val_tf_list)
    val_gene = np.concatenate(val_gene_list)
    val_pred = np.concatenate(val_pred_list)
    val_label = np.concatenate(val_label_list)
    
    test_tf = np.concatenate(test_tf_list)
    test_gene = np.concatenate(test_gene_list)
    test_pred = np.concatenate(test_pred_list)
    test_label = np.concatenate(test_label_list)


    # Save the results
    data_type_list = []
    TF_list = []
    GENE_list = []
    output_list = []
    label_list = []

    for i in range(len(train_pred)):
        data_type_list.append('train')
        TF_list.append(train_tf[i])
        GENE_list.append(train_gene[i])
        output_list.append(train_pred[i])
        label_list.append(train_label[i])
    
    for i in range(len(val_pred)):
        data_type_list.append('val')
        TF_list.append(val_tf[i])
        GENE_list.append(val_gene[i])
        output_list.append(val_pred[i])
        label_list.append(val_label[i])

    for i in range(len(test_pred)):
        data_type_list.append('test')
        TF_list.append(test_tf[i])
        GENE_list.append(test_gene[i])
        output_list.append(test_pred[i])
        label_list.append(test_label[i])

    df = pd.DataFrame({'data_type': data_type_list, 'TF': TF_list, 'GENE': GENE_list, 'output': output_list, 'label': label_list})
    df.to_csv(os.path.join(result_path, 'result_PBMC3.csv'), index=False)
    AUC = roc_auc_score(y_true=label_list, y_score=output_list)
    AUPR = average_precision_score(y_true=label_list, y_score=output_list)
    AUPR_norm = AUPR/np.mean(label_list)
    EP, EPR = calculate_ep(output_list, label_list)
    log = {'auroc': AUC, 'aupr': AUPR, 'aupr_norm': AUPR_norm, 'ep': EP, 'epr': EPR}
    # write log to txt file
    with open(os.path.join(result_path, 'log_PBMC3.txt'), 'w') as f:
        f.write(str(log))



if __name__ == '__main__':
    # Arguments Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Human-Blood-Transfer', help='dataset path')
    parser.add_argument('--result_path', type=str, default='../result/tissue_specific_data/Human-Blood-Transfer', help='result path')
    parser.add_argument('--seed', type=int, default=8, help='random seed')
    parser.add_argument('--cell', type=float, default=2000, help='the cell of cell')
    parser.add_argument('--size', type=float, default=1.0, help='the size of training data')
    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Main
    os.makedirs(args.result_path, exist_ok=True) # Create result path
    main(data_path=args.data_path, result_path=args.result_path, cell=args.cell, size=args.size)
    