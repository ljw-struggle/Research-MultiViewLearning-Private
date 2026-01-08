# -*- coding: utf-8 -*-
import os 
import re
import random
import argparse
import numpy as np
import pandas as pd
from operator import itemgetter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class Peptide_Cell_Uncertainty(nn.Module):
    def __init__(self, num_amino_acid, max_peptide_length, conv_layers, kernel_nums, kernel_sizes, max_pool_size, hidden_dim, output_dim, dropout_rate):
        super(Peptide_Cell_Uncertainty, self).__init__()
        self.num_amino_acid = num_amino_acid
        self.max_peptide_length = max_peptide_length
        self.conv_layers = conv_layers
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.max_pool_size = max_pool_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.convs = []
        for i in range(self.conv_layers):
            in_channels = self.num_amino_acid if i == 0 else self.kernel_nums[i-1]
            base_conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=self.kernel_nums[i], kernel_size = self.kernel_sizes[i]),
                                      nn.BatchNorm1d(num_features=self.kernel_nums[i]), nn.ReLU(), nn.MaxPool1d(kernel_size=self.max_pool_size))
            self.convs.append(base_conv)
        self.convs = nn.ModuleList(self.convs) # convert to module list

        self.dropout = nn.Dropout(dropout_rate)
        dim = self.kernel_nums[-1] * (self.max_peptide_length - sum(self.kernel_sizes) + len(self.kernel_sizes)) # 3100
        self.fc_1 = nn.Linear(dim, self.hidden_dim)
        self.fc_2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # x: (batch_size, num_amino_acid, max_peptide_length)
        # x_output: (batch_size, 2*num_cells)
        for i in range(self.conv_layers):
            x = self.convs[i](x)
        x = torch.flatten(x, start_dim=1, end_dim=-1) # flatten the output of conv layers
        x = self.dropout(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x
    
    @staticmethod
    def loss_uncertainty(batch_x_output, batch_y):
        # Multi-Task Heteroscedastic Regression Loss
        # batch_x_output: (batch_size, 2*num_cells)
        # batch_y: (batch_size, num_cells)
        batch_size = batch_x_output.shape[0]
        batch_x_output = batch_x_output.reshape(batch_size, -1, 2) # (batch_size, num_cells, 2)
        batch_x_output_mu = batch_x_output[:,:,0] # (batch_size, num_cells), predicted expression value
        batch_x_output_log_sigma = batch_x_output[:,:,1] # (batch_size, num_cells), uncertainty estimation: log(sigma)
        batch_x_output_sigma = torch.exp(batch_x_output_log_sigma) # (batch_size, num_cells)
        loss = torch.mean(2*batch_x_output_log_sigma + ((batch_y-batch_x_output_mu)/batch_x_output_sigma)**2) # shape: scalar
        return loss


if __name__ == '__main__':
    # Parse arguments and create directory to save results
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../data/SCoPE2_Specht/', help='directory to load data.')
    parser.add_argument("--result_dir", type=str, default='../result/SCoPE2_Specht/uncertainty_estimation/seed_666_epoch_100_patience_15/', help='directory to save results.')

    parser.add_argument("--seed", type=int, default=666, help='random seed.')
    parser.add_argument("--percentage", type=float, default=1.0, help='percentage of training data.')
    parser.add_argument("--batch_size", type=int, default=256, help='batch size.')
    parser.add_argument("--num_epochs", type=int, default=100, help='number of epochs.')
    parser.add_argument("--patience", type=int, default=15, help='patience for early stopping.')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate.')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay.')

    parser.add_argument("--conv_layers", type=int, default=3, help='layer nums of conv.')
    parser.add_argument("--kernel_nums", type=int, default=[300,200,100], help='kernel num of each conv block.')
    parser.add_argument("--kernel_sizes", type=int, default=[2,2,2], help='kernel size of each conv block.')
    parser.add_argument("--max_pool_size", type=int, default=1, help='max pooling size.')
    parser.add_argument("--hidden_dim", type=int, default=3000, help='hidden dim for fc layer.')
    parser.add_argument("--dropout_rate", type=float, default=0.5, help='drop out rate.')
    args = parser.parse_args()

    data_dir = args.data_dir
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] =str(args.seed)

    # Load data
    data_frame = pd.read_csv(os.path.join(data_dir, 'Peptides-raw.csv'), index_col=None, header=0).fillna(0.)
    cell_list = list(data_frame.columns[2:])
    protein_list = list(data_frame['protein'])
    peptide_list = list(data_frame['peptide'])
    peptide_list = [re.sub(r'\(.*?\)', '', peptide).replace(')', '').split('_')[1] for peptide in peptide_list]
    expression_data = data_frame.iloc[:,2:].values.astype(np.float32)

    amino_acid_dict = {'N': 0, 'M': 1, 'K': 2, 'W': 3, 'V': 4, 'C': 5, 'R': 6, 'Q': 7, 'G': 8, 'F': 9, 
                       'P': 10, 'S': 11, 'H': 12, 'Y': 13, 'D': 14, 'T': 15, 'L': 16, 'E': 17, 'A': 18, 'I': 19}
    peptide_onehot_list = [F.one_hot(torch.tensor(itemgetter(*list(i))(amino_acid_dict)), num_classes = 20) for i in peptide_list] # shape: (num_peptides, peptide_length, num_amino_acid)
    peptide_onehot_list_padding = pad_sequence(peptide_onehot_list, batch_first = True, padding_value=0).permute(0, 2, 1).float() # shape: (num_peptides, num_amino_acid, max_peptide_length)
    expression_data = torch.from_numpy(expression_data) # shape: (num_peptides, num_cells)
    print('num_peptides: {}, num_amino_acid: {}, max_peptide_length: {}'.format(peptide_onehot_list_padding.shape[0], peptide_onehot_list_padding.shape[1], peptide_onehot_list_padding.shape[2]))
    print('num_peptides: {}, num_cells: {}'.format(expression_data.shape[0], expression_data.shape[1]))

    indices = list(range(peptide_onehot_list_padding.shape[0]))
    train_indices = np.random.choice(indices, size=int((args.percentage) * len(indices)), replace=False)
    x_train = peptide_onehot_list_padding[train_indices]
    y_train = expression_data[train_indices]
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Peptide_Cell_Uncertainty(num_amino_acid=20, max_peptide_length=peptide_onehot_list_padding.shape[2],
                                     conv_layers=args.conv_layers, kernel_nums=args.kernel_nums, kernel_sizes=args.kernel_sizes, max_pool_size=args.max_pool_size, 
                                     hidden_dim=args.hidden_dim, output_dim=2*len(cell_list), dropout_rate=args.dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Train model for peptide uncertainty estimation
    best_loss = np.inf
    not_improved_count = 0
    for epoch in range(args.num_epochs):
        model.train()
        loss_total = 0.
        for _, (batch_x, batch_y) in enumerate(loader):
            optimizer.zero_grad()
            batch_x = batch_x.to(device) # shape: (batch_size, num_amino_acid, max_peptide_length)
            batch_y = batch_y.to(device) # shape: (batch_size, num_cells)
            batch_x_output = model(batch_x)
            loss = model.loss_uncertainty(batch_x_output, batch_y)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
        
        print('epoch {}, training_loss {}'.format(epoch, loss_total))

        if loss_total < best_loss:
            best_loss = loss_total
            not_improved_count = 0
            torch.save(model.state_dict(), os.path.join(result_dir, 'best_uncertainty_estimation.pth'))
        else:
            not_improved_count += 1
            if not_improved_count == args.patience:
                break

    # Save peptide uncertainty
    model.load_state_dict(torch.load(os.path.join(result_dir, 'best_uncertainty_estimation.pth')))
    model.eval()
    y_predict_all = model(peptide_onehot_list_padding.to(device))
    y_predict_all = y_predict_all.cpu().detach().numpy()
    mu = y_predict_all[:,0::2]
    log_uncertainty = y_predict_all[:,1::2]
    uncertainty = np.exp(log_uncertainty) # shape: (num_peptides, num_cells), represent the standard deviation of the Gaussian distribution.
    np.savez(os.path.join(result_dir, 'mu.npz'), mu=mu, cell_list=cell_list, protein_list=protein_list, peptide_list=peptide_list)
    np.savez(os.path.join(result_dir, 'uncertainty.npz'), uncertainty=uncertainty, cell_list=cell_list, protein_list=protein_list, peptide_list=peptide_list)
    
