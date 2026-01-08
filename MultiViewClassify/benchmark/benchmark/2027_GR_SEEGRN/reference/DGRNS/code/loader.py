# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
import pandas as pd

from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class DGRNS_Data(Dataset):
    def __init__(self, exp_file, data_file, data_cell=5000, data_size=1):
        node_vector = pd.read_csv(exp_file, index_col=0, header=0).values # shape: (n_gene, n_feature)
        node_name = pd.read_csv(exp_file, index_col=0, header=0).index.values # shape: (n_gene, )
        node_pair = pd.read_csv(data_file, index_col=None, header=0).values # shape: (n_pair, 3)

        # Feature Normalization
        node_vector = StandardScaler().fit_transform(node_vector.T).T # shape: (n_gene, n_select_feature) 

        # For Different Cell Number
        if data_cell < node_vector.shape[1]:
            np.random.seed(0)
            index = np.random.choice(node_vector.shape[1], int(data_cell), replace=False)
            index = np.sort(index)
            node_vector = node_vector[:, index] # shape: (n_gene, data_cell)

        # For Different Data Size
        np.random.seed(0)
        index = np.random.choice(node_pair.shape[0], int(node_pair.shape[0] * data_size), replace=False)
        index = np.sort(index)
        node_pair = node_pair[index, :] # shape: (n_select_pair, 3)

        self.node_name_vector_dict = {name: node_vector[i, :] for i, name in enumerate(node_name)} # shape: (n_gene, n_select_feature)
        self.node_pair_list = node_pair.tolist() # shape: (n_select_pair, 3)
        self.cell_number = node_vector.shape[1]

        # DGRNS parameters
        # cell_number = windows_size + tf_gap * (m - 1) + target_gap * (n - 1), while feature matrix shape: (m, n)
        self.m = 8
        self.n = 8
        self.tf_gap = 5
        self.target_gap = 5
        self.window_size = self.cell_number - self.tf_gap * (self.m - 1) - self.target_gap * (self.n - 1)  
        print('windows_size is less than 0, please check the parameters of DGRNS_Data') if self.window_size < 0 else None

    def __len__(self):
        return len(self.node_pair_list)
    
    def __getitem__(self, idx):
        # Load the node pair and the label
        node_pair = self.node_pair_list[idx]
        node_1_name = node_pair[0]
        node_2_name = node_pair[1]
        label = node_pair[2]

        # Load the node vector
        node_1_vector = self.node_name_vector_dict[node_1_name][:self.cell_number]
        node_2_vector = self.node_name_vector_dict[node_2_name][:self.cell_number]
        label = label
        
        # Generate the sample
        sample = self.DGRNS(node_1_vector, node_2_vector) # shape: (m, n)

        # Convert the sample to tensor
        sample = torch.from_numpy(sample).float() # shape: (m, n)
        label = torch.from_numpy(np.array(label)).long() # shape: (1)

        return sample, label
    
    def DGRNS(self, node_1_vector, node_2_vector):
        if self.window_size < 0:
            return np.random.rand(self.m, self.n)
        else:
            # Generate the feature matrix
            for i in range(0, self.m):
                if i == 0:
                    tf_matrix = node_1_vector[:self.window_size]
                else:
                    tf_matrix = np.vstack((tf_matrix, node_1_vector[i * self.tf_gap:i * self.tf_gap + self.window_size]))
            
            for j in range(0, self.n):
                if j == 0:
                    target_matrix = node_2_vector[:self.window_size]
                else:
                    target_matrix = np.vstack((target_matrix, node_2_vector[j * self.target_gap:j * self.target_gap + self.window_size]))
            
            # # calculate the pearson correlation coefficient
            # pcc_matrix = np.zeros((self.m, self.n))
            # for i in range(0, self.m):
            #     for j in range(0, self.n):
            #         xi = pd.Series(tf_matrix[i, :])
            #         yi = pd.Series(target_matrix[j, :])
            #         pcc = xi.corr(yi, method='pearson')
            #         pcc_matrix[i, j] = xi.corr(yi, method='pearson') if not np.isnan(pcc) else 0

            pcc_matrix = np.matmul(tf_matrix, target_matrix.T) # shape: (m, n)
        
        return pcc_matrix
