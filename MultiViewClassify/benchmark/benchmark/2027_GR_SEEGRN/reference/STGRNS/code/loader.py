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


class STGRNS_Data(Dataset):
    def __init__(self, exp_file, data_file, data_cell=5000, data_size=1):
        node_vector = pd.read_csv(exp_file, index_col=0, header=0).values # shape: (n_gene, n_feature)
        node_name = pd.read_csv(exp_file, index_col=0, header=0).index.values # shape: (n_gene, )
        node_pair = pd.read_csv(data_file, index_col=None, header=0).values # shape: (n_pair, 3)

        # Feature Normalization
        node_vector = StandardScaler().fit_transform(node_vector.T).T # shape: (n_gene, n_feature) 
        
        # For Different Data Ratio
        if data_cell < node_vector.shape[1]:
            np.random.seed(0)
            index = np.random.choice(node_vector.shape[1], int(data_cell), replace=False)
            index = np.sort(index)
            node_vector = node_vector[:, index] # shape: (n_gene, data_cell)

        if node_vector.shape[1] < 100:
            np.random.seed(0)
            node_vector = np.random.normal(0, 1, [node_vector.shape[0], 100])

        # For Different Data Size
        np.random.seed(0)
        index = np.random.choice(node_pair.shape[0], int(node_pair.shape[0] * data_size), replace=False)
        index = np.sort(index)
        node_pair = node_pair[index, :] # shape: (n_select_pair, 3)

        self.node_name_vector_dict = {name: node_vector[i, :] for i, name in enumerate(node_name)}
        self.node_pair_list = node_pair.tolist() # shape: (n_select_pair, 3)

        self.cell_number = (node_vector.shape[1] // 100) * 100 # make sure the cell number is a multiple of 100
        assert self.cell_number >= 100, 'cell_number is less than 100, please check the parameters of STGRNS_Data'

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
        node_1_vector = node_1_vector.reshape(-1, 100) # shape: (self.cell_number//100, 100)
        node_2_vector = node_2_vector.reshape(-1, 100) # shape: (self.cell_number//100, 100)
        sample = np.concatenate((node_1_vector, node_2_vector), axis=1) # shape: (self.cell_number//100, 200)        

        # Convert the sample to tensor
        sample = torch.from_numpy(sample).float() # shape: (self.cell_number//100, 200)
        label = torch.from_numpy(np.array(label)).long() # shape: (1, )

        return sample, label, node_1_name, node_2_name

