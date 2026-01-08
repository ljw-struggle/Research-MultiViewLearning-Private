# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd

from base import BaseLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class Tissue_Specific_Data(Dataset):
    def __init__(self, logger, mode='train', feature='onehot', data='Adult-Heart', train_cell_number=5000, train_data_size=1.0):
        self.logger = logger
        self.mode = mode
        self.feature = feature
        self.data = data
        self.train_cell_number = train_cell_number
        self.train_data_size = train_data_size
        assert self.mode in ['train', 'valid', 'test'], 'mode must be train or valid or test'
        assert self.feature in ['onehot', 'vector', 'bert'], 'feature must be onehot or vector or bert'
        # assert self.data in ('Human-Blood', 'Human-Blood-Transfer', 'Human-Bone-Marrow', 'Human-Cerebral-Cortex', 'Mouse-Brain', 'Mouse-Brain-Cortex', 'Mouse-Brain-Cortex-Transfer', 'Mouse-Lung', 'Mouse-Skin'), \
        #                      'data must be Human-Blood or Human-Bone-Marrow or Human-Cerebral-Cortex or Mouse-Brain or Mouse-Brain-Cortex or Mouse-Lung or Mouse-Skin'
        assert self.train_cell_number in [10, 20, 50, 500, 2000, 5000], 'train_data_ratio must be 0.2 or 0.4 or 0.6 or 0.8 or 1.0'
        assert self.train_data_size in [0.05, 0.1, 0.2, 0.5, 1.0], 'train_data_size must be 0.2 or 0.4 or 0.6 or 0.8 or 1.0'

        data_path = '../data/tissue_specific_data/ProcessedData/{}'.format(self.data)

        # Load gene regulation file
        regulation_file_dict = {'train': os.path.join(data_path, 'train_set.csv'),
                                'valid': os.path.join(data_path, 'val_set.csv'),
                                'test': os.path.join(data_path, 'test_set.csv')}
        regulation_file = regulation_file_dict[self.mode]
        regulation_file = pd.read_csv(regulation_file, header=0, sep=',')
        edge_list = regulation_file.values[:, :2]
        label_list = regulation_file.values[:, 2].astype(np.int64)
        weight_list = torch.tensor([1 if label_list[i] == 0 else ((len(label_list) - np.sum(label_list)) / np.sum(label_list)) for i in range(len(label_list))], dtype=torch.float)

        np.random.seed(0)
        index = np.random.choice(len(label_list), int(len(label_list) * self.train_data_size), replace=False)
        index = np.sort(index)
        self.edge_list = edge_list[index]
        self.label_list = label_list[index]
        self.weight_list = weight_list[index]

        # Load gene expression file
        gene_sc_expression = pd.read_csv(os.path.join(data_path, 'expression.csv'), header=0, index_col=0, low_memory=False) # shape = (gene_num, cell_num)
        gene_list = np.array(gene_sc_expression.index.values, dtype=np.str_) # shape = (gene_num)
        gene_index_dict = dict(zip(gene_list, range(len(gene_list))))
        index_gene_dict = dict(zip(range(len(gene_list)), gene_list))

        np.random.seed(0)
        print('gene_sc_expression.shape = {}'.format(gene_sc_expression.shape))
        print('train_cell_number = {}'.format(self.train_cell_number))
        index = np.random.choice(gene_sc_expression.shape[1], self.train_cell_number, replace=False)
        index = np.sort(index)
        gene_sc_expression = pd.DataFrame(MinMaxScaler().fit_transform(gene_sc_expression.iloc[:, index].values.T).T, index=None, columns=None) # shape = (gene_num, cell_num)
        gene_correlation_coefficient = np.abs(gene_sc_expression.T.corr(method='spearman')) > 0.5 # shape = (gene_num, gene_num)
        co_expression_gene_edge = np.where(np.triu(gene_correlation_coefficient, k=1) == True) # shape = (2, edge_num)
        prior_graph_topology = np.stack([co_expression_gene_edge[0], co_expression_gene_edge[1]], axis=0) # shape = (2, edge_num)
        self.gene_sc_expression = torch.tensor(gene_sc_expression.values, dtype=torch.float) # shape = (gene_num, cell_num)
        self.prior_graph_topology = torch.tensor(prior_graph_topology, dtype=torch.long)
        self.gene_index_dict = gene_index_dict
        self.index_gene_dict = index_gene_dict

        # Load gene sequence file
        # if self.data include 'Mouse', load mouse sequence file
        species = 'HomoSapiens' if 'Human' in self.data else 'MusMusculus'
        protein_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/{}/protein.seq'.format(species), header=None, sep='\t')
        protein_seq_file[0] = [protein_seq_file[0][i].upper() for i in range(len(protein_seq_file[0]))]
        promoter_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/{}/promoter.seq'.format(species), header=None, sep='\t')
        promoter_seq_file[0] = [promoter_seq_file[0][i].upper() for i in range(len(promoter_seq_file[0]))]
        self.protein_seq_dict = dict(zip(protein_seq_file[0], protein_seq_file[1]))
        self.promoter_seq_dict = dict(zip(promoter_seq_file[0], promoter_seq_file[1]))
        protein_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/{}/protein.onehot.npy'.format(species),
                                    'vector': '../data/species_specific_data/ProcessedData/{}/protein.vector.npy'.format(species),
                                    'bert': '../data/species_specific_data/ProcessedData/{}/protein.bert.npy'.format(species)}
        promoter_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/{}/promoter.onehot.npy'.format(species),
                                    'vector': '../data/species_specific_data/ProcessedData/{}/promoter.vector.npy'.format(species),
                                    'bert': '../data/species_specific_data/ProcessedData/{}/promoter.bert.npy'.format(species)}
        protein_feature_file = np.load(protein_feature_file_dict[self.feature], allow_pickle=True).item()
        promoter_feature_file = np.load(promoter_feature_file_dict[self.feature], allow_pickle=True).item()
        self.protein_feature_dict = {k.upper(): v for k, v in protein_feature_file.items()}
        self.promoter_feature_dict = {k.upper(): v for k, v in promoter_feature_file.items()}
        
        if logger is not None:
            self.logger.info('Tissue Specific Dataset: ({} {} {})'.format(self.mode, self.feature, self.data))
            self.logger.info('TF: {}'.format(len(np.unique(self.edge_list[:, 0]))))
            self.logger.info('GENE: {}'.format(len(np.unique(np.concatenate([self.edge_list[:, 0], self.edge_list[:, 1]])))))
            self.logger.info('POS_EDGE: {}'.format(np.sum(self.label_list)))
            self.logger.info('NEG_EDGE: {}'.format(len(self.label_list) - np.sum(self.label_list)))
            self.logger.info('POS / NEG: {}'.format(np.sum(self.label_list) / (len(self.label_list) - np.sum(self.label_list))))
            self.logger.info('Density: {}'.format(np.sum(self.label_list) / len(self.label_list)))

    def __len__(self):
        return len(self.label_list)
        
    def __getitem__(self, i):
        TF, GENE = self.edge_list[i]
        EDGE = np.array([self.gene_index_dict[TF], self.gene_index_dict[GENE]], dtype=np.int64)
        LABEL = self.label_list[i]     
        TF_seq = self.protein_seq_dict[TF]
        GENE_seq = self.promoter_seq_dict[GENE]
        TF_feature = self.protein_feature_dict[TF]
        GENE_feature = self.promoter_feature_dict[GENE]
        TF_len = len(self.protein_seq_dict[TF][:1000])
        WEIGHT = self.weight_list[i]
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len, 'EDGE': EDGE}, 'label': LABEL, 'weight': WEIGHT}
    

class Tissue_Specific_Loader(BaseLoader):
    """
    Tissue specific data loading demo using BaseDataLoader
    """
    def __init__(self, logger, batch_size, shuffle=True, num_workers=1, mode='train', feature='onehot', data='Human-Blood', train_cell_number=5000, train_data_size=1.0):
        if mode == 'train':
            self.dataset = Tissue_Specific_Data(logger=logger, mode=mode, feature=feature, data=data, train_cell_number=train_cell_number, train_data_size=train_data_size)
            super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn)
        elif mode == 'valid':
            self.dataset = Tissue_Specific_Data(logger=logger, mode=mode, feature=feature, data=data, train_cell_number=train_cell_number)
            super().__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn) # shuffle = True
        elif mode == 'test':
            self.dataset = Tissue_Specific_Data(logger=logger, mode=mode, feature=feature, data=data, train_cell_number=train_cell_number)
            super().__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn)
        else:
            raise ValueError('mode should be train, valid or test')
        
    def collate_fn(self, data_tuple):
        data_tuple.sort(key=lambda x: x['input']['TF_len'], reverse=True)
        TF = default_collate([data['input']['TF'] for data in data_tuple])
        GENE = default_collate([data['input']['GENE'] for data in data_tuple])
        EDGE = default_collate([data['input']['EDGE'] for data in data_tuple])
        TF_seq = default_collate([data['input']['TF_seq'] for data in data_tuple])
        GENE_seq = default_collate([data['input']['GENE_seq'] for data in data_tuple])
        TF_feature = default_collate([data['input']['TF_feature'] for data in data_tuple])
        GENE_feature = default_collate([data['input']['GENE_feature'] for data in data_tuple])
        TF_len = default_collate([data['input']['TF_len'] for data in data_tuple])
        LABEL = default_collate([data['label'] for data in data_tuple])
        WEIGHT = default_collate([data['weight'] for data in data_tuple])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len, 'EDGE': EDGE}, 'label': LABEL, 'weight': WEIGHT}
    
    def get_node_expression(self):
        return self.dataset.gene_sc_expression
    
    def get_prior_graph_topology(self):
        return self.dataset.prior_graph_topology
    
    def get_index_gene_dict(self):
        return self.dataset.index_gene_dict
    
    def get_gene_index_dict(self):
        return self.dataset.gene_index_dict


class Cell_Type_Specific_Data(Dataset):
    def __init__(self, logger, mode='train', feature='onehot', data='ERY'):
        self.logger = logger
        self.mode = mode
        self.feature = feature
        self.data = data
        assert self.mode in ['train', 'valid', 'test'], 'mode must be train or valid or test'
        assert self.feature in ['onehot', 'vector', 'bert'], 'feature must be onehot or vector or bert'
        assert self.data in ('ERY', 'HSC', 'MEP', 'CDC', 'CLP', 'HMP', 'MONO', 'PDC'), 'data must be ERY or HSC or MEP or CDC or CLP or HMP or MONO or PDC'

        data_path = '../data/cell_type_specific_data/ProcessedData/{}'.format(self.data)

        # Load gene regulation file
        regulation_file_dict = {'train': os.path.join(data_path, 'train_set.csv'),
                                'valid': os.path.join(data_path, 'val_set.csv'),
                                'test': os.path.join(data_path, 'test_set.csv')}
        regulation_file = regulation_file_dict[self.mode]
        regulation_file = pd.read_csv(regulation_file, header=0, sep=',')
        edge_list = regulation_file.values[:, :2]
        label_list = regulation_file.values[:, 2].astype(np.int64)
        weight_list = torch.tensor([1 if label_list[i] == 0 else ((len(label_list) - np.sum(label_list)) / np.sum(label_list)) for i in range(len(label_list))], dtype=torch.float)

        self.edge_list = edge_list
        self.label_list = label_list
        self.weight_list = weight_list

        # Load gene expression file
        gene_sc_expression = pd.read_csv(os.path.join(data_path, 'expression.csv'), header=0, index_col=0, low_memory=False) # shape = (gene_num, cell_num)
        gene_list = np.array(gene_sc_expression.index.values, dtype=np.str_) # shape = (gene_num)
        gene_index_dict = dict(zip(gene_list, range(len(gene_list))))
        index_gene_dict = dict(zip(range(len(gene_list)), gene_list))

        gene_sc_expression = pd.DataFrame(MinMaxScaler().fit_transform(gene_sc_expression.values.T).T, index=None, columns=None) # shape = (gene_num, cell_num)
        gene_correlation_coefficient = np.abs(gene_sc_expression.T.corr(method='spearman')) > 0.5 # shape = (gene_num, gene_num)
        co_expression_gene_edge = np.where(np.triu(gene_correlation_coefficient, k=1) == True) # shape = (2, edge_num)
        prior_graph_topology = np.stack([co_expression_gene_edge[0], co_expression_gene_edge[1]], axis=0) # shape = (2, edge_num)
        self.gene_sc_expression = torch.tensor(gene_sc_expression.values, dtype=torch.float) # shape = (gene_num, cell_num)
        self.prior_graph_topology = torch.tensor(prior_graph_topology, dtype=torch.long)
        self.gene_index_dict = gene_index_dict
        self.index_gene_dict = index_gene_dict

        # Load gene sequence file
        protein_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/HomoSapiens/protein.seq', header=None, sep='\t')
        protein_seq_file[0] = [protein_seq_file[0][i].upper() for i in range(len(protein_seq_file[0]))]
        promoter_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/HomoSapiens/promoter.seq', header=None, sep='\t')
        promoter_seq_file[0] = [promoter_seq_file[0][i].upper() for i in range(len(promoter_seq_file[0]))]
        self.protein_seq_dict = dict(zip(protein_seq_file[0], protein_seq_file[1]))
        self.promoter_seq_dict = dict(zip(promoter_seq_file[0], promoter_seq_file[1]))
        protein_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.onehot.npy',
                                    'vector': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.vector.npy',
                                    'bert': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.bert.npy'}
        promoter_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.onehot.npy',
                                    'vector': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.vector.npy',
                                    'bert': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.bert.npy'}
        protein_feature_file = np.load(protein_feature_file_dict[self.feature], allow_pickle=True).item()
        promoter_feature_file = np.load(promoter_feature_file_dict[self.feature], allow_pickle=True).item()
        self.protein_feature_dict = {k.upper(): v for k, v in protein_feature_file.items()}
        self.promoter_feature_dict = {k.upper(): v for k, v in promoter_feature_file.items()}

        if logger is not None:
            self.logger.info('Cell Type Specific Dataset: ({} {} {})'.format(self.mode, self.feature, self.data))
            self.logger.info('TF: {}'.format(len(np.unique(self.edge_list[:, 0]))))
            self.logger.info('GENE: {}'.format(len(np.unique(np.concatenate([self.edge_list[:, 0], self.edge_list[:, 1]])))))
            self.logger.info('POS_EDGE: {}'.format(np.sum(self.label_list)))
            self.logger.info('NEG_EDGE: {}'.format(len(self.label_list) - np.sum(self.label_list)))
            self.logger.info('POS / NEG: {}'.format(np.sum(self.label_list) / (len(self.label_list) - np.sum(self.label_list))))
            self.logger.info('Density: {}'.format(np.sum(self.label_list) / len(self.label_list)))

    def __len__(self):
        return len(self.label_list)
        
    def __getitem__(self, i):
        TF, GENE = self.edge_list[i]
        EDGE = np.array([self.gene_index_dict[TF], self.gene_index_dict[GENE]], dtype=np.int64)
        LABEL = self.label_list[i]     
        TF_seq = self.protein_seq_dict[TF]
        GENE_seq = self.promoter_seq_dict[GENE]
        TF_feature = self.protein_feature_dict[TF]
        GENE_feature = self.promoter_feature_dict[GENE]
        TF_len = len(self.protein_seq_dict[TF][:1000])
        WEIGHT = self.weight_list[i]
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len, 'EDGE': EDGE}, 'label': LABEL, 'weight': WEIGHT}
    

class Cell_Type_Specific_Loader(BaseLoader):
    """
    Cell type specific data loading demo using BaseDataLoader
    """
    def __init__(self, logger, batch_size, shuffle=True, num_workers=1, mode='train', feature='onehot', data='ERY'):
        if mode == 'train':
            self.dataset = Cell_Type_Specific_Data(logger=logger, mode=mode, feature=feature, data=data)
            super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn)
        elif mode == 'valid':
            self.dataset = Cell_Type_Specific_Data(logger=logger, mode=mode, feature=feature, data=data)
            super().__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn) # shuffle = True
        elif mode == 'test':
            self.dataset = Cell_Type_Specific_Data(logger=logger, mode=mode, feature=feature, data=data)
            super().__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn)
        else:
            raise ValueError('mode should be train, valid or test')
        
    def collate_fn(self, data_tuple):
        data_tuple.sort(key=lambda x: x['input']['TF_len'], reverse=True)
        TF = default_collate([data['input']['TF'] for data in data_tuple])
        GENE = default_collate([data['input']['GENE'] for data in data_tuple])
        EDGE = default_collate([data['input']['EDGE'] for data in data_tuple])
        TF_seq = default_collate([data['input']['TF_seq'] for data in data_tuple])
        GENE_seq = default_collate([data['input']['GENE_seq'] for data in data_tuple])
        TF_feature = default_collate([data['input']['TF_feature'] for data in data_tuple])
        GENE_feature = default_collate([data['input']['GENE_feature'] for data in data_tuple])
        TF_len = default_collate([data['input']['TF_len'] for data in data_tuple])
        LABEL = default_collate([data['label'] for data in data_tuple])
        WEIGHT = default_collate([data['weight'] for data in data_tuple])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len, 'EDGE': EDGE}, 'label': LABEL, 'weight': WEIGHT}
    
    def get_node_expression(self):
        return self.dataset.gene_sc_expression
    
    def get_prior_graph_topology(self):
        return self.dataset.prior_graph_topology
    
    def get_index_gene_dict(self):
        return self.dataset.index_gene_dict
    
    def get_gene_index_dict(self):
        return self.dataset.gene_index_dict
    