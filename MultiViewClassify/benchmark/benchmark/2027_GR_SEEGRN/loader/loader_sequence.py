# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from base import BaseLoader
from torchvision import datasets, transforms

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class HomoSapiens(Dataset):
    def __init__(self, logger=None, mode='train', feature='onehot'):
        self.logger = logger
        self.mode = mode
        self.feature = feature
        assert self.mode in ['train', 'test'], 'mode must be train or test'
        assert self.feature in ['onehot', 'vector', 'bert'], 'feature must be onehot or vector or bert'

        regulation_file_dict = {'train': '../data/species_specific_data/ProcessedData/HomoSapiens/regulation_train.csv', 
                                'test': '../data/species_specific_data/ProcessedData/HomoSapiens/regulation_test.csv'}
        regulation_file = regulation_file_dict[self.mode]
        regulation_file = pd.read_csv(regulation_file, header=None, sep=',')
        self.edge_list = regulation_file.values[:, :2]
        self.label_list = regulation_file.values[:, 2].astype(np.int64)

        protein_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/HomoSapiens/protein.seq', header=None, sep='\t')
        promoter_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/HomoSapiens/promoter.seq', header=None, sep='\t')
        self.protein_seq_dict = dict(zip(protein_seq_file[0], protein_seq_file[1]))
        self.promoter_seq_dict = dict(zip(promoter_seq_file[0], promoter_seq_file[1]))

        protein_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.onehot.npy',
                                     'vector': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.vector.npy',
                                     'bert': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.bert.npy'}
        promoter_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.onehot.npy',
                                      'vector': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.vector.npy',
                                      'bert': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.bert.npy'}
        protein_feature_file = protein_feature_file_dict[self.feature]
        promoter_feature_file = promoter_feature_file_dict[self.feature]
        self.protein_feature_dict = np.load(protein_feature_file, allow_pickle=True).item()
        self.promoter_feature_dict = np.load(promoter_feature_file, allow_pickle=True).item()

        if logger is not None:
            self.logger.info('Homo Sapiens Dataset: ({} {})'.format(self.mode, self.feature))
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
        LABEL = self.label_list[i]     
        TF_seq = self.protein_seq_dict[TF]
        GENE_seq = self.promoter_seq_dict[GENE]
        TF_feature = self.protein_feature_dict[TF]
        GENE_feature = self.promoter_feature_dict[GENE]
        TF_len = len(self.protein_seq_dict[TF][:1000])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len}, 'label': LABEL}


class HomosapiensLoader_sequence_pretrain(BaseLoader):
    """
    Homo Sapiens data loading demo using BaseDataLoader
    """
    def __init__(self, logger, batch_size, shuffle=True, validation_split=0.1, num_workers=1, mode='train', feature='onehot'):
        if mode == 'train':
            self.dataset = HomoSapiens(logger=logger, mode=mode, feature=feature)
            super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workers, validation_split=validation_split, collate_fn=self.collate_fn)
        elif mode == 'test':
            self.dataset = HomoSapiens(logger=logger, mode=mode, feature=feature)
            super().__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn)
        else:
            raise ValueError('mode should be train or test')
        
    def collate_fn(self, data_tuple):
        data_tuple.sort(key=lambda x: x['input']['TF_len'], reverse=True)
        TF = default_collate([data['input']['TF'] for data in data_tuple])
        GENE = default_collate([data['input']['GENE'] for data in data_tuple])
        TF_seq = default_collate([data['input']['TF_seq'] for data in data_tuple])
        GENE_seq = default_collate([data['input']['GENE_seq'] for data in data_tuple])
        TF_feature = default_collate([data['input']['TF_feature'] for data in data_tuple])
        GENE_feature = default_collate([data['input']['GENE_feature'] for data in data_tuple])
        TF_len = default_collate([data['input']['TF_len'] for data in data_tuple])
        LABEL = default_collate([data['label'] for data in data_tuple])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len}, 'label': LABEL}


class MusMusculus(Dataset):
    def __init__(self, logger=None, mode='train', feature='onehot'):
        self.logger = logger
        self.mode = mode
        self.feature = feature
        assert self.mode in ['train', 'test'], 'mode must be train or test'
        assert self.feature in ['onehot', 'vector', 'bert'], 'feature must be onehot or vector or bert'

        regulation_file_dict = {'train': '../data/species_specific_data/ProcessedData/MusMusculus/regulation_train.csv', 
                                'test': '../data/species_specific_data/ProcessedData/MusMusculus/regulation_test.csv'}
        regulation_file = regulation_file_dict[self.mode]
        regulation_file = pd.read_csv(regulation_file, header=None, sep=',')
        self.edge_list = regulation_file.values[:, :2]
        self.label_list = regulation_file.values[:, 2].astype(np.int64)

        protein_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/MusMusculus/protein.seq', header=None, sep='\t')
        promoter_seq_file = pd.read_csv('../data/species_specific_data/ProcessedData/MusMusculus/promoter.seq', header=None, sep='\t')
        self.protein_seq_dict = dict(zip(protein_seq_file[0], protein_seq_file[1]))
        self.promoter_seq_dict = dict(zip(promoter_seq_file[0], promoter_seq_file[1]))

        protein_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/MusMusculus/protein.onehot.npy',
                                     'vector': '../data/species_specific_data/ProcessedData/MusMusculus/protein.vector.npy',
                                     'bert': '../data/species_specific_data/ProcessedData/MusMusculus/protein.bert.npy'}
        promoter_feature_file_dict = {'onehot': '../data/species_specific_data/ProcessedData/MusMusculus/promoter.onehot.npy',
                                      'vector': '../data/species_specific_data/ProcessedData/MusMusculus/promoter.vector.npy',
                                      'bert': '../data/species_specific_data/ProcessedData/MusMusculus/promoter.bert.npy'}
        protein_feature_file = protein_feature_file_dict[self.feature]
        promoter_feature_file = promoter_feature_file_dict[self.feature]
        self.protein_feature_dict = np.load(protein_feature_file, allow_pickle=True).item()
        self.promoter_feature_dict = np.load(promoter_feature_file, allow_pickle=True).item()

        if logger is not None:
            self.logger.info('Mus Musculus Dataset: (mode: {} {})'.format(self.mode, self.feature))
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
        LABEL = self.label_list[i]     
        TF_seq = self.protein_seq_dict[TF]
        GENE_seq = self.promoter_seq_dict[GENE]
        TF_feature = self.protein_feature_dict[TF]
        GENE_feature = self.promoter_feature_dict[GENE]
        TF_len = len(self.protein_seq_dict[TF][:1000])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len}, 'label': LABEL}


class MusmusculusLoader_sequence_pretrain(BaseLoader):
    """
    Mus musculus data loading demo using BaseDataLoader
    """
    def __init__(self, logger, batch_size, shuffle=True, validation_split=0.1, num_workers=1, mode='train', feature='onehot'):
        if mode == 'train':
            self.dataset = MusMusculus(logger=logger, mode=mode, feature=feature)
            super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workers, validation_split=validation_split, collate_fn=self.collate_fn)
        elif mode == 'test':
            self.dataset = MusMusculus(logger=logger, mode=mode, feature=feature)
            super().__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn)
        else:
            raise ValueError('mode should be train or test')
        
    def collate_fn(self, data_tuple):
        data_tuple.sort(key=lambda x: x['input']['TF_len'], reverse=True)
        TF = default_collate([data['input']['TF'] for data in data_tuple])
        GENE = default_collate([data['input']['GENE'] for data in data_tuple])
        TF_seq = default_collate([data['input']['TF_seq'] for data in data_tuple])
        GENE_seq = default_collate([data['input']['GENE_seq'] for data in data_tuple])
        TF_feature = default_collate([data['input']['TF_feature'] for data in data_tuple])
        GENE_feature = default_collate([data['input']['GENE_feature'] for data in data_tuple])
        TF_len = default_collate([data['input']['TF_len'] for data in data_tuple])
        LABEL = default_collate([data['label'] for data in data_tuple])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len}, 'label': LABEL}


class CrossSpecies(Dataset):
    def __init__(self, logger=None, mode='train', feature='onehot'):
        self.logger = logger
        self.mode = mode
        self.feature = feature
        assert self.mode in ['train', 'test'], 'mode must be train or test'
        assert self.feature in ['onehot', 'vector', 'bert'], 'feature must be onehot or vector or bert'

        regulation_file_dict_homo = {'train': '../data/species_specific_data/ProcessedData/HomoSapiens/regulation_train.csv', 
                                     'test': '../data/species_specific_data/ProcessedData/HomoSapiens/regulation_test.csv'}
        regulation_file_dict_mus = {'train': '../data/species_specific_data/ProcessedData/MusMusculus/regulation_train.csv', 
                                    'test': '../data/species_specific_data/ProcessedData/MusMusculus/regulation_test.csv'}
        regulation_file_homo = regulation_file_dict_homo[self.mode]
        regulation_file_mus = regulation_file_dict_mus[self.mode]
        regulation_file_homo = pd.read_csv(regulation_file_homo, header=None, sep=',')
        regulation_file_mus = pd.read_csv(regulation_file_mus, header=None, sep=',')
        self.edge_list_homo = np.array(regulation_file_homo.values[:, :2])
        self.edge_list_mus = np.array(regulation_file_mus.values[:, :2])
        self.label_list_homo = np.array(regulation_file_homo.values[:, 2]).astype(np.int64)
        self.label_list_mus = np.array(regulation_file_mus.values[:, 2]).astype(np.int64)

        protein_seq_file_homo = pd.read_csv('../data/species_specific_data/ProcessedData/HomoSapiens/protein.seq', header=None, sep='\t')
        promoter_seq_file_homo = pd.read_csv('../data/species_specific_data/ProcessedData/HomoSapiens/promoter.seq', header=None, sep='\t')
        protein_seq_file_mus = pd.read_csv('../data/species_specific_data/ProcessedData/MusMusculus/protein.seq', header=None, sep='\t')
        promoter_seq_file_mus = pd.read_csv('../data/species_specific_data/ProcessedData/MusMusculus/promoter.seq', header=None, sep='\t')
        self.protein_seq_dict_homo = dict(zip(protein_seq_file_homo[0], protein_seq_file_homo[1]))
        self.promoter_seq_dict_homo = dict(zip(promoter_seq_file_homo[0], promoter_seq_file_homo[1]))
        self.protein_seq_dict_mus = dict(zip(protein_seq_file_mus[0], protein_seq_file_mus[1]))
        self.promoter_seq_dict_mus = dict(zip(promoter_seq_file_mus[0], promoter_seq_file_mus[1]))

        protein_feature_file_dict_homo = {'onehot': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.onehot.npy',
                                          'vector': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.vector.npy',
                                          'bert': '../data/species_specific_data/ProcessedData/HomoSapiens/protein.bert.npy'}
        protein_feature_file_dict_mus = {'onehot': '../data/species_specific_data/ProcessedData/MusMusculus/protein.onehot.npy',
                                         'vector': '../data/species_specific_data/ProcessedData/MusMusculus/protein.vector.npy',
                                         'bert': '../data/species_specific_data/ProcessedData/MusMusculus/protein.bert.npy'}
        promoter_feature_file_dict_homo = {'onehot': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.onehot.npy',
                                           'vector': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.vector.npy',
                                           'bert': '../data/species_specific_data/ProcessedData/HomoSapiens/promoter.bert.npy'}
        promoter_feature_file_dict_mus = {'onehot': '../data/species_specific_data/ProcessedData/MusMusculus/promoter.onehot.npy',
                                          'vector': '../data/species_specific_data/ProcessedData/MusMusculus/promoter.vector.npy',
                                          'bert': '../data/species_specific_data/ProcessedData/MusMusculus/promoter.bert.npy'}
        protein_feature_file_homo = protein_feature_file_dict_homo[self.feature]
        protein_feature_file_mus = protein_feature_file_dict_mus[self.feature]
        promoter_feature_file_homo = promoter_feature_file_dict_homo[self.feature]
        promoter_feature_file_mus = promoter_feature_file_dict_mus[self.feature]
        self.protein_feature_dict_homo = np.load(protein_feature_file_homo, allow_pickle=True).item()
        self.protein_feature_dict_mus = np.load(protein_feature_file_mus, allow_pickle=True).item()
        self.promoter_feature_dict_homo = np.load(promoter_feature_file_homo, allow_pickle=True).item()
        self.promoter_feature_dict_mus = np.load(promoter_feature_file_mus, allow_pickle=True).item()

        if logger is not None:
            self.logger.info('Cross Species Dataset: ({} {})'.format(self.mode, self.feature))
            self.logger.info('Homo Sapiens Dataset: ({} {})'.format(self.mode, self.feature))
            self.logger.info('TF: {}'.format(len(np.unique(self.edge_list_homo[:, 0]))))
            self.logger.info('GENE: {}'.format(len(np.unique(np.concatenate([self.edge_list_homo[:, 0], self.edge_list_homo[:, 1]])))))
            self.logger.info('POS_EDGE: {}'.format(np.sum(self.label_list_homo)))
            self.logger.info('NEG_EDGE: {}'.format(len(self.label_list_homo) - np.sum(self.label_list_homo)))
            self.logger.info('POS / NEG: {}'.format(np.sum(self.label_list_homo) / (len(self.label_list_homo) - np.sum(self.label_list_homo))))
            self.logger.info('Density: {}'.format(np.sum(self.label_list_homo) / len(self.label_list_homo)))

            self.logger.info('Mus Musculus Dataset: (mode: {} {})'.format(self.mode, self.feature))
            self.logger.info('TF: {}'.format(len(np.unique(self.edge_list_mus[:, 0]))))
            self.logger.info('GENE: {}'.format(len(np.unique(np.concatenate([self.edge_list_mus[:, 0], self.edge_list_mus[:, 1]])))))
            self.logger.info('POS_EDGE: {}'.format(np.sum(self.label_list_mus)))
            self.logger.info('NEG_EDGE: {}'.format(len(self.label_list_mus) - np.sum(self.label_list_mus)))
            self.logger.info('POS / NEG: {}'.format(np.sum(self.label_list_mus) / (len(self.label_list_mus) - np.sum(self.label_list_mus))))
            self.logger.info('Density: {}'.format(np.sum(self.label_list_mus) / len(self.label_list_mus)))

    def __len__(self):
        return len(self.label_list_homo) + len(self.label_list_mus)
        
    def __getitem__(self, i):
        if i < len(self.label_list_homo):
            TF, GENE = self.edge_list_homo[i]
            LABEL = self.label_list_homo[i]     
            TF_seq = self.protein_seq_dict_homo[TF]
            GENE_seq = self.promoter_seq_dict_homo[GENE]
            TF_feature = self.protein_feature_dict_homo[TF]
            GENE_feature = self.promoter_feature_dict_homo[GENE]
            TF_len = len(self.protein_seq_dict_homo[TF][:1000])
        else:
            i = i - len(self.label_list_homo)
            TF, GENE = self.edge_list_mus[i]
            LABEL = self.label_list_mus[i]     
            TF_seq = self.protein_seq_dict_mus[TF]
            GENE_seq = self.promoter_seq_dict_mus[GENE]
            TF_feature = self.protein_feature_dict_mus[TF]
            GENE_feature = self.promoter_feature_dict_mus[GENE]
            TF_len = len(self.protein_seq_dict_mus[TF][:1000])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len}, 'label': LABEL}


class CrossspeciesLoader_sequence_pretrain(BaseLoader):
    """
    Cross Species data loading demo using BaseDataLoader
    """
    def __init__(self, logger, batch_size, shuffle=True, validation_split=0.1, num_workers=1, mode='train', feature='onehot'):
        if mode == 'train':
            self.dataset = CrossSpecies(logger=logger, mode=mode, feature=feature)
            super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workers, validation_split=validation_split, collate_fn=self.collate_fn)
        elif mode == 'test':
            self.dataset = CrossSpecies(logger=logger, mode=mode, feature=feature)
            super().__init__(self.dataset, batch_size, shuffle=False, num_workers=num_workers, validation_split=0, collate_fn=self.collate_fn)
        else:
            raise ValueError('mode should be train or test')
        
    def collate_fn(self, data_tuple):
        data_tuple.sort(key=lambda x: x['input']['TF_len'], reverse=True)
        TF = default_collate([data['input']['TF'] for data in data_tuple])
        GENE = default_collate([data['input']['GENE'] for data in data_tuple])
        TF_seq = default_collate([data['input']['TF_seq'] for data in data_tuple])
        GENE_seq = default_collate([data['input']['GENE_seq'] for data in data_tuple])
        TF_feature = default_collate([data['input']['TF_feature'] for data in data_tuple])
        GENE_feature = default_collate([data['input']['GENE_feature'] for data in data_tuple])
        TF_len = default_collate([data['input']['TF_len'] for data in data_tuple])
        LABEL = default_collate([data['label'] for data in data_tuple])
        return {'input': {'TF': TF, 'GENE': GENE, 'TF_seq': TF_seq, 'GENE_seq': GENE_seq, 'TF_feature': TF_feature, 'GENE_feature': GENE_feature, 'TF_len': TF_len}, 'label': LABEL}
    
    