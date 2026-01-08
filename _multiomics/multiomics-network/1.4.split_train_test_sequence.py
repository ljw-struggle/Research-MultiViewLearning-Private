# -*- coding: utf-8 -*-
import os
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split


cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.INFO)
cons_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler = logging.FileHandler(filename='species_specific_data/ProcessedData/split_train_test.log', mode='a+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('PreProcess')
logger.addHandler(cons_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def split_train_test(data_dir, test_size=0.2, random_state=0):
    whole_data_path = os.path.join(data_dir, 'regulation.csv')
    train_data_path = os.path.join(data_dir, 'regulation_train.csv')
    test_data_path = os.path.join(data_dir, 'regulation_test.csv')
    data = pd.read_csv(whole_data_path, sep=',', header=0, low_memory=False).values

    TF_list = np.unique(data[:, 0])
    TF_dict = dict(zip(TF_list, range(len(TF_list))))
    GENE_list = np.unique(np.concatenate([np.unique(data[:, 0]), np.unique(data[:, 1])], axis=0))
    GENE_dict = dict(zip(GENE_list, range(len(GENE_list))))

    NEG_EDGE_list = []
    adj_matrix = np.zeros((len(TF_list), len(GENE_list)))
    for i in tqdm(range(len(data)), ascii=True, desc='Constructing adjacency matrix'):
        TF = data[i, 0]
        GENE = data[i, 1]
        adj_matrix[TF_dict[TF], GENE_dict[GENE]] = 1
    
    for i in tqdm(range(len(TF_list)), ascii=True, desc='Constructing negative edges'):
        for j in range(len(GENE_list)):
            if adj_matrix[i, j] == 0 and TF_list[i] != GENE_list[j]:
                NEG_EDGE_list.append([TF_list[i], GENE_list[j]])
    
    data = np.concatenate((data, NEG_EDGE_list), axis=0)
    label = np.concatenate((np.ones(len(data)-len(NEG_EDGE_list)), np.zeros(len(NEG_EDGE_list))), axis=0)
    
    logger.info('Begin to split data into train and test set')
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=test_size, random_state=random_state)

    pd.DataFrame(np.concatenate((data_train, label_train.reshape(-1, 1).astype(int)), axis=1)).to_csv(train_data_path, sep=',', header=None, index=None)
    pd.DataFrame(np.concatenate((data_test, label_test.reshape(-1, 1).astype(int)), axis=1)).to_csv(test_data_path, sep=',', header=None, index=None)

    logger.info('Train data shape of unbalanced data: {}'.format(data_train.shape))
    logger.info('Train data POS/NEG of unbalanced data: {}/{} = {}'.format(np.sum(label_train), len(label_train)-np.sum(label_train), np.sum(label_train)/(len(label_train)-np.sum(label_train))))
    logger.info('Test data shape of unbalanced data: {}'.format(data_test.shape))
    logger.info('Test data POS/NEG of unbalanced data: {}/{} = {}'.format(np.sum(label_test), len(label_test)-np.sum(label_test), np.sum(label_test)/(len(label_test)-np.sum(label_test))))
    return data_train, data_test


if __name__ == '__main__':
    processed_data_dir = './species_specific_data/ProcessedData/'

    # Homo sapiens
    logger.info('HomoSapiens start.')
    split_train_test(os.path.join(processed_data_dir, 'HomoSapiens'))
    logger.info('HomoSapiens done.')

    # Mus musculus
    logger.info('MusMusculus start.')
    split_train_test(os.path.join(processed_data_dir, 'MusMusculus'))
    logger.info('MusMusculus done.')
