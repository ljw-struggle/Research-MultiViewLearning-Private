# -*- coding: utf-8 -*-
import os
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

from utils import train_test_val_split_hard, train_test_val_split_balance

# filter the warnings
warnings.filterwarnings('ignore')

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.INFO)
cons_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler = logging.FileHandler(filename='cell_type_specific_data/ProcessedData/expression_data.log', mode='a+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('PreProcess')
logger.addHandler(cons_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def preprocess_cell_type_specific_data(expression_file, regulation_file, balance, save_path):
    # 1\ Read expression data
    adata = sc.read_csv(expression_file, delimiter=',', first_column_names=True) # shape: (genes, cells)
    adata = adata.transpose() # shape: (cells, genes)
    adata.var_names = adata.var_names.str.upper() # convert gene names to uppercase
    if 'ERY' not in expression_file and 'HSC' not in expression_file and 'MEP' not in expression_file:
        hvg_index = sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=5000, inplace=False) # select highly variable genes
        adata = adata[:, hvg_index['highly_variable']] # filter highly variable genes
    expression_data = adata.X.transpose() # shape: (genes, cells)
    expression_data = pd.DataFrame(expression_data, index=adata.var_names, columns=adata.obs_names) # convert to dataframe

    # 2\ Read regulation data
    regulation_data = pd.read_csv(regulation_file, delimiter=',', header=0, index_col=0)
    regulation_data = regulation_data[(regulation_data['Score'] < -0.5) | (regulation_data['Score'] > 0.5)]
    regulation_data = regulation_data[['Motif', 'DORC']]
    regulation_data = regulation_data[regulation_data['DORC'] != regulation_data['Motif']] # remove self-loop
    regulation_data.drop_duplicates(keep='first', inplace=True) # remove duplicates
    regulation_data['DORC'] = regulation_data['DORC'].str.upper()
    regulation_data['Motif'] = regulation_data['Motif'].str.upper()
    regulation_data = pd.DataFrame(regulation_data.values, columns=['TF', 'Target'])

    # 3\ Filter the data that have corresponding sequence data and expression data
    tf_list = pd.read_csv('./species_specific_data/ProcessedData/HomoSapiens/protein.seq', sep='\t', header=None)[0].str.upper().tolist()
    target_list = pd.read_csv('./species_specific_data/ProcessedData/HomoSapiens/promoter.seq', sep='\t', header=None)[0].str.upper().tolist()
    regulation_data = regulation_data[regulation_data['TF'].isin(tf_list)]
    regulation_data = regulation_data[regulation_data['Target'].isin(target_list)]
    regulation_data = regulation_data[regulation_data['TF'].isin(expression_data.index.tolist())]
    regulation_data = regulation_data[regulation_data['Target'].isin(expression_data.index.tolist())]
    expression_data = expression_data[expression_data.index.isin(regulation_data['TF'].tolist() + regulation_data['Target'].tolist())] # If we use trick, we need to comment this line.
    # expression_data = expression_data[expression_data.index.isin(target_list)] # If we use trick, we need to comment this line.
    TF = regulation_data['TF'].unique()
    GENE = np.unique(np.concatenate([regulation_data['TF'].unique(), regulation_data['Target'].unique()]))

    # 4\ Save data
    expression_data.to_csv(os.path.join(save_path, 'expression.csv'), header=True, index=True)
    regulation_data.to_csv(os.path.join(save_path, 'regulation.csv'), header=True, index=False)

    # 5\ Split train, validation and test set
    regulation_file = os.path.join(save_path, 'regulation.csv')
    expression_file = os.path.join(save_path, 'expression.csv')
    if balance == 'False':
        train_num_pos, train_num_neg, val_num_pos, val_num_neg, test_num_pos, test_num_neg = train_test_val_split_hard(regulation_file, expression_file, save_path, trick=False)
    if balance == 'True':
        train_num_pos, train_num_neg, val_num_pos, val_num_neg, test_num_pos, test_num_neg = train_test_val_split_balance(regulation_file, expression_file, save_path, trick=False)

    logger.info('Start preprocessing...')
    logger.info('Writing data to %s', save_path)
    logger.info('Number of TFs of Network: %d', len(TF))
    logger.info('Number of genes of Network: %d', len(GENE))
    logger.info('Number of edges: %d', regulation_data.shape[0])
    logger.info('Density: %.3f', regulation_data.shape[0] / ((len(TF) * len(GENE)) - len(TF)))
    logger.info('Number of the cells: %d', adata.shape[0])
    logger.info('Train set: ' + str(train_num_pos) + ' positive edges, ' + str(train_num_neg) + ' negative edges, ' + str(train_num_pos + train_num_neg) + ' total edges, ' + str(round(train_num_pos/(train_num_pos + train_num_neg), 4)) + ' positive ratio')
    logger.info('Validation set: ' + str(val_num_pos) + ' positive edges, ' + str(val_num_neg) + ' negative edges, ' + str(val_num_pos + val_num_neg) + ' total edges, ' + str(round(val_num_pos/(val_num_pos + val_num_neg), 4)) + ' positive ratio')
    logger.info('Test set: ' + str(test_num_pos) + ' positive edges, ' + str(test_num_neg) + ' negative edges, ' + str(test_num_pos + test_num_neg) + ' total edges, ' + str(round(test_num_pos/(test_num_pos + test_num_neg), 4)) + ' positive ratio')
    logger.info('End preprocessing...')

    return expression_file, regulation_file


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Cell type specific data preprocess.')
    parser.add_argument('-e', '--exp_file', type = str, default = 'None', help='Path to expression data file.')
    parser.add_argument('-f', '--network_file', type = str, default = 'None', help='Path to network file.')
    parser.add_argument('-b', '--balance', type = str, default = 'False', help='Whether to balance the positive and negative samples.')
    parser.add_argument('-o', '--out_prefix', type = str, default = './', help='Prefix for writing output files.')
    args = parser.parse_args()

    # Set random seed.
    np.random.seed(0)

    # Preprocess data.
    specific_expression_file, specific_regulation_file = preprocess_cell_type_specific_data(args.exp_file, args.network_file, balance=args.balance, save_path=args.out_prefix)
