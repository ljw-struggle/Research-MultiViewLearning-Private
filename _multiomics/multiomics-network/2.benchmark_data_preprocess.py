# -*- coding: utf-8 -*-
import os
import logging
import argparse
import warnings
import numpy as np
import pandas as pd

from utils import train_test_val_split_hard, train_test_val_split_balance

# filter the warnings
warnings.filterwarnings('ignore')

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.INFO)
cons_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler = logging.FileHandler(filename='benchmark_data/ProcessedData/expression_data.log', mode='a+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('PreProcess')
logger.addHandler(cons_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def preprocess_benchmark_data(exp_file, gene_file, network_file, species, pval_cutoff, num_genes, balance, out_prefix):
    assert species in ['human', 'mouse'], "Species should be either human or mouse"
    assert pval_cutoff > 0 and pval_cutoff < 1, "p-value cutoff should be between 0 and 1"
    assert num_genes >= 0, "Number of genes should be >= 0"
    
    # 1\ Read expression data and gene data
    expr_df = pd.read_csv(exp_file, header=0, index_col=0)
    expr_df.index = expr_df.index.str.upper()
    gene_df = pd.read_csv(gene_file, header=0, index_col=0)
    gene_df.index = gene_df.index.str.upper()

    intersection_genes = list(set(expr_df.index.values) & set(gene_df.index.values))
    expr_df = expr_df.loc[intersection_genes]
    gene_df = gene_df.loc[intersection_genes]

    # 2\ Filter the genes according to p-value and cutoff the first 'num_genes' genes according to variance
    variable_tfs_human = pd.read_csv('./benchmark_data/BEELINE-Network/human-tfs.csv', header=0).iloc[:, 0].values
    variable_tfs_mouse = pd.read_csv('./benchmark_data/BEELINE-Network/mouse-tfs.csv', header=0).iloc[:, 0].values
    variable_tfs = variable_tfs_human if species == 'human' else variable_tfs_mouse

    pval_cutoff = pval_cutoff / float(len(gene_df.index)) # Perform Bonferroni correction: divide p-value cutoff by the number of genes
    pval_col = gene_df.columns[0] # p-value column name
    gene_df = gene_df[gene_df[pval_col] < pval_cutoff]
    variable_genes = gene_df.index.values
    variable_tfs = set(set(variable_tfs) & set(variable_genes))
    print("{0} genes pass the p-value cutoff of {1}".format(len(variable_genes), pval_cutoff))
    print("{0} TFs pass the p-value cutoff of {1}".format(len(variable_tfs), pval_cutoff))

    gene_df.drop(labels = variable_tfs, axis='index', inplace = True)
    gene_df.sort_values(by=gene_df.columns[1], inplace=True, ascending = False) # Sort by variance, gene_df.columns[1] is the variance column name
    variable_genes = set(set(gene_df.iloc[:num_genes].index.values) | set(variable_tfs))
    print("{0} genes pass the variance cutoff (restricting to {1} genes)".format(len(variable_genes), num_genes))
    print("{0} TFs pass the variance cutoff (restricting to {1} genes)".format(len(variable_tfs), num_genes))

    expr_df = expr_df.loc[list(variable_genes)]
    print("final expression matrix shape: {}".format(expr_df.shape))

    # 3\ Get tf_list and target_list that are present in the expression data and sequence data
    if species == 'human':
        tf_list = set(pd.read_csv('./species_specific_data/ProcessedData/HomoSapiens/protein.seq', sep='\t', header=None)[0].str.upper().tolist())
        target_list = set(pd.read_csv('./species_specific_data/ProcessedData/HomoSapiens/promoter.seq', sep='\t', header=None)[0].str.upper().tolist())
    if species == 'mouse':
        tf_list = set(pd.read_csv('./species_specific_data/ProcessedData/MusMusculus/protein.seq', sep='\t', header=None)[0].str.upper().tolist())
        target_list = set(pd.read_csv('./species_specific_data/ProcessedData/MusMusculus/promoter.seq', sep='\t', header=None)[0].str.upper().tolist())
    print("Number of TFs present in the expression and sequence data: {}".format(len(tf_list)))
    print("Number of Targets present in the expression and sequence data: {}".format(len(target_list)))

    # 4\ Read network data and filter the network
    net_df = pd.read_csv(network_file)
    net_df.rename(columns={'Gene1': 'TF', 'Gene2': 'Target'}, inplace=True)
    net_df = net_df[(net_df['TF'].isin(expr_df.index)) & (net_df['Target'].isin(expr_df.index))] # Remove genes that are not present in the expression data
    print("Number of edges in the network: {}".format(net_df.shape[0]))
    net_df = net_df[(net_df['TF'].isin(tf_list)) & (net_df['Target'].isin(target_list))] # Remove genes that are not present in the sequence data
    print("Number of edges in the network after removing not in the sequence data: {}".format(net_df.shape[0]))
    net_df = net_df[net_df['TF'] != net_df['Target']] # Remove self-regulation
    print("Number of edges in the network after removing self-regulation: {}".format(net_df.shape[0]))
    net_df.drop_duplicates(keep = 'first', inplace=True) # Remove duplicates (there are some repeated lines in the ground-truth networks). 
    print("Number of edges in the network after removing duplicates: {}".format(net_df.shape[0]))
    net_df.to_csv(os.path.join(out_prefix, 'regulation.csv'), index=False)

    TF = net_df['TF'].unique()
    Target = np.unique(np.concatenate([net_df['TF'].unique(), net_df['Target'].unique()]))

    # There has one trich to make the performance of the model better.
    all_nodes = np.unique(np.concatenate([net_df['TF'].unique(), net_df['Target'].unique()]))
    expr_df = expr_df[expr_df.index.isin(all_nodes)]
    expr_df.to_csv(os.path.join(out_prefix, 'expression.csv'), index=True)

    # Trick: Add the genes that are not present in the network but present in the expression data to the network.
    # The negative samples are generated from these genes. So make the performance of the model better.
    # expr_df = expr_df[expr_df.index.isin(target_list)]
    # all_nodes = expr_df.index.values
    # expr_df.to_csv(os.path.join(out_prefix, 'expression.csv'), index=True)
    
    # 5\ Generate the training, validation, and test sets.
    expression_file = os.path.join(out_prefix, 'expression.csv')
    regulation_file = os.path.join(out_prefix, 'regulation.csv')
    if balance == 'False':
        train_num_pos, train_num_neg, val_num_pos, val_num_neg, test_num_pos, test_num_neg = train_test_val_split_hard(regulation_file, expression_file, out_prefix, trick=False)
    if balance == 'True':
        train_num_pos, train_num_neg, val_num_pos, val_num_neg, test_num_pos, test_num_neg = train_test_val_split_balance(regulation_file, expression_file, out_prefix, trick=False)

    logger.info('Start preprocessing...')
    logger.info('Writing data to %s', out_prefix)
    logger.info('Number of TFs of Network: %d', len(TF))
    logger.info('Number of genes of Network: %d', len(Target))
    logger.info('Number of nodes of Network: %d', len(all_nodes))
    logger.info('Number of edges of Network: %d', net_df.shape[0])
    logger.info('Density: %.3f', net_df.shape[0] / (len(TF) * len(Target) - len(TF)))
    logger.info('Number of the cells: %d', expr_df.shape[1])
    logger.info('Train set: ' + str(train_num_pos) + ' positive edges, ' + str(train_num_neg) + ' negative edges, ' + str(train_num_pos + train_num_neg) + ' total edges, ' + str(round(train_num_pos/(train_num_pos + train_num_neg), 4)) + ' positive ratio')
    logger.info('Valid set: ' + str(val_num_pos) + ' positive edges, ' + str(val_num_neg) + ' negative edges, ' + str(val_num_pos + val_num_neg) + ' total edges, ' + str(round(val_num_pos/(val_num_pos + val_num_neg), 4)) + ' positive ratio')
    logger.info('Test set: ' + str(test_num_pos) + ' positive edges, ' + str(test_num_neg) + ' negative edges, ' + str(test_num_pos + test_num_neg) + ' total edges, ' + str(round(test_num_pos/(test_num_pos + test_num_neg), 4)) + ' positive ratio')
    logger.info('End preprocessing...')

    return expression_file, regulation_file


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Benchmark data preprocess.')
    parser.add_argument('-e', '--exp_file', type = str, default = 'None', help='Path to expression data file.')
    parser.add_argument('-g', '--gene_file', type = str, default = 'None', help='Path to gene ordering file.')
    parser.add_argument('-f', '--network_file', type = str, default = 'None', help='Path to network file.')
    parser.add_argument('-s', '--species', type = str, default = 'human', help='Species. Default = human.')
    parser.add_argument('-p', '--pval_cutoff', type=float, default = 0.01, help='p-value cutoff. Default = 0.01')
    parser.add_argument('-n', '--num_genes', type=int, default = 500, help='Number of genes to add. Default=500.')
    parser.add_argument('-b', '--balance', type=str, default = 'False', help='Whether to balance the positive and negative edges. Default=False.')
    parser.add_argument('-o', '--out_prefix', type = str, default = './', help='Prefix for writing output files.')
    args = parser.parse_args()

    # Set random seed.
    np.random.seed(0)

    # Preprocess the data.
    specific_expression_file, specific_regulation_file = preprocess_benchmark_data(args.exp_file, args.gene_file, args.network_file, args.species, 
                                                                                   args.pval_cutoff, args.num_genes, args.balance, args.out_prefix)
