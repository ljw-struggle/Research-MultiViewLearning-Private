# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

def build_save_set(pos_dict, neg_dict, file):
    pos_set = []
    neg_set = []
    [pos_set.append([k, j]) for k in pos_dict.keys() for j in pos_dict[k]]
    [neg_set.append([k, j]) for k in neg_dict.keys() for j in neg_dict[k]]

    set_sample = np.array(pos_set + neg_set)
    set_label = [1 for _ in range(len(pos_set))] + [0 for _ in range(len(neg_set))]

    set_df = pd.DataFrame()
    set_df['TF'] = set_sample[:, 0]
    set_df['Target'] = set_sample[:, 1]
    set_df['Label'] = set_label
    set_df.to_csv(file, index=False)
    return len(pos_set), len(neg_set)
    

def train_test_val_split_hard(edge_file, expression_file, save_path, trick=False):
    # Read input data.
    edge_list = pd.read_csv(edge_file, index_col=None)
    edge_list.rename(columns={'Gene1': 'TF'}, inplace=True)
    edge_list.rename(columns={'Gene2': 'Target'}, inplace=True)
    if not trick:
        gene_list = np.unique(np.concatenate((edge_list['TF'].values, edge_list['Target'].values)))
    else:
        # trick: use expression file to get gene list, so that the negative samples become more.
        expression_file = pd.read_csv(expression_file, index_col=0)
        gene_list = expression_file.index.values
    tf_list = np.unique(edge_list['TF'].values)

    # Initialize positive and negative examples.
    pos_dict = {i: [] for i in tf_list}
    neg_dict = {i: [] for i in tf_list}

    # Compute positive and negative examples.
    for i in pos_dict.keys():
        pos_dict[i].extend(edge_list.loc[edge_list['TF'] == i]['Target'].values)

    for i in neg_dict.keys():
        neg_item = np.setdiff1d(gene_list, np.concatenate((pos_dict[i], [i])))
        neg_dict[i].extend(neg_item)

    # Split positive examples into train, validation, and test sets.
    train_pos, val_pos, test_pos = {}, {}, {}
    for k in pos_dict.keys():
        train_pos[k], val_pos[k], test_pos[k] = [], [], []
        if len(pos_dict[k]) <= 1:
            if np.random.uniform(0, 1) <= 0.5:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) == 2:
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k]) * (3/5))]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k]) * (3/5)):int(len(pos_dict[k]) * (4/5))]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k]) * (4/5)):]
    
    # Split negative examples into train, validation, and test sets.
    train_neg, val_neg, test_neg = {}, {}, {}
    for k in neg_dict.keys():
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(len(neg_dict[k]) * (3/5))]
        val_neg[k] = neg_dict[k][int(len(neg_dict[k]) * (3/5)):int(len(neg_dict[k]) * (4/5))]
        test_neg[k] = neg_dict[k][int(len(neg_dict[k]) * (4/5)):]

    # Build and save train, validation, and test sets.
    train_set_file = os.path.join(save_path, 'train_set.csv')
    val_set_file = os.path.join(save_path, 'val_set.csv')
    test_set_file = os.path.join(save_path, 'test_set.csv')
    train_num_pos, train_num_neg = build_save_set(train_pos, train_neg, train_set_file)
    val_num_pos, val_num_neg = build_save_set(val_pos, val_neg, val_set_file)
    test_num_pos, test_num_neg = build_save_set(test_pos, test_neg, test_set_file)

    return train_num_pos, train_num_neg, val_num_pos, val_num_neg, test_num_pos, test_num_neg


def train_test_val_split_balance(edge_file, expression_file, save_path, trick=False):
    # Read input data.
    edge_list = pd.read_csv(edge_file, index_col=None)
    edge_list.rename(columns={'Gene1': 'TF'}, inplace=True)
    edge_list.rename(columns={'Gene2': 'Target'}, inplace=True)
    if not trick:
        gene_list = np.unique(np.concatenate((edge_list['TF'].values, edge_list['Target'].values)))
    else:
        # trick: use expression file to get gene list, so that the negative samples become more.
        expression_file = pd.read_csv(expression_file, index_col=0)
        gene_list = expression_file.index.values
    tf_list = np.unique(edge_list['TF'].values)

    # Initialize positive and negative examples.
    pos_dict = {i: [] for i in tf_list}
    neg_dict = {i: [] for i in tf_list}

    # Compute positive and negative examples.
    for i in pos_dict.keys():
        pos_dict[i].extend(edge_list.loc[edge_list['TF'] == i]['Target'].values)

    for i in neg_dict.keys():
        neg_item = np.setdiff1d(gene_list, np.concatenate((pos_dict[i], [i])))
        neg_dict[i].extend(neg_item)

    # Split positive examples into train, validation, and test sets.
    train_pos, val_pos, test_pos = {}, {}, {}
    for k in pos_dict.keys():
        train_pos[k], val_pos[k], test_pos[k] = [], [], []
        if len(pos_dict[k]) <= 1:
            if np.random.uniform(0, 1) <= 0.5:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]
        elif len(pos_dict[k]) == 2:
            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]
        else:
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:int(len(pos_dict[k]) * (3/5))]
            val_pos[k] = pos_dict[k][int(len(pos_dict[k]) * (3/5)):int(len(pos_dict[k]) * (4/5))]
            test_pos[k] = pos_dict[k][int(len(pos_dict[k]) * (4/5)):]
    
    # Split negative examples into train, validation, and test sets.
    train_neg, val_neg, test_neg = {}, {}, {}
    for k in neg_dict.keys():
        np.random.shuffle(neg_dict[k])
        train_neg[k] = neg_dict[k][:int(len(neg_dict[k]) * (3/5))]
        val_neg[k] = neg_dict[k][int(len(neg_dict[k]) * (3/5)):int(len(neg_dict[k]) * (4/5))]
        test_neg[k] = neg_dict[k][int(len(neg_dict[k]) * (4/5)):]

    
    # Balance positive and negative examples in train, validation, and test sets. 
    for k in train_neg.keys():
        if len(train_neg[k]) < len(train_pos[k]):
            train_neg[k] = np.random.choice(train_neg[k], size=len(train_pos[k]), replace=True)
        else:
            train_neg[k] = train_neg[k][:len(train_pos[k])]

    for k in val_neg.keys():
        if len(val_neg[k]) < len(val_pos[k]):
            val_neg[k] = np.random.choice(val_neg[k], size=len(val_pos[k]), replace=True)
        else:
            val_neg[k] = val_neg[k][:len(val_pos[k])]


    # Build and save train, validation, and test sets.
    train_set_file = os.path.join(save_path, 'train_set.csv')
    val_set_file = os.path.join(save_path, 'val_set.csv')
    test_set_file = os.path.join(save_path, 'test_set.csv')
    train_num_pos, train_num_neg = build_save_set(train_pos, train_neg, train_set_file)
    val_num_pos, val_num_neg = build_save_set(val_pos, val_neg, val_set_file)
    test_num_pos, test_num_neg = build_save_set(test_pos, test_neg, test_set_file)

    return train_num_pos, train_num_neg, val_num_pos, val_num_neg, test_num_pos, test_num_neg
