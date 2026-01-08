# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from motif_utils import load_all_meme, load_all_pwm, gene_to_one_hot


sys.path.append('../code/')
from loader.loader_deepmix import Cell_Type_Specific_Data, Cell_Type_Specific_Loader
from loader import loader_deepmix as module_loader
from model import model as module_model
from model import criterion as module_criterion
from model import metric as module_metric
from utils import read_json, write_json, calculate_auroc, calculate_aupr, calculate_ep, plot_roc_curve, plot_pr_curve
from parse_config import ConfigParser


def get_tissue_specific_result(tissue='Human-Blood-Transfer', data_dir='PBMC1'):
    config_file = '../code/config/DeepMixModel/tissue_specific_data/{}/config_mmdynamics.json'.format(tissue)
    config = read_json(config_file)
    configer = ConfigParser(config, modification={'loader;args;train_cell_number': 2000, 'loader;args;data': tissue + '/' + data_dir, 'model;args;graph_model_args;input_dim': 2000})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_file = '../result/DeepMixModel/tissue_specific_data/{}/mmdynamics/ORIGIN/model/model_best.pth'.format(tissue)
    model = configer.init_obj('model', module_model)
    model.load_state_dict(torch.load(model_file)['model_state_dict'])
    model.to(device)

    train_data_loader = configer.init_obj('loader', module_loader, logger=None, mode='train', shuffle=False)
    valid_data_loader = configer.init_obj('loader', module_loader, logger=None, mode='valid')
    test_data_loader = configer.init_obj('loader', module_loader, logger=None, mode='test')
    gene_node_expression = train_data_loader.get_node_expression().to(device)
    prior_graph_topology = train_data_loader.get_prior_graph_topology().to(device)

    data_type_list = []
    TF_list = []
    GENE_list = []
    output_list = []
    label_list = []
    classifier_expression_list = []
    confidence_expression_list = []
    classifier_sequence_list = []
    confidence_sequence_list = []

    for batch_idx, data in enumerate(train_data_loader):
        TF = data['input']['TF']
        GENE = data['input']['GENE']

        graph_input = {'NODE': gene_node_expression,
                    'ENCODE': prior_graph_topology,
                    'DECODE': data['input']['EDGE'].transpose(0, 1).to(device)}
        branch_input = {'TF': data['input']['TF_feature'].to(device),
                        'TF_len': data['input']['TF_len'].to(device),
                        'GENE': data['input']['GENE_feature'].to(device)}

        target = data['label'].to(device)
        weight = data['weight'].to(device)
        batch_size = branch_input['TF'].size(0)


        output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = model(graph_input, branch_input)
        output = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
        classifier_expression = torch.softmax(classifier_expression, dim=1)[:, 1].detach().cpu().numpy()
        confidence_expression = torch.sigmoid(confidence_expression)[:, 0].detach().cpu().numpy()
        classifier_sequence = torch.softmax(classifier_sequence, dim=1)[:, 1].detach().cpu().numpy()
        confidence_sequence = torch.sigmoid(confidence_sequence)[:, 0].detach().cpu().numpy()

        data_type_list.append(['train' for _ in range(batch_size)])
        TF_list.append(TF)
        GENE_list.append(GENE)
        output_list.append(output)
        classifier_expression_list.append(classifier_expression)
        confidence_expression_list.append(confidence_expression)
        classifier_sequence_list.append(classifier_sequence)
        confidence_sequence_list.append(confidence_sequence)
        label_list.append(target.detach().cpu().numpy())


    for batch_idx, data in enumerate(valid_data_loader):
        TF = data['input']['TF']
        GENE = data['input']['GENE']

        graph_input = {'NODE': gene_node_expression,
                    'ENCODE': prior_graph_topology,
                    'DECODE': data['input']['EDGE'].transpose(0, 1).to(device)}
        branch_input = {'TF': data['input']['TF_feature'].to(device),
                        'TF_len': data['input']['TF_len'].to(device),
                        'GENE': data['input']['GENE_feature'].to(device)}

        target = data['label'].to(device)
        weight = data['weight'].to(device)
        batch_size = branch_input['TF'].size(0)


        output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = model(graph_input, branch_input)
        output = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
        classifier_expression = torch.softmax(classifier_expression, dim=1)[:, 1].detach().cpu().numpy()
        confidence_expression = torch.sigmoid(confidence_expression)[:, 0].detach().cpu().numpy()
        classifier_sequence = torch.softmax(classifier_sequence, dim=1)[:, 1].detach().cpu().numpy()
        confidence_sequence = torch.sigmoid(confidence_sequence)[:, 0].detach().cpu().numpy()

        data_type_list.append(['valid' for _ in range(batch_size)])
        TF_list.append(TF)
        GENE_list.append(GENE)
        output_list.append(output)
        classifier_expression_list.append(classifier_expression)
        confidence_expression_list.append(confidence_expression)
        classifier_sequence_list.append(classifier_sequence)
        confidence_sequence_list.append(confidence_sequence)
        label_list.append(target.detach().cpu().numpy())


    for batch_idx, data in enumerate(test_data_loader):
        TF = data['input']['TF']
        GENE = data['input']['GENE']

        graph_input = {'NODE': gene_node_expression,
                    'ENCODE': prior_graph_topology,
                    'DECODE': data['input']['EDGE'].transpose(0, 1).to(device)}
        branch_input = {'TF': data['input']['TF_feature'].to(device),
                        'TF_len': data['input']['TF_len'].to(device),
                        'GENE': data['input']['GENE_feature'].to(device)}

        target = data['label'].to(device)
        weight = data['weight'].to(device)
        batch_size = branch_input['TF'].size(0)


        output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = model(graph_input, branch_input)
        output = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
        classifier_expression = torch.softmax(classifier_expression, dim=1)[:, 1].detach().cpu().numpy()
        confidence_expression = torch.sigmoid(confidence_expression)[:, 0].detach().cpu().numpy()
        classifier_sequence = torch.softmax(classifier_sequence, dim=1)[:, 1].detach().cpu().numpy()
        confidence_sequence = torch.sigmoid(confidence_sequence)[:, 0].detach().cpu().numpy()

        data_type_list.append(['test' for _ in range(batch_size)])
        TF_list.append(TF)
        GENE_list.append(GENE)
        output_list.append(output)
        classifier_expression_list.append(classifier_expression)
        confidence_expression_list.append(confidence_expression)
        classifier_sequence_list.append(classifier_sequence)
        confidence_sequence_list.append(confidence_sequence)
        label_list.append(target.detach().cpu().numpy())


    data_type_list = np.concatenate(data_type_list)
    TF_list = np.concatenate(TF_list)
    GENE_list = np.concatenate(GENE_list)
    output_list = np.concatenate(output_list).round(4)
    classifier_expression_list = np.concatenate(classifier_expression_list).round(4)
    confidence_expression_list = np.concatenate(confidence_expression_list).round(4)
    classifier_sequence_list = np.concatenate(classifier_sequence_list).round(4)
    confidence_sequence_list = np.concatenate(confidence_sequence_list).round(4)
    label_list = np.concatenate(label_list)


    # save as csv
    df = pd.DataFrame({'data_type': data_type_list, 'TF': TF_list, 'GENE': GENE_list, 'output': output_list, 
                    'classifier_expression': classifier_expression_list, 'confidence_expression': confidence_expression_list,
                    'classifier_sequence': classifier_sequence_list, 'confidence_sequence': confidence_sequence_list, 'label': label_list})
    df.to_csv('../result/DeepMixModel/tissue_specific_data/{}/output_{}.csv'.format(tissue, data_dir), index=False)

    fpr_list, tpr_list, auroc = calculate_auroc(output_list, label_list)
    precision_list, recall_list, aupr = calculate_aupr(output_list, label_list)
    aupr_norm = aupr / np.mean(label_list)
    plot_roc_curve(fpr_list, tpr_list, os.path.join(configer.result_dir, 'roc_curve_{}.png'.format(data_dir)))
    plot_pr_curve(precision_list, recall_list, os.path.join(configer.result_dir, 'pr_curve_{}.png'.format(data_dir)))
    ep, epr = calculate_ep(output_list, label_list)
    log = {'auroc': auroc, 'aupr': aupr, 'aupr_norm': aupr_norm, 'ep': ep, 'epr': epr}

    with open('../result/DeepMixModel/tissue_specific_data/{}/log_{}.txt'.format(tissue, data_dir), 'w') as f:
        f.write(str(log))


# get_tissue_specific_result(tissue='Human-Blood-Transfer', data_dir='PBMC1')
# get_tissue_specific_result(tissue='Human-Blood-Transfer', data_dir='PBMC2')
# get_tissue_specific_result(tissue='Human-Blood-Transfer', data_dir='PBMC3')
# get_tissue_specific_result(tissue='Mouse-Brain-Cortex-Transfer', data_dir='Adult')
get_tissue_specific_result(tissue='Mouse-Brain-Cortex-Transfer', data_dir='Fetal')


# $ nohup python 6.get_tissue_specific_result.py > /dev/null 2>&1 &
# $ ps -ef | grep 6.get_tissue_specific_result.py | grep -v grep | awk '{print $2}' | xargs kill -9
