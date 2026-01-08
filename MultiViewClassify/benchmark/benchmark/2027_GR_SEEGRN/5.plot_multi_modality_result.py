# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from motif_utils import load_all_meme, load_all_pwm, gene_to_one_hot

sys.path.append('../code/')

from loader.loader_deepmix import Tissue_Specific_Data, Tissue_Specific_Loader
from model import model as module_model
from model import criterion as module_criterion
from model import metric as module_metric
from utils import read_json
from parse_config import ConfigParser


config_file = '../code/config/DeepMixModel/tissue_specific_data/Adult-Liver/config_mmdynamics.json'
config = read_json(config_file)
configer = ConfigParser(config, modification={})

model_file = '../result/DeepMixModel/tissue_specific_data/Adult-Liver/mmdynamics/ORIGIN/model/model_best.pth'
model = configer.init_obj('model', module_model)
model.load_state_dict(torch.load(model_file)['model_state_dict'])

criterion = configer.init_obj('criterion', module_criterion)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Tissue_Specific_Data(logger=None, mode='train', feature='bert', data='Adult-Liver', train_data_ratio=1.0, train_data_size=1.0)

# Create a new edge list
TF = np.unique(dataset.edge_list[:, 0])
edge_list = []
for i in range(len(TF)):
    for j in range(len(TF)):
        edge_list.append([TF[i], TF[j]])
dataset.edge_list = np.array(edge_list)

regulation = pd.read_csv('../data/tissue_specific_data/ProcessedData/Adult-Liver/regulation.csv', header=0)
regulation = regulation.values
label_list = []
for edge in dataset.edge_list:
    index = np.logical_and(regulation[:, 0] == edge[0], regulation[:, 1] == edge[1])
    label_list.append(np.sum(index))
dataset.label_list = np.array(label_list)
dataset.weight_list = np.ones(len(dataset.label_list))

print("TF num: %s" % len(TF))
print("Dataset num: %s" % len(dataset))
print("Positive num: %s" % np.sum(dataset.label_list))


gene_sc_expression = dataset.gene_sc_expression.to(device)
prior_graph_topology = dataset.prior_graph_topology.to(device)


output_list = []
classifier_expression_list = []
confidence_expression_list = []
classifier_sequence_list = []
confidence_sequence_list = []
label_list = []
for index in range(len(TF)):
    # print('TF: %s' % TF[index])
    batch_data = [dataset[i + index * len(TF)] for i in range(len(TF))]

    TF_name = [data['input']['TF'] for data in batch_data]
    GENE_name = [data['input']['GENE'] for data in batch_data]
    TF_sequence = [data['input']['TF_seq'] for data in batch_data]
    GENE_sequence = [data['input']['GENE_seq'] for data in batch_data]
    LABEL = [data['label'] for data in batch_data]

    graph_input = {'NODE': gene_sc_expression,
                   'ENCODE': prior_graph_topology,
                   'DECODE': torch.Tensor(np.array([data['input']['EDGE'] for data in batch_data])).transpose(0, 1).to(device).to(torch.long)}
    branch_input = {'TF': torch.Tensor(np.array([data['input']['TF_feature'] for data in batch_data])).to(device),
                    'GENE': torch.Tensor(np.array([data['input']['GENE_feature'] for data in batch_data])).to(device),
                    'TF_len': torch.Tensor(np.array([data['input']['TF_len'] for data in batch_data])).to(device).to(torch.long)}

    model.to(device)

    output, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence = model(graph_input, branch_input)
    output = torch.softmax(output, dim=1)[:, 1].detach().cpu().numpy()
    classifier_expression = torch.softmax(classifier_expression, dim=1)[:, 1].detach().cpu().numpy()
    confidence_expression = torch.sigmoid(confidence_expression)[:, 0].detach().cpu().numpy()
    classifier_sequence = torch.softmax(classifier_sequence, dim=1)[:, 1].detach().cpu().numpy()
    confidence_sequence = torch.sigmoid(confidence_sequence)[:, 0].detach().cpu().numpy()

    output_list.append(output)
    classifier_expression_list.append(classifier_expression)
    confidence_expression_list.append(confidence_expression)
    classifier_sequence_list.append(classifier_sequence)
    confidence_sequence_list.append(confidence_sequence)
    label_list.append(LABEL)

output_list = np.array(output_list)
classifier_expression_list = np.array(classifier_expression_list)
confidence_expression_list = np.array(confidence_expression_list)
classifier_sequence_list = np.array(classifier_sequence_list)
confidence_sequence_list = np.array(confidence_sequence_list)
label_list = np.array(label_list)

# generate the diagonal matrix
diagonal_matrix = np.eye(len(TF))
mask_matrix = np.ones((len(TF), len(TF))) - diagonal_matrix

output_list = output_list * mask_matrix
classifier_expression_list = classifier_expression_list * mask_matrix
confidence_expression_list = confidence_expression_list * mask_matrix
classifier_sequence_list = classifier_sequence_list * mask_matrix
confidence_sequence_list = confidence_sequence_list * mask_matrix
label_list = label_list * mask_matrix

# randomly select 30 rows and 30 columns
index = np.arange(len(TF))
np.random.seed(0)
np.random.shuffle(index)
index = index[:15]
output_list = output_list[index][:, index]
classifier_expression_list = classifier_expression_list[index][:, index]
confidence_expression_list = confidence_expression_list[index][:, index]
classifier_sequence_list = classifier_sequence_list[index][:, index]
confidence_sequence_list = confidence_sequence_list[index][:, index]
label_list = label_list[index][:, index]

print(label_list)


# plot the expression pattern
# plot three figures: output_list, classifier_expression_list, classifier_sequence_list
fig, ax = plt.subplots(1, 3, figsize=(20, 7))
ax[0].matshow(label_list, cmap=plt.cm.Blues)
ax[0].set_title('Label', fontsize=20)
ax[0].set_xticks(np.arange(len(TF[index])))
ax[0].set_xticklabels(TF[index], rotation=45, ha='left')
ax[0].set_yticks(np.arange(len(TF[index])))
ax[0].set_yticklabels(TF[index])
ax[1].matshow(classifier_expression_list, cmap=plt.cm.Blues)
ax[1].set_title('Expression', fontsize=20)
ax[1].set_xticks(np.arange(len(TF[index])))
ax[1].set_xticklabels(TF[index], rotation=45, ha='left')
ax[1].set_yticks(np.arange(len(TF[index])))
ax[1].set_yticklabels(TF[index])
ax[2].matshow(classifier_sequence_list, cmap=plt.cm.Blues)
ax[2].set_title('Sequence', fontsize=20)
ax[2].set_xticks(np.arange(len(TF[index])))
ax[2].set_xticklabels(TF[index], rotation=45, ha='left')
ax[2].set_yticks(np.arange(len(TF[index])))
ax[2].set_yticklabels(TF[index])

# plot the colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
fig.colorbar(ax[0].imshow(label_list, cmap=plt.cm.Blues), cax=cbar_ax)
plt.savefig('./_figure/multi_modality_result/adult-liver_1.png', dpi=300)


# $ mkdir -p ./_figure/multi_modality_result/
# $ nohup python 5.plot_multi_modality_result.py > /dev/null 2>&1 &
# $ ps -ef | grep 5.plot_multi_modality_result.py | grep -v grep | awk '{print $2}' | xargs kill -9
