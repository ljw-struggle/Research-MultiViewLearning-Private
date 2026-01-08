import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from motif_utils import load_all_meme, load_all_pwm, gene_to_one_hot

sys.path.append('../code/')

from loader.loader_sequence import HomoSapiens, MusMusculus
from model import model as module_model
from model import criterion as module_criterion
from model import metric as module_metric
from utils import read_json
from parse_config import ConfigParser


config_file = '../code/config/SequenceModel_PreTrain/species_specific_data/HomoSapiens/config_bert_LSTM_3_true_BinaryCrossEntropy.json'
config = read_json(config_file)
configer = ConfigParser(config, modification={})

model_file = '../result/SequenceModel_PreTrain/species_specific_data/HomoSapiens/bert_LSTM_3_true_BinaryCrossEntropy/ORIGIN/model/model_best.pth'
model = configer.init_obj('model', module_model)
model.load_state_dict(torch.load(model_file)['model_state_dict'])

criterion = configer.init_obj('criterion', module_criterion)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = HomoSapiens(logger=None, mode='train', feature='bert')

# meme_dict = load_all_meme()
pwm_dict = load_all_pwm()
pwm_list = list(pwm_dict.keys())

# filter the dataset that has corresponding TF and GENE
filter_index_1 = np.isin(dataset.edge_list[:, 0], pwm_list)
filter_index_2 = np.isin(dataset.edge_list[:, 1], pwm_list)
filter_index = np.logical_and(filter_index_1, filter_index_2)
print('Dataset num by filter: %s' % np.sum(filter_index))
dataset.edge_list = dataset.edge_list[filter_index]
dataset.label_list = dataset.label_list[filter_index]

count_TF = {}
for data in dataset:
    TF_name = data['input']['TF']
    GENE_name = data['input']['GENE']
    TF_sequence = data['input']['TF_seq']
    GENE_sequence = data['input']['GENE_seq']
    LABEL = data['label']

    if TF_name in pwm_dict.keys():
        input = {'TF': torch.Tensor([data['input']['TF_feature']]).to(device),
                'GENE': torch.Tensor([data['input']['GENE_feature']]).to(device).requires_grad_(),
                'TF_len': torch.Tensor([data['input']['TF_len']]).to(device).to(torch.long)}
        model.to(device)

        output = model(input)
        prediction = output[0][0].detach().cpu().numpy()

        # get the attention of the gene input
        score_attention = output[1][0].detach().cpu().numpy()
        score_attention = np.sum(score_attention, axis=0)
        score_attention = (score_attention - np.min(score_attention)) / (np.max(score_attention) - np.min(score_attention))

        # get the gradient of the gene input and gradient of the attention score
        loss = criterion(output[0], torch.Tensor([LABEL]).to(device).to(torch.float32))
        loss.backward()
        score_gradient = input['GENE'].grad.detach().cpu().numpy() # shape: (1, 1000, 4)
        score_gradient = np.sum(score_gradient[0], axis=1) # (1000,)
        score_gradient = np.abs(score_gradient)
        score_gradient = (score_gradient - np.min(score_gradient)) / (np.max(score_gradient) - np.min(score_gradient))

        # get the motif score
        GENE_sequence_one_hot = gene_to_one_hot(GENE_sequence)
        motif = pwm_dict[TF_name]
        score_motif = []
        for i in range(len(GENE_sequence_one_hot)):
            fragment = GENE_sequence_one_hot[i:i + len(motif)]
            if len(fragment) < len(motif):
                fragment = np.concatenate((fragment, np.zeros((len(motif) - len(fragment), 4))))
            score = np.sum(fragment * motif)
            score_motif.append(score)
        score_motif = np.array(score_motif)
        score_motif[score_motif < 0] = 0
        score_motif = (score_motif - np.min(score_motif)) / (np.max(score_motif) - np.min(score_motif))

        # count_TF[TF_name] = count_TF.get(TF_name, 0) + 1
        # if count_TF[TF_name] > 100:
        #     continue

        # plot two figures in one figure
        fig, axs = plt.subplots(3, 1, figsize=(20, 5))
        axs[0].plot(range(len(score_motif)), score_motif)
        axs[0].set_xlim([0, len(score_motif)])
        axs[0].set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        axs[0].set_ylabel('Motif Score')
        axs[1].plot(range(len(score_attention)), score_attention)
        axs[1].set_xlim([0, len(score_attention)])
        axs[1].set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        axs[1].set_ylabel('Attention Score')
        axs[2].plot(range(len(score_gradient)), score_gradient)
        axs[2].set_xlim([0, len(score_gradient)])
        axs[2].set_xticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        axs[2].set_ylabel('Gradient Score')

        # plot the title on the top of the figure
        plt.suptitle('TF: %s, GENE: %s, Label: %s, Prediction: %s' % (TF_name, GENE_name, LABEL, np.round(prediction, 2)))

        # save the figure
        plt.tight_layout()
        plt.savefig('./_figure/motif_score/%s_%s.png' % (TF_name, GENE_name))
        plt.close()


# $ mkdir -p ./_figure/motif_score/
# $ nohup python 4.plot_motif_score.py > /dev/null 2>&1 &
# $ ps -ef | grep 4.plot_motif_score.py | grep -v grep | awk '{print $2}' | xargs kill -9
