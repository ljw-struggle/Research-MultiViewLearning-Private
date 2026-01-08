# -*- coding: utf-8 -*-
import os
import torch
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.INFO)
cons_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler = logging.FileHandler(filename='species_specific_data/ProcessedData/feature_dna.log', mode='a+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('PreProcess')
logger.addHandler(cons_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def DNA_ONEHOT_FEATURE_EXTRACTION(raw_data_path, feature_save_path):
    dna_one_hot_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = pd.read_csv(raw_data_path, header=None, sep='\t', low_memory=False)
    name_sequence_dict = dict(zip(data[0], data[1]))

    def dna2onehot(dna_sequence):
        dna_sequence = dna_sequence.upper()
        dna_seq_onehot = np.zeros((1000, 4), dtype=np.float32)
        for i in range(len(dna_sequence)):
            if dna_sequence[i] in dna_one_hot_dict.keys():
                dna_seq_onehot[i, dna_one_hot_dict[dna_sequence[i]]] = 1
        return dna_seq_onehot.astype(np.float32)
    
    for name, sequence in tqdm(name_sequence_dict.items(), ascii=True, desc='DNA_ONEHOT_FEATURE_EXTRACTION'):
        name_sequence_dict[name] = dna2onehot(sequence)
    
    np.save(feature_save_path, name_sequence_dict)


def DNA_VECTOR_FEATURE_EXTRACTION(raw_data_path, feature_save_path):
    # https://github.com/pnpnpn/dna2vec 
    dna2vec_dataframe = pd.read_csv('./species_specific_data/RawData/dna2vec.csv', header=None, index_col=0, sep=' ', low_memory=False)
    # retain the rows that name is 3mer
    dna2vec_dataframe = dna2vec_dataframe[dna2vec_dataframe.index.str.len() == 3]
    data = pd.read_csv(raw_data_path, header=None, sep='\t', low_memory=False)
    name_sequence_dict = dict(zip(data[0], data[1]))

    def dna2vec(dna_sequence):
        dna_sequence = dna_sequence.upper()
        dna_sequence_3mer = [dna_sequence[i:i+3] for i in range(len(dna_sequence) - 2)]

        dna_sequence_3mer_vec = np.zeros((1000, 100))
        for i in range(len(dna_sequence_3mer)):
            if dna_sequence_3mer[i] in dna2vec_dataframe.index:
                dna_sequence_3mer_vec[i] = dna2vec_dataframe.loc[dna_sequence_3mer[i]]
        
        return dna_sequence_3mer_vec.astype(np.float32)

    for name, sequence in tqdm(name_sequence_dict.items(), ascii=True, desc='DNA_VECTOR_FEATURE_EXTRACTION'):
        name_sequence_dict[name] = dna2vec(sequence)
    
    np.save(feature_save_path, name_sequence_dict)


def DNA_BERT_FEATURE_EXTRACTION(raw_data_path, feature_save_path):
    # https://github.com/jerryji1993/DNABERT
    # https://huggingface.co/zhihan1996
    data = pd.read_csv(raw_data_path, header=None, sep='\t', low_memory=False)
    name_sequence_dict = dict(zip(data[0], data[1]))
    DNA_BERT_tokenizer_3 = AutoTokenizer.from_pretrained("armheb/DNA_bert_3", trust_remote_code=True)
    DNA_BERT_model_3 = AutoModelForMaskedLM.from_pretrained("armheb/DNA_bert_3", trust_remote_code=True).to('cuda')
    # DNA_BERT_tokenizer_4 = AutoTokenizer.from_pretrained("armheb/DNA_bert_4")
    # DNA_BERT_model_4 = AutoModelForMaskedLM.from_pretrained("armheb/DNA_bert_4").to('cuda')
    # DNA_BERT_tokenizer_5 = AutoTokenizer.from_pretrained("armheb/DNA_bert_5")
    # DNA_BERT_model_5 = AutoModelForMaskedLM.from_pretrained("armheb/DNA_bert_5").to('cuda')
    # DNA_BERT_tokenizer_6 = AutoTokenizer.from_pretrained("armheb/DNA_bert_6")
    # DNA_BERT_model_6 = AutoModelForMaskedLM.from_pretrained("armheb/DNA_bert_6").to('cuda')

    def dna2bert(dna_sequence):
        sequence_a = dna_sequence[:502]
        sequence_b = dna_sequence[498:]
        sequence_a_3mer = [sequence_a[i:i+3] for i in range(len(sequence_a) - 2)]
        sequence_b_3mer = [sequence_b[i:i+3] for i in range(len(sequence_b) - 2)]

        inputs = [' '.join(sequence_a_3mer), ' '.join(sequence_b_3mer)]
        inputs = DNA_BERT_tokenizer_3(inputs, return_tensors='pt')
        outputs = DNA_BERT_model_3(**inputs.to('cuda')).logits.detach().cpu().numpy()

        return np.concatenate((outputs[0][1:-1], outputs[1][1:-1]), axis=0).astype(np.float32)

    for name, sequence in tqdm(name_sequence_dict.items(), ascii=True, desc='DNA_BERT_FEATURE_EXTRACTION'):
        name_sequence_dict[name] = dna2bert(sequence)

    np.save(feature_save_path, name_sequence_dict)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    processed_data_dir = './species_specific_data/ProcessedData/'

    # Homo sapiens
    logger.info('HomoSapiens start.')
    HomoSapiens_dir = os.path.join(processed_data_dir, 'HomoSapiens')
    DNA_ONEHOT_FEATURE_EXTRACTION(os.path.join(HomoSapiens_dir, 'promoter.seq'), os.path.join(HomoSapiens_dir, 'promoter.onehot.npy')) # (1000, 4)
    DNA_VECTOR_FEATURE_EXTRACTION(os.path.join(HomoSapiens_dir, 'promoter.seq'), os.path.join(HomoSapiens_dir, 'promoter.vector.npy')) # (1000, 100)
    DNA_BERT_FEATURE_EXTRACTION(os.path.join(HomoSapiens_dir, 'promoter.seq'), os.path.join(HomoSapiens_dir, 'promoter.bert.npy')) # (1000, 69)
    logger.info('HomoSapiens done.')

    # Mus musculus
    logger.info('MusMusculus start.')
    MusMusculus_dir = os.path.join(processed_data_dir, 'MusMusculus')
    DNA_ONEHOT_FEATURE_EXTRACTION(os.path.join(MusMusculus_dir, 'promoter.seq'), os.path.join(MusMusculus_dir, 'promoter.onehot.npy')) # (1000, 4)
    DNA_VECTOR_FEATURE_EXTRACTION(os.path.join(MusMusculus_dir, 'promoter.seq'), os.path.join(MusMusculus_dir, 'promoter.vector.npy')) # (1000, 100)
    DNA_BERT_FEATURE_EXTRACTION(os.path.join(MusMusculus_dir, 'promoter.seq'), os.path.join(MusMusculus_dir, 'promoter.bert.npy')) # (1000, 69)
    logger.info('MusMusculus done.')
