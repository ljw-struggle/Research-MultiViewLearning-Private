# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import pandas as pd

from tqdm import tqdm
from proteinbert import load_pretrained_model

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.INFO)
cons_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler = logging.FileHandler(filename='species_specific_data/ProcessedData/feature_protein.log', mode='a+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('PreProcess')
logger.addHandler(cons_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def PROTEIN_ONEHOT_FEATURE_EXTRACTION(raw_data_path, feature_save_path):
    prot_one_hot_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
                         'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    data = pd.read_csv(raw_data_path, header=None, sep='\t', low_memory=False)
    name_sequence_dict = dict(zip(data[0], data[1]))

    def prot2onehot(prot_sequence):
        prot_sequence = prot_sequence.upper()
        prot_seq_onehot = np.zeros((1000, 20), dtype=np.float32)
        for i in range(len(prot_sequence)):
            if prot_sequence[i] in prot_one_hot_dict.keys():
                prot_seq_onehot[i, prot_one_hot_dict[prot_sequence[i]]] = 1
        return prot_seq_onehot.astype(np.float32)

    for name, sequence in tqdm(name_sequence_dict.items(), ascii=True, desc='PROTEIN_ONEHOT_FEATURE_EXTRACTION'):
        name_sequence_dict[name] = prot2onehot(sequence[:1000])
    
    np.save(feature_save_path, name_sequence_dict)

    
def PROTEIN_VECTOR_FEATURE_EXTRACTION(raw_data_path, feature_save_path):
    # https://github.com/ehsanasgari/Deep-Proteomics
    prot2vec_dataframe = pd.read_csv('./species_specific_data/RawData/prot2vec.csv', header=None, index_col=0, sep='\t', low_memory=False)
    data = pd.read_csv(raw_data_path, header=None, sep='\t', low_memory=False)
    name_sequence_dict = dict(zip(data[0], data[1]))

    def prot2vec(prot_sequence):
        prot_sequence = prot_sequence.upper()
        prot_sequence_3mer = [prot_sequence[i:i+3] for i in range(len(prot_sequence) - 2)]
        
        prot_sequence_3mer_vec = np.zeros((1000, 100))
        for i in range(len(prot_sequence_3mer)):
            if prot_sequence_3mer[i] in prot2vec_dataframe.index:
                prot_sequence_3mer_vec[i] = prot2vec_dataframe.loc[prot_sequence_3mer[i]]
        
        return prot_sequence_3mer_vec.astype(np.float32)

    for name, sequence in tqdm(name_sequence_dict.items(), ascii=True, desc='PROTEIN_VECTOR_FEATURE_EXTRACTION'):
        name_sequence_dict[name] = prot2vec(sequence[:1000])
    
    np.save(feature_save_path, name_sequence_dict)  


def PROTEIN_BERT_FEATURE_EXTRACTION(raw_data_path, feature_save_path):
    # https://github.com/nadavbra/protein_bert
    data = pd.read_csv(raw_data_path, header=None, sep='\t', low_memory=False)
    name_sequence_dict = dict(zip(data[0], data[1]))

    pretrained_model_generator, input_encoder = load_pretrained_model('./species_specific_data/RawData/', 'epoch_92400_sample_23500000.pkl')
    model = pretrained_model_generator.create_model(1002)

    def prot2bert(sequence):
        temp = input_encoder.encode_X([sequence], len(sequence))
        result = model(temp)
        return result[0][0][1:-1].numpy().astype(np.float32)

    for name, sequence in tqdm(name_sequence_dict.items(), ascii=True, desc='PROTEIN_BERT_FEATURE_EXTRACTION'):
        length = len(sequence)
        if length < 1000:
            sequence = sequence + '-' * (1000 - length)
        elif length > 1000:
            sequence = sequence[0:1000]
        name_sequence_dict[name] = prot2bert(sequence)

    np.save(feature_save_path, name_sequence_dict)


def PROTEIN_ESM2_FEATURE_EXTRACTION(raw_data_path, feature_save_path):
    # https://huggingface.co/facebook/esm2_t33_650M_UR50D
    # https://github.com/facebookresearch/esm
    pass


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    processed_data_dir = './species_specific_data/ProcessedData/'

    # Homo sapiens
    logger.info('HomoSapiens start.')
    HomoSapiens_dir = os.path.join(processed_data_dir, 'HomoSapiens')
    PROTEIN_ONEHOT_FEATURE_EXTRACTION(os.path.join(HomoSapiens_dir, 'protein.seq'), os.path.join(HomoSapiens_dir, 'protein.onehot.npy')) # (1000, 20)
    PROTEIN_VECTOR_FEATURE_EXTRACTION(os.path.join(HomoSapiens_dir, 'protein.seq'), os.path.join(HomoSapiens_dir, 'protein.vector.npy')) # (1000, 100)
    PROTEIN_BERT_FEATURE_EXTRACTION(os.path.join(HomoSapiens_dir, 'protein.seq'), os.path.join(HomoSapiens_dir, 'protein.bert.npy')) # (1000, 26)
    logger.info('HomoSapiens done.')

    # Mus musculus
    logger.info('MusMusculus start.')
    MusMusculus_dir = os.path.join(processed_data_dir, 'MusMusculus')
    PROTEIN_ONEHOT_FEATURE_EXTRACTION(os.path.join(MusMusculus_dir, 'protein.seq'), os.path.join(MusMusculus_dir, 'protein.onehot.npy')) # (1000, 20)
    PROTEIN_VECTOR_FEATURE_EXTRACTION(os.path.join(MusMusculus_dir, 'protein.seq'), os.path.join(MusMusculus_dir, 'protein.vector.npy')) # (1000, 100)
    PROTEIN_BERT_FEATURE_EXTRACTION(os.path.join(MusMusculus_dir, 'protein.seq'), os.path.join(MusMusculus_dir, 'protein.bert.npy')) # (1000, 26)
    logger.info('MusMusculus done.')

