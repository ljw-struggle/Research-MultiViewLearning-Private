# -*- coding: utf-8 -*-
import os
import re
import logging
import pandas as pd
import numpy as np

from Bio.Seq import Seq

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.INFO)
cons_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
file_handler = logging.FileHandler(filename='species_specific_data/ProcessedData/data_process.log', mode='a+')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('PreProcess')
logger.addHandler(cons_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)


def preprocess_sequences(raw_gbff_file, raw_regulation_file, promoter_seq_file, protein_seq_file):
    regulation = pd.read_csv(raw_regulation_file, header=0, sep='\t', low_memory=False)
    TF_list = regulation['Name.TF'].unique()
    Target_list = np.unique(np.concatenate([regulation['Name.Target'].unique(), regulation['Name.TF'].unique()]))

    promoter_seq_data = {}
    protein_seq_data = {}

    with open(raw_gbff_file, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line:
                break

            gene_info_list = []; 
            if line.startswith('LOCUS'):
                temp_gene_name = None
                temp_gene_pos_a = None; temp_gene_pos_b = None; temp_complement_tag = None; 
                temp_protein_seq = None
                while True:
                    line = f.readline().strip()
                    
                    if line.startswith('gene            '):
                        if temp_gene_name and temp_gene_pos_a and temp_gene_pos_b and temp_complement_tag and temp_protein_seq:
                            gene_info_list.append([temp_gene_name, temp_gene_pos_a, temp_gene_pos_b, temp_complement_tag, temp_protein_seq])
                            temp_gene_name = None
                            temp_gene_pos_a = None; temp_gene_pos_b = None; temp_complement_tag = None; 
                            temp_protein_seq = None
                        
                        temp_gene_pos_a, temp_gene_pos_b = re.findall(r'\d+', line)[:2]
                        if 'complement' in line:
                            temp_complement_tag = 'complement'
                        else:
                            temp_complement_tag = 'normal'

                    if line.startswith('/gene="'):
                        temp_gene_name = line.split('"')[1]

                    if line.startswith('/translation="'):
                        if not temp_protein_seq:
                            temp_protein_seq = line.split('"')[1]
                            if not line.endswith('"'):
                                while True:
                                    line = f.readline().strip()
                                    if line.endswith('"'):
                                        temp_protein_seq += line.split()[0].strip('"')
                                        break

                                    temp_protein_seq += line.split()[0]

                    if line.startswith('ORIGIN'):
                        if temp_gene_name and temp_gene_pos_a and temp_gene_pos_b and temp_complement_tag and temp_protein_seq:
                            gene_info_list.append([temp_gene_name, temp_gene_pos_a, temp_gene_pos_b, temp_complement_tag, temp_protein_seq])
                            temp_gene_name = None
                            temp_gene_pos_a = None; temp_gene_pos_b = None; temp_complement_tag = None; 
                            temp_protein_seq = None

                        seq = ''
                        while True:
                            line = f.readline().strip()
                            if line.startswith('//'):
                                break
                            seq += ''.join(line.split()[1:])

                    if line.startswith('//'):
                        break
            
            for gene_name, gene_pos_a, gene_pos_b, complement_tag, protein_seq in gene_info_list:
                if complement_tag == 'complement':
                    promoter_seq = str(Seq(seq[int(gene_pos_b):int(gene_pos_b)+1000]).reverse_complement()).upper()
                elif complement_tag == 'normal':
                    promoter_seq = seq[np.max([int(gene_pos_a)-1000-1, 0]):int(gene_pos_a)-1].upper()

                if len(promoter_seq) == 1000:
                    if gene_name not in promoter_seq_data.keys():
                        promoter_seq_data[gene_name] = promoter_seq
                if protein_seq:
                    if gene_name not in protein_seq_data.keys():
                        protein_seq_data[gene_name] = protein_seq


    promoter_seq_data = [[k, v] for k, v in promoter_seq_data.items() if k in Target_list]
    promoter_seq_data = [[k.upper(), v.upper()] for k, v in promoter_seq_data]
    protein_seq_data = [[k, v] for k, v in protein_seq_data.items() if k in TF_list]
    protein_seq_data = [[k.upper(), v.upper()] for k, v in protein_seq_data]
    pd.DataFrame(promoter_seq_data, columns=['gene_name', 'seq']).to_csv(promoter_seq_file, sep='\t', index=False, header=False)
    pd.DataFrame(protein_seq_data, columns=['gene_name', 'seq']).to_csv(protein_seq_file, sep='\t', index=False, header=False)
    logger.info('Promoter sequence file: %s' % promoter_seq_file)
    logger.info('Protein sequence file: %s' % protein_seq_file)

    return np.unique(np.array(promoter_seq_data)[:,0]).tolist(), np.unique(np.array(protein_seq_data)[:,0]).tolist()


def preprocess_regulations(raw_regulation_file, regulation_file, promoter_seq_name_list, protein_seq_name_list):
    data = pd.read_csv(raw_regulation_file, header=0, sep='\t', low_memory=False)
    data = data[['Name.TF', 'Name.Target']]
    data.columns = ['TF', 'Target']

    # replace the '-' with np.nan
    data = data.replace('-', np.nan)

    # delete the rows that have np.nan
    data = data.dropna(axis=0, how='any')

    # upper the TF and Target
    data['TF'] = data['TF'].str.upper()
    data['Target'] = data['Target'].str.upper()

    # only keep the TFs and Targets that have sequences
    data = data[data['TF'].isin(protein_seq_name_list)]
    data = data[data['Target'].isin(promoter_seq_name_list)]

    # delete the TF column that has the same name as the Target column
    data = data[data['TF'] != data['Target']]

    # delete the duplicate rows
    data = data.drop_duplicates()
    
    # save the regulation file
    data.to_csv(regulation_file, index=False)

    TF = data['TF'].unique()
    Target = np.unique(np.concatenate((data['Target'].unique(), data['TF'].unique())))
    TF_num = len(TF)
    Target_num = len(Target)

    POS_num = len(data)
    NEG_num = TF_num * Target_num - TF_num - POS_num
    POS_NEG_ratio = POS_num / NEG_num
    Density = POS_num / (TF_num * Target_num - TF_num)
    logger.info('Regulation file: %s' % regulation_file)
    logger.info('TF_num: %d, Target_num: %d, POS_num: %d, NEG_num: %d, POS_NEG_ratio: %f, Density: %f' % (TF_num, Target_num, POS_num, NEG_num, POS_NEG_ratio, Density))


if __name__ == '__main__':
    raw_data_dir = './species_specific_data/RawData/'
    processed_data_dir = './species_specific_data/ProcessedData/'

    # Homo sapiens
    logger.info('HomoSapiens start.')
    os.makedirs(os.path.join(processed_data_dir, 'HomoSapiens'), exist_ok=True)
    raw_gbff_file = os.path.join(raw_data_dir, 'HomoSapiens', 'GCF_000001405.40_GRCh38.p14_genomic.gbff')
    raw_regulation_file = os.path.join(raw_data_dir, 'HomoSapiens', 'TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv')
    promoter_seq_file = os.path.join(processed_data_dir, 'HomoSapiens', 'promoter.seq')
    protein_seq_file = os.path.join(processed_data_dir, 'HomoSapiens', 'protein.seq')
    regulation_file = os.path.join(processed_data_dir, 'HomoSapiens', 'regulation.csv')
    promoter_seq_name_list, protein_seq_name_list = preprocess_sequences(raw_gbff_file, raw_regulation_file, promoter_seq_file, protein_seq_file)
    preprocess_regulations(raw_regulation_file, regulation_file, promoter_seq_name_list, protein_seq_name_list)
    logger.info('HomoSapiens done.')

    # Mus musculus
    logger.info('MusMusculus start.')
    os.makedirs(os.path.join(processed_data_dir, 'MusMusculus'), exist_ok=True)
    raw_gbff_file = os.path.join(raw_data_dir, 'MusMusculus', 'GCF_000001635.27_GRCm39_genomic.gbff')
    raw_regulation_file = os.path.join(raw_data_dir, 'MusMusculus', 'TFLink_Mus_musculus_interactions_All_simpleFormat_v1.0.tsv')
    promoter_seq_file = os.path.join(processed_data_dir, 'MusMusculus', 'promoter.seq')
    protein_seq_file = os.path.join(processed_data_dir, 'MusMusculus', 'protein.seq')
    regulation_file = os.path.join(processed_data_dir, 'MusMusculus', 'regulation.csv')
    promoter_seq_name_list, protein_seq_name_list = preprocess_sequences(raw_gbff_file, raw_regulation_file, promoter_seq_file, protein_seq_file)
    preprocess_regulations(raw_regulation_file, regulation_file, promoter_seq_name_list, protein_seq_name_list)
    logger.info('MusMusculus done.')
