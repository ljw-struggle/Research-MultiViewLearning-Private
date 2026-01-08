# -*- coding: utf-8 -*-
import os
import numpy as np


def load_all_meme():
    meme_dir = './_doc/human_HOCOMO/'
    meme_list = os.listdir(meme_dir)
    meme_dict = {}
    for meme_file in meme_list:
        meme_name = meme_file.split('_')[0]
        meme_path = os.path.join(meme_dir, meme_file)
        matrix = []
        matrix_started = False
        with open(meme_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('letter-probability matrix'):
                    matrix_started = True
                    continue
                if not matrix_started:
                    continue
                if line == '':
                    break
                probabilities = list(map(float, line.split()))
                matrix.append(probabilities)
        matrix = np.array(matrix)
        meme_dict[meme_name] = matrix
    return meme_dict


def load_all_pwm():
    pwm_dir = './_doc/human_HOCOMO_pwm/'
    pwm_list = os.listdir(pwm_dir)
    pwm_dict = {}
    for pwm_file in pwm_list:
        pwm_name = pwm_file.split('_')[0]
        pwm_path = os.path.join(pwm_dir, pwm_file)
        matrix = []
        with open(pwm_path, 'r') as file:
            file.readline()
            for line in file:
                line = line.strip()
                if line == '':
                    break
                probabilities = list(map(float, line.split()))
                matrix.append(probabilities)
        matrix = np.array(matrix)
        pwm_dict[pwm_name] = matrix
    return pwm_dict


def gene_to_one_hot(gene_sequence):
    base_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_sequence = []
    for base in gene_sequence:
        one_hot_sequence.append(base_dict.get(base, [0, 0, 0, 0]))
    one_hot_sequence = np.array(one_hot_sequence)
    return one_hot_sequence

