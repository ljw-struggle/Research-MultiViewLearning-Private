# -*- coding: utf-8 -*-
import re
import os
import pandas as pd

def read_log(file_path):
    # Read the text file
    with open(file_path, 'r') as file:
        content = file.read()
    # Extract test metrics
    test_metrics = re.findall(r'Test AUC: ([\d.]+) AUPR: ([\d.]+) AUPR_norm: ([\d.]+) EP: ([\d.]+) EPR: ([\d.]+)', content)
    test_auc, test_aupr, test_aupr_norm, test_ep, test_epr = test_metrics[0]
    # Extract model information
    macs = re.findall(r'MACs: ([\d.]+ [A-Z]+)', content)[0]
    params = re.findall(r'Params: ([\d.]+ [A-Z]+)', content)[0]
    memory_usage = re.findall(r'Memory Usage: ([\d.]+ [A-Z]+)', content)[0]
    return test_auc, test_aupr, test_aupr_norm, test_ep, test_epr, macs, params, memory_usage

def collect_different_configucellns(dir, dataset_name, only_seed=True):
    result = pd.DataFrame(columns=['dataset_name', 'seed', 'cell', 'size', 'auroc', 'aupr', 'aupr_norm', 'ep', 'epr', 'macs', 'params', 'memory_usage'])
    seed_8_result = read_log(os.path.join(dir, 'seed/8/log.txt'))
    seed_16_result = read_log(os.path.join(dir, 'seed/16/log.txt'))
    seed_24_result = read_log(os.path.join(dir, 'seed/24/log.txt'))
    seed_32_result = read_log(os.path.join(dir, 'seed/32/log.txt'))
    seed_40_result = read_log(os.path.join(dir, 'seed/40/log.txt'))
    result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 5000, 'size': 1.0, 'auroc': seed_8_result[0], 'aupr': seed_8_result[1], 'aupr_norm': seed_8_result[2], 'ep': seed_8_result[3], 'epr': seed_8_result[4], 'macs': seed_8_result[5], 'params': seed_8_result[6], 'memory_usage': seed_8_result[7]}, index=[0])], ignore_index=True)
    result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 16, 'cell': 5000, 'size': 1.0, 'auroc': seed_16_result[0], 'aupr': seed_16_result[1], 'aupr_norm': seed_16_result[2], 'ep': seed_16_result[3], 'epr': seed_16_result[4], 'macs': seed_16_result[5], 'params': seed_16_result[6], 'memory_usage': seed_16_result[7]}, index=[0])], ignore_index=True)
    result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 24, 'cell': 5000, 'size': 1.0, 'auroc': seed_24_result[0], 'aupr': seed_24_result[1], 'aupr_norm': seed_24_result[2], 'ep': seed_24_result[3], 'epr': seed_24_result[4], 'macs': seed_24_result[5], 'params': seed_24_result[6], 'memory_usage': seed_24_result[7]}, index=[0])], ignore_index=True)
    result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 32, 'cell': 5000, 'size': 1.0, 'auroc': seed_32_result[0], 'aupr': seed_32_result[1], 'aupr_norm': seed_32_result[2], 'ep': seed_32_result[3], 'epr': seed_32_result[4], 'macs': seed_32_result[5], 'params': seed_32_result[6], 'memory_usage': seed_32_result[7]}, index=[0])], ignore_index=True)
    result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 40, 'cell': 5000, 'size': 1.0, 'auroc': seed_40_result[0], 'aupr': seed_40_result[1], 'aupr_norm': seed_40_result[2], 'ep': seed_40_result[3], 'epr': seed_40_result[4], 'macs': seed_40_result[5], 'params': seed_40_result[6], 'memory_usage': seed_40_result[7]}, index=[0])], ignore_index=True)
    if not only_seed:
        cell_0_2_result = read_log(os.path.join(dir, 'cell/10/log.txt'))
        cell_0_4_result = read_log(os.path.join(dir, 'cell/20/log.txt'))
        cell_0_6_result = read_log(os.path.join(dir, 'cell/50/log.txt'))
        cell_0_8_result = read_log(os.path.join(dir, 'cell/500/log.txt'))
        cell_1_0_result = read_log(os.path.join(dir, 'cell/5000/log.txt'))
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 10, 'size': 1.0, 'auroc': cell_0_2_result[0], 'aupr': cell_0_2_result[1], 'aupr_norm': cell_0_2_result[2], 'ep': cell_0_2_result[3], 'epr': cell_0_2_result[4], 'macs': cell_0_2_result[5], 'params': cell_0_2_result[6], 'memory_usage': cell_0_2_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 20, 'size': 1.0, 'auroc': cell_0_4_result[0], 'aupr': cell_0_4_result[1], 'aupr_norm': cell_0_4_result[2], 'ep': cell_0_4_result[3], 'epr': cell_0_4_result[4], 'macs': cell_0_4_result[5], 'params': cell_0_4_result[6], 'memory_usage': cell_0_4_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 50, 'size': 1.0, 'auroc': cell_0_6_result[0], 'aupr': cell_0_6_result[1], 'aupr_norm': cell_0_6_result[2], 'ep': cell_0_6_result[3], 'epr': cell_0_6_result[4], 'macs': cell_0_6_result[5], 'params': cell_0_6_result[6], 'memory_usage': cell_0_6_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 500, 'size': 1.0, 'auroc': cell_0_8_result[0], 'aupr': cell_0_8_result[1], 'aupr_norm': cell_0_8_result[2], 'ep': cell_0_8_result[3], 'epr': cell_0_8_result[4], 'macs': cell_0_8_result[5], 'params': cell_0_8_result[6], 'memory_usage': cell_0_8_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 5000, 'size': 1.0, 'auroc': cell_1_0_result[0], 'aupr': cell_1_0_result[1], 'aupr_norm': cell_1_0_result[2], 'ep': cell_1_0_result[3], 'epr': cell_1_0_result[4], 'macs': cell_1_0_result[5], 'params': cell_1_0_result[6], 'memory_usage': cell_1_0_result[7]}, index=[0])], ignore_index=True)
        size_0_2_result = read_log(os.path.join(dir, 'size/0.05/log.txt'))
        size_0_4_result = read_log(os.path.join(dir, 'size/0.1/log.txt'))
        size_0_6_result = read_log(os.path.join(dir, 'size/0.2/log.txt'))
        size_0_8_result = read_log(os.path.join(dir, 'size/0.5/log.txt'))
        size_1_0_result = read_log(os.path.join(dir, 'size/1.0/log.txt'))
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 5000, 'size': 0.05, 'auroc': size_0_2_result[0], 'aupr': size_0_2_result[1], 'aupr_norm': size_0_2_result[2], 'ep': size_0_2_result[3], 'epr': size_0_2_result[4], 'macs': size_0_2_result[5], 'params': size_0_2_result[6], 'memory_usage': size_0_2_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 5000, 'size': 0.1, 'auroc': size_0_4_result[0], 'aupr': size_0_4_result[1], 'aupr_norm': size_0_4_result[2], 'ep': size_0_4_result[3], 'epr': size_0_4_result[4], 'macs': size_0_4_result[5], 'params': size_0_4_result[6], 'memory_usage': size_0_4_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 5000, 'size': 0.2, 'auroc': size_0_6_result[0], 'aupr': size_0_6_result[1], 'aupr_norm': size_0_6_result[2], 'ep': size_0_6_result[3], 'epr': size_0_6_result[4], 'macs': size_0_6_result[5], 'params': size_0_6_result[6], 'memory_usage': size_0_6_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 5000, 'size': 0.5, 'auroc': size_0_8_result[0], 'aupr': size_0_8_result[1], 'aupr_norm': size_0_8_result[2], 'ep': size_0_8_result[3], 'epr': size_0_8_result[4], 'macs': size_0_8_result[5], 'params': size_0_8_result[6], 'memory_usage': size_0_8_result[7]}, index=[0])], ignore_index=True)
        result = pd.concat([result, pd.DataFrame({'dataset_name': dataset_name, 'seed': 8, 'cell': 5000, 'size': 1.0, 'auroc': size_1_0_result[0], 'aupr': size_1_0_result[1], 'aupr_norm': size_1_0_result[2], 'ep': size_1_0_result[3], 'epr': size_1_0_result[4], 'macs': size_1_0_result[5], 'params': size_1_0_result[6], 'memory_usage': size_1_0_result[7]}, index=[0])], ignore_index=True)
    return result

def collect_result_tissue_specific(dir, result_file):
    tissue_human_blood = os.path.join(dir, 'tissue_specific_data/Human-Blood')
    tissue_human_blood_result = collect_different_configucellns(tissue_human_blood, 'tissue_human_blood', only_seed=False)
    tissue_human_bone_marrow = os.path.join(dir, 'tissue_specific_data/Human-Bone-Marrow')
    tissue_human_bone_marrow_result = collect_different_configucellns(tissue_human_bone_marrow, 'tissue_human_bone_marrow', only_seed=False)
    tissue_human_cerebral_cortex = os.path.join(dir, 'tissue_specific_data/Human-Cerebral-Cortex')
    tissue_human_cerebral_cortex_result = collect_different_configucellns(tissue_human_cerebral_cortex, 'tissue_human_cerebral_cortex', only_seed=False)
    tissue_mouse_brain_cortex = os.path.join(dir, 'tissue_specific_data/Mouse-Brain-Cortex')
    tissue_mouse_brain_cortex_result = collect_different_configucellns(tissue_mouse_brain_cortex, 'tissue_mouse_brain_cortex', only_seed=False)
    tissue_mouse_skin = os.path.join(dir, 'tissue_specific_data/Mouse-Skin')
    tissue_mouse_skin_result = collect_different_configucellns(tissue_mouse_skin, 'tissue_mouse_skin', only_seed=False)
    # Merge all data frames into one
    result = pd.concat([tissue_human_blood_result, tissue_human_bone_marrow_result, tissue_human_cerebral_cortex_result, tissue_mouse_brain_cortex_result, tissue_mouse_skin_result], ignore_index=True)
    # Save the data frame
    result.to_csv(result_file, index=False)

def collect_result_cell_type_specific(dir, result_file):
    cell_type_ery = os.path.join(dir, 'cell_type_specific_data/ERY')
    cell_type_ery_result = collect_different_configucellns(cell_type_ery, 'cell_type_ery', only_seed=True)
    cell_type_hsc = os.path.join(dir, 'cell_type_specific_data/HSC')
    cell_type_hsc_result = collect_different_configucellns(cell_type_hsc, 'cell_type_hsc', only_seed=True)
    cell_type_mep = os.path.join(dir, 'cell_type_specific_data/MEP')
    cell_type_mep_result = collect_different_configucellns(cell_type_mep, 'cell_type_mep', only_seed=True)
    cell_type_cdc = os.path.join(dir, 'cell_type_specific_data/CDC')
    cell_type_cdc_result = collect_different_configucellns(cell_type_cdc, 'cell_type_cdc', only_seed=True)
    cell_type_clp = os.path.join(dir, 'cell_type_specific_data/CLP')
    cell_type_clp_result = collect_different_configucellns(cell_type_clp, 'cell_type_clp', only_seed=True)
    cell_type_hmp = os.path.join(dir, 'cell_type_specific_data/HMP')
    cell_type_hmp_result = collect_different_configucellns(cell_type_hmp, 'cell_type_hmp', only_seed=True)
    cell_type_mono = os.path.join(dir, 'cell_type_specific_data/MONO')
    cell_type_mono_result = collect_different_configucellns(cell_type_mono, 'cell_type_mono', only_seed=True)
    cell_type_pdc = os.path.join(dir, 'cell_type_specific_data/PDC')
    cell_type_pdc_result = collect_different_configucellns(cell_type_pdc, 'cell_type_pdc', only_seed=True)
    # Merge all data frames into one
    result = pd.concat([cell_type_ery_result, cell_type_hsc_result, cell_type_mep_result, cell_type_cdc_result, cell_type_clp_result, cell_type_hmp_result, cell_type_mono_result, cell_type_pdc_result], ignore_index=True)
    # Save the data frame
    result.to_csv(result_file, index=False)

if __name__ == '__main__':
    collect_result_tissue_specific('./GENELink/result/', './_result/result_tissue_specific_genelink.csv')
    collect_result_tissue_specific('./DeepTFni/result/', './_result/result_tissue_specific_deeptfni.csv')
    collect_result_tissue_specific('./DeepSEM/result/', './_result/result_tissue_specific_deepsem.csv')
    collect_result_tissue_specific('./CNNC/result/', './_result/result_tissue_specific_cnnc.csv')
    collect_result_tissue_specific('./DGRNS/result/', './_result/result_tissue_specific_dgrns.csv')
    collect_result_tissue_specific('./STGRNS/result/', './_result/result_tissue_specific_stgrns.csv')
    collect_result_cell_type_specific('./GENELink/result/', './_result/result_cell_type_specific_genelink.csv')
    collect_result_cell_type_specific('./DeepTFni/result/', './_result/result_cell_type_specific_deeptfni.csv')
    collect_result_cell_type_specific('./DeepSEM/result/', './_result/result_cell_type_specific_deepsem.csv')
    collect_result_cell_type_specific('./CNNC/result/', './_result/result_cell_type_specific_cnnc.csv')
    collect_result_cell_type_specific('./DGRNS/result/', './_result/result_cell_type_specific_dgrns.csv')
    collect_result_cell_type_specific('./STGRNS/result/', './_result/result_cell_type_specific_stgrns.csv')
