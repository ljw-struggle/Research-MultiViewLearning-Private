#!/bin/bash

python main.py \
    --data_dir ./data/murine_cellline/ \
    --result_dir ./result/murine_cellline/ \
    --source_dataset_name murine_N2_SCP_exp.csv \
    --source_metadata_name murine_N2_SCP_meta.csv \
    --target_dataset_name murine_nanoPOTS_SCP_exp.csv \
    --target_metadata_name murine_nanoPOTS_SCP_meta.csv \
    --source_sample_num 4000 \
    --target_sample_num 1000 \
    --cell_type_name CellType \
    --cell_type_list C10 SVEC RAW \
    --target_type simulated \
    --mixup_sample_size 15 \
    --hvp_num 500 \
    --epoch_num 30 \
    --batch_size 50 \
    --learning_rate 0.0001

############################################################################################################
# from collections import defaultdict
# option_list = defaultdict(list)
# if dataset == 'human_breast_atlas_PP':
#     option_list['data_dir'] = './data/human_breast_atlas_PP/'
#     option_list['result_dir'] = './result/human_breast_atlas_PP/'
#     option_list['source_dataset_name'] = 'Human_Breast_Atlas_scProteome_normed_aligned_individual1.h5ad'
#     option_list['source_metadata_name'] = None
#     option_list['target_dataset_name'] = 'Human_Breast_Atlas_scProteome_normed_aligned_individual3.h5ad'
#     option_list['target_metadata_name'] = None
#     option_list['source_sample_num'] = 4000
#     option_list['target_sample_num'] = 1000
#     option_list['cell_type_name'] = 'cell_type'
#     option_list['cell_type_list'] = None
#     option_list['target_type'] = "simulated"
#     option_list['mixup_sample_size'] = 50
#     option_list['hvp_num'] = 0
#     option_list['epoch_num'] = 30
#     option_list['batch_size'] = 50
#     option_list['learning_rate'] = 0.0001
# if dataset == 'murine_cellline':
#     option_list['data_dir']='./data/murine_cellline/'
#     option_list['result_dir'] = "./result/murine_cellline/"
#     option_list['source_dataset_name'] = 'murine_N2_SCP_exp.csv'
#     option_list['source_metadata_name'] = 'murine_N2_SCP_meta.csv'
#     option_list['target_dataset_name'] = 'murine_nanoPOTS_SCP_exp.csv'
#     option_list['target_metadata_name'] = 'murine_nanoPOTS_SCP_meta.csv'
#     option_list['source_sample_num'] = 4000
#     option_list['target_sample_num'] = 1000	
#     option_list['cell_type_name']="CellType"
#     option_list['cell_type_list']=['C10','SVEC','RAW']
#     option_list['target_type'] = "simulated"
#     option_list['mixup_sample_size'] = 15
#     option_list['hvp_num'] = 500
#     option_list['epoch_num'] = 30
#     option_list['batch_size'] = 50
#     option_list['learning_rate'] = 0.0001
# if dataset == 'human_cellline':
#     option_list['data_dir']='./data/human_cellline/'
#     option_list['result_dir'] = './result/human_cellline/'
#     option_list['source_dataset_name'] = 'pSCoPE_Huffman_PDAC+pSCoPE_Leduc+SCoPE2_Leduc_integrated_SCP.h5ad'
#     option_list['source_metadata_name'] = None
#     option_list['target_dataset_name'] = 'T-SCP+plexDIA_integrated_SCP.h5ad'
#     option_list['target_metadata_name'] = None
#     option_list['source_sample_num'] = 4000
#     option_list['target_sample_num'] = 1000	
#     option_list['cell_type_name']="cell_type"
#     option_list['cell_type_list']=None
#     option_list['target_type'] = "simulated"
#     option_list['mixup_sample_size'] = 50
#     option_list['hvp_num'] = 500
#     option_list['epoch_num'] = 30
#     option_list['batch_size'] = 50
#     option_list['learning_rate'] = 0.0001