#!/bin/bash

# # Human Blood Transfer 1
# human_blood_expression_file='./tissue_specific_data/RawData/Human-Blood-Transfer/human_PBMC1.rna.normed.csv'
# human_blood_regulation_file='./tissue_specific_data/RawData/Human-Blood-Transfer/human_PBMC1_figR_cisCorrp_0.05_overlapHVG_562.csv'
# mkdir -p ./tissue_specific_data/ProcessedData/Human-Blood-Transfer/PBMC1
# python 5.tissue_specific_data_transfer_preprocess.py \
# -e $human_blood_expression_file \
# -f $human_blood_regulation_file \
# -b False \
# -o ./tissue_specific_data/ProcessedData/Human-Blood-Transfer/PBMC1

# # Human Blood Transfer 2
# human_blood_expression_file='./tissue_specific_data/RawData/Human-Blood-Transfer/human_PBMC2.rna.normed.csv'
# human_blood_regulation_file='./tissue_specific_data/RawData/Human-Blood-Transfer/human_PBMC2_figR_cisCorrp_0.05_overlapHVG_562.csv'
# mkdir -p ./tissue_specific_data/ProcessedData/Human-Blood-Transfer/PBMC2
# python 5.tissue_specific_data_transfer_preprocess.py \
# -e $human_blood_expression_file \
# -f $human_blood_regulation_file \
# -b False \
# -o ./tissue_specific_data/ProcessedData/Human-Blood-Transfer/PBMC2

# # Human Blood Transfer 3
# human_blood_expression_file='./tissue_specific_data/RawData/Human-Blood-Transfer/human_PBMC3.rna.normed.csv'
# human_blood_regulation_file='./tissue_specific_data/RawData/Human-Blood-Transfer/human_PBMC3_figR_cisCorrp_0.05_overlapHVG_562.csv'
# mkdir -p ./tissue_specific_data/ProcessedData/Human-Blood-Transfer/PBMC3
# python 5.tissue_specific_data_transfer_preprocess.py \
# -e $human_blood_expression_file \
# -f $human_blood_regulation_file \
# -b False \
# -o ./tissue_specific_data/ProcessedData/Human-Blood-Transfer/PBMC3

# Mouse Brain Transfer Adult
mouse_brain_expression_file='./tissue_specific_data/RawData/Mouse-Brain-Cortex-Transfer/mouse_adult_brain.rna.normed.csv'
mouse_brain_regulation_file='./tissue_specific_data/RawData/Mouse-Brain-Cortex-Transfer/mouse_adult_brain_figR_cisCorrp_0.1_overlapHVG_510.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Mouse-Brain-Cortex-Transfer/Adult
python 5.tissue_specific_data_transfer_preprocess.py \
-e $mouse_brain_expression_file \
-f $mouse_brain_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Mouse-Brain-Transfer/Adult

# Mouse Brain Transfer Fetal
mouse_brain_expression_file='./tissue_specific_data/RawData/Mouse-Brain-Transfer/mouse_fetal_brain.rna.normed.csv'
mouse_brain_regulation_file='./tissue_specific_data/RawData/Mouse-Brain-Transfer/mouse_fetal_brain_figR_cisCorrp_0.1_overlapHVG_510.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Mouse-Brain-Transfer/Fetal
python 5.tissue_specific_data_transfer_preprocess.py \
-e $mouse_brain_expression_file \
-f $mouse_brain_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Mouse-Brain-Transfer/Fetal