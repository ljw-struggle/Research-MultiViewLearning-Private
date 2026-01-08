#!/bin/bash

# Human Blood
human_blood_expression_file='./tissue_specific_data/RawData/Human-Blood/human_PBMC.rna.normed.csv'
human_blood_regulation_file='./tissue_specific_data/RawData/Human-Blood/figR_cutoff5_cisCorrp_0.05.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Human-Blood
python 3.tissue_specific_data_preprocess.py \
-e $human_blood_expression_file \
-f $human_blood_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Human-Blood/

# Human Bone Marrow
human_bonemarrow_expression_file='./tissue_specific_data/RawData/Human-Bone-Marrow/cd34_multiome_rna.csv'
human_bonemarrow_regulation_file='./tissue_specific_data/RawData/Human-Bone-Marrow/figR_cutoff6_cisCorrp_0.05.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Human-Bone-Marrow
python 3.tissue_specific_data_preprocess.py \
-e $human_bonemarrow_expression_file \
-f $human_bonemarrow_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Human-Bone-Marrow/

# Human Cerebral Cortex
human_cerebral_cortex_expression_file='./tissue_specific_data/RawData/Human-Cerebral-Cortex/human_cerebral_cortex.rna.normed.csv'
human_cerebral_cortex_regulation_file='./tissue_specific_data/RawData/Human-Cerebral-Cortex/figR_cutoff8_cisCorrp_0.05.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Human-Cerebral-Cortex
python 3.tissue_specific_data_preprocess.py \
-e $human_cerebral_cortex_expression_file \
-f $human_cerebral_cortex_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Human-Cerebral-Cortex/

# Mouse Brain
mouse_brain_expression_file='./tissue_specific_data/RawData/Mouse-Brain/mouse_brain.rna.normed.csv'
mouse_brain_regulation_file='./tissue_specific_data/RawData/Mouse-Brain/figR_cutoff10_cisCorrp_0.05.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Mouse-Brain
python 3.tissue_specific_data_preprocess.py \
-e $mouse_brain_expression_file \
-f $mouse_brain_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Mouse-Brain/

# Mouse Brain Cortex
mouse_brain_cortex_expression_file='./tissue_specific_data/RawData/Mouse-Brain-Cortex/mouse_brain_cortex.rna.normed.csv'
mouse_brain_cortex_regulation_file='./tissue_specific_data/RawData/Mouse-Brain-Cortex/figR_cutoff4_cisCorrp_0.05.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Mouse-Brain-Cortex
python 3.tissue_specific_data_preprocess.py \
-e $mouse_brain_cortex_expression_file \
-f $mouse_brain_cortex_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Mouse-Brain-Cortex/

# Mouse Lung
mouse_lung_expression_file='./tissue_specific_data/RawData/Mouse-Lung/mouse_lung.rna.normed.csv'
mouse_lung_regulation_file='./tissue_specific_data/RawData/Mouse-Lung/figR_cutoff4_cisCorrp_0.1.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Mouse-Lung
python 3.tissue_specific_data_preprocess.py \
-e $mouse_lung_expression_file \
-f $mouse_lung_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Mouse-Lung/

# Mouse Skin
mouse_skin_expression_file='./tissue_specific_data/RawData/Mouse-Skin/shareseq_skin_RNAnorm_final.csv'
mouse_skin_regulation_file='./tissue_specific_data/RawData/Mouse-Skin/shareseq_skin_figR.csv'
mkdir -p ./tissue_specific_data/ProcessedData/Mouse-Skin
python 3.tissue_specific_data_preprocess.py \
-e $mouse_skin_expression_file \
-f $mouse_skin_regulation_file \
-b False \
-o ./tissue_specific_data/ProcessedData/Mouse-Skin/

