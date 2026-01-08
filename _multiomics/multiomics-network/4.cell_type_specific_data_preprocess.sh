#!/bin/bash

# Erythroid Cells (ERY)
ery_expression_file='./cell_type_specific_data/RawData/ERY/cd34_multiome_rna_Ery.csv'
ery_regulation_file='./cell_type_specific_data/RawData/ERY/cd34_multiome_figR_markers_cisCorrp_0.1_Ery.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/ERY
python 4.cell_type_specific_data_preprocess.py \
-e $ery_expression_file \
-f $ery_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/ERY/


# Hematopoietic Stem Cells (HSC)
hsc_expression_file='./cell_type_specific_data/RawData/HSC/cd34_multiome_rna_HSC.csv'
hsc_regulation_file='./cell_type_specific_data/RawData/HSC/cd34_multiome_figR_markers_cisCorrp_0.1_HSC.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/HSC
python 4.cell_type_specific_data_preprocess.py \
-e $hsc_expression_file \
-f $hsc_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/HSC/


# MegaKaryocyte-Erythroid Progenitor Cells (MEP)
mep_expression_file='./cell_type_specific_data/RawData/MEP/cd34_multiome_rna_MEP.csv'
mep_regulation_file='./cell_type_specific_data/RawData/MEP/cd34_multiome_figR_markers_cisCorrp_0.1_MEP.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/MEP
python 4.cell_type_specific_data_preprocess.py \
-e $mep_expression_file \
-f $mep_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/MEP/


# Conventional Dendritic Cell (cDC)
cdc_expression_file='./cell_type_specific_data/RawData/CDC/cd34_multiome_rna_cDC.csv'
cdc_regulation_file='./cell_type_specific_data/RawData/CDC/cd34_multiome_figR_cutoff7_cisCorrp_0.1_cDC.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/CDC
python 4.cell_type_specific_data_preprocess.py \
-e $cdc_expression_file \
-f $cdc_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/CDC/


# Common Lymphoid Progenitor Cells (CLP)
clp_expression_file='./cell_type_specific_data/RawData/CLP/cd34_multiome_rna_CLP.csv'
clp_regulation_file='./cell_type_specific_data/RawData/CLP/cd34_multiome_figR_cutoff7_cisCorrp_0.1_CLP.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/CLP
python 4.cell_type_specific_data_preprocess.py \
-e $clp_expression_file \
-f $clp_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/CLP/


# Hematopoietic Multipotent Progenitor Cells (HMP)
hmp_expression_file='./cell_type_specific_data/RawData/HMP/cd34_multiome_rna_HMP.csv'
hmp_regulation_file='./cell_type_specific_data/RawData/HMP/cd34_multiome_figR_cutoff7_cisCorrp_0.1_HMP.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/HMP
python 4.cell_type_specific_data_preprocess.py \
-e $hmp_expression_file \
-f $hmp_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/HMP/


# Monocytes (MONO)
mono_expression_file='./cell_type_specific_data/RawData/MONO/cd34_multiome_rna_Mono.csv'
mono_regulation_file='./cell_type_specific_data/RawData/MONO/cd34_multiome_figR_cutoff7_cisCorrp_0.1_Mono.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/MONO
python 4.cell_type_specific_data_preprocess.py \
-e $mono_expression_file \
-f $mono_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/MONO/


# Plasmacytoid Dendritic Cells (pDC)
pdc_expression_file='./cell_type_specific_data/RawData/PDC/cd34_multiome_rna_pDC.csv'
pdc_regulation_file='./cell_type_specific_data/RawData/PDC/cd34_multiome_figR_cutoff7_cisCorrp_0.1_pDC.csv'
mkdir -p ./cell_type_specific_data/ProcessedData/PDC
python 4.cell_type_specific_data_preprocess.py \
-e $pdc_expression_file \
-f $pdc_regulation_file \
-b False \
-o ./cell_type_specific_data/ProcessedData/PDC/
