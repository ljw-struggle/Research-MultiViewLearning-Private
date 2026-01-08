#!/bin/bash
# nohup bash run_config_generate.sh > run_config_generate.log 2>&1 &

# # sequence pretrain
# config_dir_path_list_sequence_pretrain=( \
# ./SequenceModel_PreTrain/species_specific_data/HomoSapiens \
# ./SequenceModel_PreTrain/species_specific_data/MusMusculus \
# ./SequenceModel_PreTrain/species_specific_data/CrossSpecies \
# )

# for config_dir_path in ${config_dir_path_list_sequence_pretrain[@]}
# do
#     echo $config_dir_path
#     cd $config_dir_path
    
#     config_file_list=(`ls config_*.json`)
#     if [ ${#config_file_list[@]} -gt 0 ]; then
#         rm config_*.json
#     fi
#     bash _config_generate.sh
#     cd -
# done

# deepmix
config_dir_path_list_deepmix=( \
./DeepMixModel/tissue_specific_data/Human-Blood \
./DeepMixModel/tissue_specific_data/Human-Blood-Transfer \
./DeepMixModel/tissue_specific_data/Human-Bone-Marrow \
./DeepMixModel/tissue_specific_data/Human-Cerebral-Cortex \
./DeepMixModel/tissue_specific_data/Mouse-Brain-Cortex \
./DeepMixModel/tissue_specific_data/Mouse-Brain-Cortex-Transfer \
./DeepMixModel/tissue_specific_data/Mouse-Skin \
./DeepMixModel/cell_type_specific_data/ERY \
./DeepMixModel/cell_type_specific_data/HSC \
./DeepMixModel/cell_type_specific_data/MEP \
./DeepMixModel/cell_type_specific_data/CDC \
./DeepMixModel/cell_type_specific_data/CLP \
./DeepMixModel/cell_type_specific_data/HMP \
./DeepMixModel/cell_type_specific_data/PDC \
./DeepMixModel/cell_type_specific_data/MONO \
)

for config_dir_path in ${config_dir_path_list_deepmix[@]}
do
    echo $config_dir_path
    cd $config_dir_path
    
    config_file_list=(`ls config_*.json`)
    if [ ${#config_file_list[@]} -gt 0 ]; then
        rm config_*.json
    fi
    bash _config_generate.sh
    cd -
done
