#!/bin/bash
# nohup bash run_sequence_pretrain.sh > run_sequence_pretrain.log 2>&1 &
gpu_device_list=(0 1 2 3)
config_dir_path_list=( \ 
./config/SequenceModel_PreTrain/species_specific_data/HomoSapiens \ 
./config/SequenceModel_PreTrain/species_specific_data/MusMusculus \
./config/SequenceModel_PreTrain/species_specific_data/CrossSpecies \
)
config_file_path_list=()

for config_dir_path in ${config_dir_path_list[@]}
do
    # Get a list of configuration files whose names start with "config_bert_LSTM"
    config_file_list=($(ls $config_dir_path | grep ^config_bert_LSTM))
    # Get a list of configuration files whose names dont start with "config_bert_LSTM"
    # config_file_list=($(ls $config_dir_path | grep -v ^config_bert_LSTM | grep ^config))
    for config_file in ${config_file_list[@]}
    do
        config_file_path_list+=($config_dir_path/$config_file)
    done
done

echo "CONFIG_FILE_PATH_LIST: ${config_file_path_list[@]}"
echo "The number of config files: ${#config_file_path_list[@]}"
echo "GPU_DEVICE_LIST: ${gpu_device_list[@]}"
echo "The number of gpu devices: ${#gpu_device_list[@]}"


RUN_CONFIG()
{
    # Get the first config file
    config_file_path=${config_file_path_list[0]}
    config_file_path_list=(${config_file_path_list[@]/$config_file_path})
    echo "CONFIG_FILE: $config_file_path"

    # Get the first gpu device
    gpu_device=${gpu_device_list[0]}
    # gpu_device_list=(${gpu_device_list[@]/$gpu_device})
    for i in ${!gpu_device_list[@]}
    do
        if [ ${gpu_device_list[i]} == $gpu_device ]
        then
            unset 'gpu_device_list[i]'
            break
        fi
    done
    gpu_device_list=(${gpu_device_list[@]})
    echo "GPU_DEVICE: $gpu_device"
    echo "GPU_DEVICE_LIST: ${gpu_device_list[@]}"
    
    # Get the result file path
    # replace the 'config' in config_file_path with 'result' and replace the 'json' with 'log'
    result_file_path=${config_file_path//config/result}
    result_file_path=${result_file_path//json/log}
    result_file_path=.$result_file_path
    result_dir_path=${result_file_path%/*}
    mkdir -p $result_dir_path

    # Run the config file
    # gpu_device is the remainder when divided by 4
    gpu_device=$(($gpu_device % 4))
    CUDA_VISIBLE_DEVICES=$gpu_device python main_sequence_pretrain.py --config $config_file_path --run_id ORIGIN > $result_file_path 2>&1 &
}


while [ ${#config_file_path_list[@]} -gt 0 ]
do
    if [ ${#gpu_device_list[@]} -gt 0 ]
    then
        RUN_CONFIG
    else
        echo "Waiting for the GPU device to be released..."
        date

        wait 
        gpu_device_list=(0 1 2 3)
    fi
done

wait

echo "All done!"


# ## Test
# # HomoSapiens
# CUDA_VISIBLE_DEVICES=3 python main_sequence_pretrain.py -m test -c ./config/SequenceModel_PreTrain/species_specific_data/HomoSapiens/config_bert_LSTM_3_true_BinaryCrossEntropy.json -r ../result/SequenceModel_PreTrain/species_specific_data/HomoSapiens/bert_LSTM_3_true_BinaryCrossEntropy/ORIGIN/model/model_best.pth -i Test_HomoSapiens -bs 1024 -ld HomosapiensLoader > ../result/SequenceModel_PreTrain/species_specific_data/HomoSapiens/bert_LSTM_3_true_BinaryCrossEntropy/Test_HomoSapiens.log
 
# CUDA_VISIBLE_DEVICES=3 python main_sequence_pretrain.py -m test -c ./config/SequenceModel_PreTrain/species_specific_data/HomoSapiens/config_bert_LSTM_3_true_BinaryCrossEntropy.json -r ../result/SequenceModel_PreTrain/species_specific_data/HomoSapiens/bert_LSTM_3_true_BinaryCrossEntropy/ORIGIN/model/model_best.pth -i Test_MusMusculus -bs 1024 -ld MusmusculusLoader > ../result/SequenceModel_PreTrain/species_specific_data/HomoSapiens/bert_LSTM_3_true_BinaryCrossEntropy/Test_MusMusculus.log
 
# # MusMusculus
# CUDA_VISIBLE_DEVICES=3 python main_sequence_pretrain.py -m test -c ./config/SequenceModel_PreTrain/species_specific_data/MusMusculus/config_bert_LSTM_3_true_BinaryCrossEntropy.json -r ../result/SequenceModel_PreTrain/species_specific_data/MusMusculus/bert_LSTM_3_true_BinaryCrossEntropy/ORIGIN/model/model_best.pth -i Test_HomoSapiens -bs 1024 -ld HomosapiensLoader > ../result/SequenceModel_PreTrain/species_specific_data/MusMusculus/bert_LSTM_3_true_BinaryCrossEntropy/Test_HomoSapiens.log
 
# CUDA_VISIBLE_DEVICES=3 python main_sequence_pretrain.py -m test -c ./config/SequenceModel_PreTrain/species_specific_data/MusMusculus/config_bert_LSTM_3_true_BinaryCrossEntropy.json -r ../result/SequenceModel_PreTrain/species_specific_data/MusMusculus/bert_LSTM_3_true_BinaryCrossEntropy/ORIGIN/model/model_best.pth -i Test_MusMusculus -bs 1024 -ld MusmusculusLoader > ../result/SequenceModel_PreTrain/species_specific_data/MusMusculus/bert_LSTM_3_true_BinaryCrossEntropy/Test_MusMusculus.log

# # CrossSpecies
# CUDA_VISIBLE_DEVICES=3 python main_sequence_pretrain.py -m test -c ./config/SequenceModel_PreTrain/species_specific_data/CrossSpecies/config_bert_LSTM_3_true_BinaryCrossEntropy.json -r ../result/SequenceModel_PreTrain/species_specific_data/CrossSpecies/bert_LSTM_3_true_BinaryCrossEntropy/ORIGIN/model/model_best.pth -i Test_HomoSapiens -bs 1024 -ld HomosapiensLoader > ../result/SequenceModel_PreTrain/species_specific_data/CrossSpecies/bert_LSTM_3_true_BinaryCrossEntropy/Test_HomoSapiens.log

# CUDA_VISIBLE_DEVICES=3 python main_sequence_pretrain.py -m test -c ./config/SequenceModel_PreTrain/species_specific_data/CrossSpecies/config_bert_LSTM_3_true_BinaryCrossEntropy.json -r ../result/SequenceModel_PreTrain/species_specific_data/CrossSpecies/bert_LSTM_3_true_BinaryCrossEntropy/ORIGIN/model/model_best.pth -i Test_MusMusculus -bs 1024 -ld MusmusculusLoader > ../result/SequenceModel_PreTrain/species_specific_data/CrossSpecies/bert_LSTM_3_true_BinaryCrossEntropy/Test_MusMusculus.log


# # # kill the process that contains "main_sequence_pretrain.py"
# # ps -ef | grep main_sequence_pretrain.py | grep -v grep | awk '{print $2}' | xargs kill -9

# # # kill the process that contains "run_sequence_pretrain.sh"
# # ps -ef | grep run_sequence_pretrain.sh | grep -v grep | awk '{print $2}' | xargs kill -9