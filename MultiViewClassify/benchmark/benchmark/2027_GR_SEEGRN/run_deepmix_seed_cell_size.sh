#!/bin/bash
# nohup bash run_deepmix_seed_cell_size.sh > run_deepmix_seed_cell_size.log 2>&1 &
gpu_device_list=(0 1 2 3)
config_dir_path_list=( \
# ./config/DeepMixModel/tissue_specific_data/Human-Blood \
# ./config/DeepMixModel/tissue_specific_data/Human-Bone-Marrow \
# ./config/DeepMixModel/tissue_specific_data/Human-Cerebral-Cortex \
# ./config/DeepMixModel/tissue_specific_data/Mouse-Brain-Cortex \
# ./config/DeepMixModel/tissue_specific_data/Mouse-Skin \
)
config_file_path_list=()

for config_dir_path in ${config_dir_path_list[@]}
do
    # Get a list of configuration files whose names start with config_mmdynamics
    config_file_list=($(ls $config_dir_path | grep ^config_mmdynamics))
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
    result_file_path=${config_file_path//config/result} # result_file_path: ./result/DeepMixModel/benchmark_data/hESC/500/config_mmdynamics.json
    result_file_path=${result_file_path//json/log} # result_file_path: ./result/DeepMixModel/benchmark_data/hESC/500/config_mmdynamics.log
    result_file_path=.$result_file_path # result_file_path: ../result/DeepMixModel/benchmark_data/hESC/500/config_mmdynamics.log
    result_dir_path=${result_file_path%/*} # result_dir_path: ../result/DeepMixModel/benchmark_data/hESC/500
    mkdir -p $result_dir_path

    # Run the config file
    # gpu_device is the remainder when divided by 4
    gpu_device=$(($gpu_device % 8))


    RUN_REPEAT()
    {
        seed_list=(8 16 24 32 40)
        for seed in ${seed_list[@]}
        do
            result_file_path_seed=${result_file_path//.log/_seed_${seed}.log}
            CUDA_VISIBLE_DEVICES=$gpu_device python main_deepmix.py --config $config_file_path --run_id seed_$seed --seed $seed > ${result_file_path_seed} 2>&1
        done

        # if result_file_path contains "tissue_specific" or "cell_type_specific", then run the following code
        # if [[ $result_file_path =~ "tissue_specific" || $result_file_path =~ "cell_type_specific" ]]
        if [[ $result_file_path =~ "tissue_specific" ]]
        then
            cell_list=(10 20 50 500 5000)
            for cell in ${cell_list[@]}
            do
                result_file_path_cell=${result_file_path//.log/_cell_${cell}.log}
                CUDA_VISIBLE_DEVICES=$gpu_device python main_deepmix.py --config $config_file_path --run_id cell_$cell --cell $cell > ${result_file_path_cell} 2>&1
            done

            size_list=(0.05 0.1 0.2 0.5 1.0)
            for size in ${size_list[@]}
            do
                result_file_path_size=${result_file_path//.log/_size_${size}.log}
                CUDA_VISIBLE_DEVICES=$gpu_device python main_deepmix.py --config $config_file_path --run_id size_$size --size $size > ${result_file_path_size} 2>&1
            done
        fi
    }
    RUN_REPEAT &    
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

# # kill the process that contains "main_deepmix.py"
# ps -ef | grep main_deepmix.py | grep -v grep | awk '{print $2}' | xargs kill -9

# # kill the process that contains "run_deepmix_seed_cell_size.sh"
# ps -ef | grep run_deepmix_seed_cell_size.sh | grep -v grep | awk '{print $2}' | xargs kill -9

