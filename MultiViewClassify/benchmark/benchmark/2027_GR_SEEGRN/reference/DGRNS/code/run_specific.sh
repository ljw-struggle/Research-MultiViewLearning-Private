#!/bin/bash
# nohup bash run_specific.sh >> ./run_specific.log 2>&1 &
# ps -ef | grep run_specific.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep main.py | grep -v grep | awk '{print $2}' | xargs kill -9
RUN_REPEAT()
{
    device=$1
    data_path=$2
    result_path=$3

    seed_list=(8 16 24 32 40)
    for seed in ${seed_list[@]}
    do
        echo "device: $device, data_path: $data_path, result_path: $result_path, seed: $seed"
        mkdir -p $result_path/seed/$seed
        CUDA_VISIBLE_DEVICES=$device python main.py --data_path $data_path --result_path $result_path/seed/$seed --seed $seed > $result_path/seed/$seed/log.txt 2>&1
    done

    if [[ $result_path =~ "tissue_specific" ]]
    then
        cell_list=(10 20 50 500 5000)
        for cell in ${cell_list[@]}
        do
            echo "device: $device, data_path: $data_path, result_path: $result_path, cell: $cell"
            mkdir -p $result_path/cell/$cell
            CUDA_VISIBLE_DEVICES=$device python main.py --data_path $data_path --result_path $result_path/cell/$cell --cell $cell > $result_path/cell/$cell/log.txt 2>&1
        done

        size_list=(0.05 0.1 0.2 0.5 1.0)
        for size in ${size_list[@]}
        do
            echo "device: $device, data_path: $data_path, result_path: $result_path, size: $size"
            mkdir -p $result_path/size/$size
            CUDA_VISIBLE_DEVICES=$device python main.py --data_path $data_path --result_path $result_path/size/$size --size $size > $result_path/size/$size/log.txt 2>&1
        done
    fi
}


#### tissue_specific_data ####
#### cell_type_specific_data ####
RUN_REPEAT 0 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/ERY ../result/cell_type_specific_data/ERY &
RUN_REPEAT 1 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/HSC ../result/cell_type_specific_data/HSC &
RUN_REPEAT 2 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/MEP ../result/cell_type_specific_data/MEP &
RUN_REPEAT 3 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/CDC ../result/cell_type_specific_data/CDC &
wait
RUN_REPEAT 0 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/CLP ../result/cell_type_specific_data/CLP &
RUN_REPEAT 1 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/HMP ../result/cell_type_specific_data/HMP &
RUN_REPEAT 2 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/PDC ../result/cell_type_specific_data/PDC &
RUN_REPEAT 3 ../../../Bioinfor-GRNInfer/data/cell_type_specific_data/ProcessedData/MONO ../result/cell_type_specific_data/MONO &
wait
RUN_REPEAT 0 ../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Human-Blood ../result/tissue_specific_data/Human-Blood &
RUN_REPEAT 1 ../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Human-Bone-Marrow ../result/tissue_specific_data/Human-Bone-Marrow &
RUN_REPEAT 2 ../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Human-Cerebral-Cortex ../result/tissue_specific_data/Human-Cerebral-Cortex &
RUN_REPEAT 3 ../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Mouse-Brain-Cortex ../result/tissue_specific_data/Mouse-Brain-Cortex &
wait
RUN_REPEAT 0 ../../../Bioinfor-GRNInfer/data/tissue_specific_data/ProcessedData/Mouse-Skin ../result/tissue_specific_data/Mouse-Skin &


#### benchmark_data ####
# RUN_REPEAT 0  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/hESC/500/specific ../result/benchmark_data/hESC/500/specific &
# RUN_REPEAT 1  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/hESC/1000/specific ../result/benchmark_data/hESC/1000/specific &
# RUN_REPEAT 2  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/hHep/500/specific ../result/benchmark_data/hHep/500/specific &
# RUN_REPEAT 3  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/hHep/1000/specific ../result/benchmark_data/hHep/1000/specific &
# RUN_REPEAT 0  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mDC/500/specific ../result/benchmark_data/mDC/500/specific &
# RUN_REPEAT 1  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mDC/1000/specific ../result/benchmark_data/mDC/1000/specific &
# RUN_REPEAT 2  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mESC/500/specific ../result/benchmark_data/mESC/500/specific &
# RUN_REPEAT 3  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mESC/1000/specific ../result/benchmark_data/mESC/1000/specific &
# RUN_REPEAT 0  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mHSC-E/500/specific ../result/benchmark_data/mHSC-E/500/specific &
# RUN_REPEAT 1  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mHSC-E/1000/specific ../result/benchmark_data/mHSC-E/1000/specific &
# RUN_REPEAT 2  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mHSC-GM/500/specific ../result/benchmark_data/mHSC-GM/500/specific &
# RUN_REPEAT 3  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mHSC-GM/1000/specific ../result/benchmark_data/mHSC-GM/1000/specific &
# RUN_REPEAT 0  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mHSC-L/500/specific ../result/benchmark_data/mHSC-L/500/specific &
# RUN_REPEAT 1  ../../../Bioinfor-GRNInfer/data/benchmark_data/ProcessedData/mHSC-L/1000/specific ../result/benchmark_data/mHSC-L/1000/specific &

wait