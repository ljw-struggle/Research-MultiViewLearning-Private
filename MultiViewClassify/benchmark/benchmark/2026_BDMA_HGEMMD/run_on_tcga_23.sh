#!/bin/bash
# nohup bash run.sh > ./run.log 2>&1 &
# ps -ef | grep run.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep main.py | grep -v grep | awk '{print $2}' | xargs kill -9
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
}
trap cleanup SIGINT SIGTERM EXIT

RUN_REPEAT_COMPARE_EXP()
{
    device=$1
    data_dir=$2
    result_dir=$3
    model=$4
    extra_args="${@:5}"
    seed_list=(0 8 16 24 32 40 48 56 64 72)
    for seed in ${seed_list[@]}
    do
        echo "device: $device, data: $data_dir, seed: $seed"
        mkdir -p $result_dir
        CUDA_VISIBLE_DEVICES=$device python ${model}.py --data_dir $data_dir --output_dir $result_dir --seed $seed --verbose 1 $extra_args > $result_dir/${seed}_output.log 2>&1 &
        wait
    done
}

####### Compare Experiment #######
date; echo "Start Running!" # 10*12*1 = 10
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/DNN/TCGA-23 1_DNN &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/GNN_GCN/TCGA-23 2_GNN --graph_model GCN &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/GNN_GAT/TCGA-23 2_GNN --graph_model GAT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/MMD/TCGA-23 3_MMD &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/MMDT/TCGA-23 3_MMDT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/MMDTT/TCGA-23 3_MMDTT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/GEMMD_GCN/TCGA-23 4_GEMMD --graph_model GCN &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/GEMMD_GAT/TCGA-23 4_GEMMD --graph_model GAT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/HF/TCGA-23 5_HF &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/MORE/TCGA-23 5_MORE &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/HFMMD/TCGA-23 6_HFMMD &
# wait

RUN_REPEAT_COMPARE_EXP 0  ./data/data_DeePathNet/TCGA-23 ./result/data_DeePathNet/HGEMMD/TCGA-23 7_HGEMMD &
wait

date; echo "All Done!"