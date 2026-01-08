#!/bin/bash
# nohup bash run.sh > ./run.log 2>&1 &
# ps -ef | grep run.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep main.py | grep -v grep | awk '{print $2}' | xargs kill -9
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
}
trap cleanup SIGINT SIGTERM EXIT

RUN_REPEAT_HP_SEARCH_K()
{
    device=$1
    data_dir=$2
    result_dir=$3
    model=$4
    k_list=("${@:5}")
    # k_list=(50 100 150 200 250 300 350)
    seed_list=(0 8 16 24 32)
    for k in ${k_list[@]}
    do
        for seed in ${seed_list[@]}
        do
            echo "device: $device, data: $data_dir, k: $k, seed: $seed"
            mkdir -p $result_dir
            CUDA_VISIBLE_DEVICES=$device python ${model}.py --data_dir $data_dir --output_dir $result_dir --k $k --seed $seed > $result_dir/k_${k}_seed_${seed}_output.log 2>&1 &
            wait
        done
    done
}

RUN_REPEAT_HP_SEARCH_LAMBDA()
{
    device=$1
    data_dir=$2
    result_dir=$3
    model=$4
    lambda_1_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
    lambda_2_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
    for lambda_1 in ${lambda_1_list[@]}
    do
        for lambda_2 in ${lambda_2_list[@]}
        do
            echo "device: $device, data: $data_dir, lambda_1: $lambda_1, lambda_2: $lambda_2"
            mkdir -p $result_dir
            CUDA_VISIBLE_DEVICES=$device python ${model}.py --data_dir $data_dir --output_dir $result_dir --lambda_1 $lambda_1 --lambda_2 $lambda_2 > $result_dir/l1_${lambda_1}_l2_${lambda_2}_output.log 2>&1 &
            wait
        done
    done
}

RUN_REPEAT_MODAL_MISSING()
{
    device=$1
    data_dir=$2
    result_dir=$3
    model=$4
    extra_args="${@:5}"
    modality_list=(0 1 2 3 4 5 6)
    seed_list=(0 8 16 24 32)
    for modality in ${modality_list[@]}
    do
        for seed in ${seed_list[@]}
        do
            echo "device: $device, data: $data_dir, modality: $modality, seed: $seed"
            mkdir -p $result_dir
            CUDA_VISIBLE_DEVICES=$device python ${model}.py --data_dir $data_dir --output_dir $result_dir --modality $modality --seed $seed $extra_args > $result_dir/modality_${modality}_seed_${seed}_output.log 2>&1 &
            wait
        done
    done
}

RUN_REPEAT_NOISY_CONDITION()
{
    device=$1
    data_dir=$2
    result_dir=$3
    model=$4
    extra_args="${@:5}"
    sigma_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
    seed_list=(0 8 16 24 32)
    for sigma in ${sigma_list[@]}
    do
        for seed in ${seed_list[@]}
        do
            echo "device: $device, data: $data_dir, sigma: $sigma, seed: $seed"
            mkdir -p $result_dir
            CUDA_VISIBLE_DEVICES=$device python ${model}.py --data_dir $data_dir --output_dir $result_dir --sigma $sigma --seed $seed $extra_args > $result_dir/sigma_${sigma}_seed_${seed}_output.log 2>&1 &
            wait
        done
    done
}

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

# ####### Hyperparameter Search #######
# date; echo "Start Running!" # 100 + 100 + 8*5 + 7*5 = 275
# RUN_REPEAT_HP_SEARCH_K 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD/BRCA_hp_k 7_HGEMMD 100 200 300 400 500 600 700 800 &
# wait
# RUN_REPEAT_HP_SEARCH_K 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD/ROSMAP_hp_k 7_HGEMMD 50 100 150 200 250 300 350 &
# wait
# RUN_REPEAT_HP_SEARCH_LAMBDA 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD/BRCA_hp_l1_l2 7_HGEMMD &
# wait
# RUN_REPEAT_HP_SEARCH_LAMBDA 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD/ROSMAP_hp_l1_l2 7_HGEMMD &
# wait
# date; echo "All Done!"


# ####### Modality Missing #######
# date; echo "Start Running!" # 7*5*3*2 = 210
# RUN_REPEAT_MODAL_MISSING 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD/BRCA_modality_missing 7_HGEMMD &
# wait
# RUN_REPEAT_MODAL_MISSING 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD/ROSMAP_modality_missing 7_HGEMMD &
# wait
# RUN_REPEAT_MODAL_MISSING 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/GEMMD_GCN/BRCA_modality_missing 4_GEMMD --graph_model GCN &
# wait
# RUN_REPEAT_MODAL_MISSING 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/GEMMD_GCN/ROSMAP_modality_missing 4_GEMMD --graph_model GCN &
# wait
# RUN_REPEAT_MODAL_MISSING 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/MMD/BRCA_modality_missing 3_MMD &
# wait
# RUN_REPEAT_MODAL_MISSING 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/MMD/ROSMAP_modality_missing 3_MMD &
# wait
# date; echo "All Done!"


# ####### Noisy Condition #######
# date; echo "Start Running!" # 10*5*3*2 = 300
# RUN_REPEAT_NOISY_CONDITION 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD/BRCA_noisy_condition 7_HGEMMD &
# wait
# RUN_REPEAT_NOISY_CONDITION 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD/ROSMAP_noisy_condition 7_HGEMMD &
# wait
# RUN_REPEAT_NOISY_CONDITION 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/GEMMD_GCN/BRCA_noisy_condition 4_GEMMD --graph_model GCN &
# wait
# RUN_REPEAT_NOISY_CONDITION 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/GEMMD_GCN/ROSMAP_noisy_condition 4_GEMMD --graph_model GCN &
# wait
# RUN_REPEAT_NOISY_CONDITION 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/MMD/BRCA_noisy_condition 3_MMD &
# wait
# RUN_REPEAT_NOISY_CONDITION 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/MMD/ROSMAP_noisy_condition 3_MMD &
# wait
# date; echo "All Done!"


####### Compare Experiment #######
# date; echo "Start Running!" # 10*12*4 = 480
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/DNN/BRCA 1_DNN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/DNN/KIPAN 1_DNN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/DNN/LGG 1_DNN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/DNN/ROSMAP 1_DNN &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/GNN_GCN/BRCA 2_GNN --graph_model GCN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/GNN_GCN/KIPAN 2_GNN --graph_model GCN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/GNN_GCN/LGG 2_GNN --graph_model GCN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/GNN_GCN/ROSMAP 2_GNN --graph_model GCN &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/GNN_GAT/BRCA 2_GNN --graph_model GAT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/GNN_GAT/KIPAN 2_GNN --graph_model GAT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/GNN_GAT/LGG 2_GNN --graph_model GAT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/GNN_GAT/ROSMAP 2_GNN --graph_model GAT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/MMD/BRCA/ 3_MMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/MMD/KIPAN/ 3_MMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/MMD/LGG/ 3_MMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/MMD/ROSMAP/ 3_MMD &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/MMDT/BRCA/ 3_MMDT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/MMDT/KIPAN/ 3_MMDT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/MMDT/LGG/ 3_MMDT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/MMDT/ROSMAP/ 3_MMDT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/MMDTT/BRCA/ 3_MMDTT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/MMDTT/KIPAN/ 3_MMDTT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/MMDTT/LGG/ 3_MMDTT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/MMDTT/ROSMAP/ 3_MMDTT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/GEMMD_GCN/BRCA/ 4_GEMMD --graph_model GCN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/GEMMD_GCN/KIPAN/ 4_GEMMD --graph_model GCN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/GEMMD_GCN/LGG/ 4_GEMMD --graph_model GCN &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/GEMMD_GCN/ROSMAP/ 4_GEMMD --graph_model GCN &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/GEMMD_GAT/BRCA/ 4_GEMMD --graph_model GAT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/GEMMD_GAT/KIPAN/ 4_GEMMD --graph_model GAT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/GEMMD_GAT/LGG/ 4_GEMMD --graph_model GAT &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/GEMMD_GAT/ROSMAP/ 4_GEMMD --graph_model GAT &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HF/BRCA/ 5_HF &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HF/KIPAN/ 5_HF &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HF/LGG/ 5_HF &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HF/ROSMAP/ 5_HF &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/MORE/BRCA/ 5_MORE &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/MORE/KIPAN/ 5_MORE &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/MORE/LGG/ 5_MORE &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/MORE/ROSMAP/ 5_MORE &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HFMMD/BRCA/ 6_HFMMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HFMMD/KIPAN/ 6_HFMMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HFMMD/LGG/ 6_HFMMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HFMMD/ROSMAP/ 6_HFMMD &
# wait

# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD/BRCA 7_HGEMMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HGEMMD/KIPAN 7_HGEMMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HGEMMD/LGG 7_HGEMMD &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD/ROSMAP 7_HGEMMD &
# wait

## revise hgnn layer from one to two
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HFMMD_revise_hgnn_layer/BRCA/ 6_HFMMD_revise_hgnn_layer &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HFMMD_revise_hgnn_layer/KIPAN/ 6_HFMMD_revise_hgnn_layer &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HFMMD_revise_hgnn_layer/LGG/ 6_HFMMD_revise_hgnn_layer &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HFMMD_revise_hgnn_layer/ROSMAP/ 6_HFMMD_revise_hgnn_layer &
# wait

## revise sparsity from mean to sum
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD_revise_sparsity/BRCA 7_HGEMMD_revise_sparsity &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HGEMMD_revise_sparsity/KIPAN 7_HGEMMD_revise_sparsity &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HGEMMD_revise_sparsity/LGG 7_HGEMMD_revise_sparsity &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD_revise_sparsity/ROSMAP 7_HGEMMD_revise_sparsity &
# wait

# remove gating
RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD_remove_gating/BRCA 7_HGEMMD_remove_gating &
wait
RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HGEMMD_remove_gating/KIPAN 7_HGEMMD_remove_gating &
wait
RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HGEMMD_remove_gating/LGG 7_HGEMMD_remove_gating &
wait
RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD_remove_gating/ROSMAP 7_HGEMMD_remove_gating &
wait

## revise hgnn layer from one to two
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD_revise_hgnn_layer/BRCA 7_HGEMMD_revise_hgnn_layer &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HGEMMD_revise_hgnn_layer/KIPAN 7_HGEMMD_revise_hgnn_layer &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HGEMMD_revise_hgnn_layer/LGG 7_HGEMMD_revise_hgnn_layer &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD_revise_hgnn_layer/ROSMAP 7_HGEMMD_revise_hgnn_layer &
# wait

## revise hgnn layer from fix to dynamic
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/BRCA ./result/data_MOGONET/HGEMMD_revise_hgnn_dynamic/BRCA 7_HGEMMD_revise_hgnn_dynamic &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/KIPAN ./result/data_MOGONET/HGEMMD_revise_hgnn_dynamic/KIPAN 7_HGEMMD_revise_hgnn_dynamic &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/LGG ./result/data_MOGONET/HGEMMD_revise_hgnn_dynamic/LGG 7_HGEMMD_revise_hgnn_dynamic &
# wait
# RUN_REPEAT_COMPARE_EXP 0  ./data/data_MOGONET/ROSMAP ./result/data_MOGONET/HGEMMD_revise_hgnn_dynamic/ROSMAP 7_HGEMMD_revise_hgnn_dynamic &
# wait

# date; echo "All Done!"
