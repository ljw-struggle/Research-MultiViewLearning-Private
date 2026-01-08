#!/bin/bash
# nohup bash run_ablation.sh > ./run_ablation.log 2>&1 &
# ps -ef | grep run_ablation.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep main.py | grep -v grep | awk '{print $2}' | xargs kill -9
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
}
trap cleanup SIGINT SIGTERM EXIT

# # Ablation Study
# mkdir -p ./result/ablation/TEA/AE/
# mkdir -p ./result/ablation/TEA/HDUR/
# mkdir -p ./result/ablation/TEA/DEC/
# mkdir -p ./result/ablation/TEA/DUCMME/
# CUDA_VISIBLE_DEVICES=0 python 1_AE.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/ablation/TEA/AE/ > ./result/ablation/TEA/AE/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 2_HDUR.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/ablation/TEA/HDUR/ > ./result/ablation/TEA/HDUR/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 3_DEC.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/ablation/TEA/DEC/ > ./result/ablation/TEA/DEC/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 4_DUCMME.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/ablation/TEA/DUCMME/ > ./result/ablation/TEA/DUCMME/output.log 2>&1 &
# wait