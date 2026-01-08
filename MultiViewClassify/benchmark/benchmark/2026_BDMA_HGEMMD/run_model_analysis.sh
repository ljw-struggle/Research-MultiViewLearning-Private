#!/bin/bash
# nohup bash run.sh > ./run.log 2>&1 &
# ps -ef | grep run.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep main.py | grep -v grep | awk '{print $2}' | xargs kill -9


mkdir -p ./result/model_analysis
# CUDA_VISIBLE_DEVICES=0 python 1_DNN.py --verbose 2 > ./result/model_analysis/DNN.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 python 2_GNN.py --graph_model GCN --verbose 2 > ./result/model_analysis/GNN_GCN.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 python 2_GNN.py --graph_model GAT --verbose 2 > ./result/model_analysis/GNN_GAT.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 python 3_MMD.py --verbose 2 > ./result/model_analysis/MMD.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 python 3_MMDT.py --verbose 2 > ./result/model_analysis/MMDT.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 python 3_MMDTT.py --verbose 2 > ./result/model_analysis/MMDTT.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 python 4_GEMMD.py --graph_model GCN --verbose 2 > ./result/model_analysis/GEMMD_GCN.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0 python 4_GEMMD.py --graph_model GAT --verbose 2 > ./result/model_analysis/GEMMD_GAT.log 2>&1 &
# wait
CUDA_VISIBLE_DEVICES=0 python 5_HF.py --verbose 2 > ./result/model_analysis/HF.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python 5_MORE.py --verbose 2 > ./result/model_analysis/MORE.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python 6_HFMMD.py --verbose 2 > ./result/model_analysis/HFMMD.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 python 7_HGEMMD.py --verbose 2 > ./result/model_analysis/HGEMMD.log 2>&1 &
wait