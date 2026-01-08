#!/bin/bash
# $ bash 2.graph_contrastive_learning.sh > ./2.graph_contrastive_learning.log 2>&1 &
# $ ps -ef | grep 2.graph_contrastive_learning.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# $ ps -ef | grep 2.graph_contrastive_learning.py | grep -v grep | awk '{print $2}' | xargs kill -9

# Run different seeds
echo "Start running 2.graph_contrastive_learning.sh..."
if [ ! -d "../result/SCoPE2_Specht/graph_contrastive_learning/seed_222_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/graph_contrastive_learning/seed_222_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=0 python -u 2.graph_contrastive_learning.py --seed 222 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/graph_contrastive_learning/seed_222_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/graph_contrastive_learning/seed_222_epoch_100_patience_15/graph_contrastive_learning.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht/graph_contrastive_learning/seed_444_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/graph_contrastive_learning/seed_444_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=1 python -u 2.graph_contrastive_learning.py --seed 444 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/graph_contrastive_learning/seed_444_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/graph_contrastive_learning/seed_444_epoch_100_patience_15/graph_contrastive_learning.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht/graph_contrastive_learning/seed_666_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/graph_contrastive_learning/seed_666_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=2 python -u 2.graph_contrastive_learning.py --seed 666 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/graph_contrastive_learning/seed_666_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/graph_contrastive_learning/seed_666_epoch_100_patience_15/graph_contrastive_learning.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht/graph_contrastive_learning/seed_888_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/graph_contrastive_learning/seed_888_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=3 python -u 2.graph_contrastive_learning.py --seed 888 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/graph_contrastive_learning/seed_888_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/graph_contrastive_learning/seed_888_epoch_100_patience_15/graph_contrastive_learning.log 2>&1 &
fi

wait
echo "All done!"
