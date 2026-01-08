#!/bin/bash
# $ bash 1.uncertainty_estimation.sh > ./1.uncertainty_estimation.log 2>&1 &
# $ ps -ef | grep 1.uncertainty_estimation.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# $ ps -ef | grep 1.uncertainty_estimation.py | grep -v grep | awk '{print $2}' | xargs kill -9

# Run different seeds
echo "Start running 1.uncertainty_estimation.sh..."
if [ ! -d "../result/SCoPE2_Specht/uncertainty_estimation/seed_222_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/uncertainty_estimation/seed_222_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=0 python -u 1.uncertainty_estimation.py --seed 222 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/uncertainty_estimation/seed_222_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/uncertainty_estimation/seed_222_epoch_100_patience_15/uncertainty_estimation.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht/uncertainty_estimation/seed_444_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/uncertainty_estimation/seed_444_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=1 python -u 1.uncertainty_estimation.py --seed 444 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/uncertainty_estimation/seed_444_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/uncertainty_estimation/seed_444_epoch_100_patience_15/uncertainty_estimation.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht/uncertainty_estimation/seed_666_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/uncertainty_estimation/seed_666_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=2 python -u 1.uncertainty_estimation.py --seed 666 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/uncertainty_estimation/seed_666_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/uncertainty_estimation/seed_666_epoch_100_patience_15/uncertainty_estimation.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht/uncertainty_estimation/seed_888_epoch_100_patience_15/" ]; then
    mkdir -p ../result/SCoPE2_Specht/uncertainty_estimation/seed_888_epoch_100_patience_15/
    CUDA_VISIBLE_DEVICES=3 python -u 1.uncertainty_estimation.py --seed 888 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht/ --result_dir ../result/SCoPE2_Specht/uncertainty_estimation/seed_888_epoch_100_patience_15/ \
        > ../result/SCoPE2_Specht/uncertainty_estimation/seed_888_epoch_100_patience_15/uncertainty_estimation.log 2>&1 &
fi

wait
echo "All done!"

# Run different noise levels
echo "Start running 1.uncertainty_estimation.sh..."
if [ ! -d "../result/SCoPE2_Specht_noise/row_noise/uncertainty_estimation" ]; then
    mkdir -p ../result/SCoPE2_Specht_noise/row_noise/uncertainty_estimation/
    CUDA_VISIBLE_DEVICES=0 python -u 1.uncertainty_estimation.py --seed 666 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht_noise/row_noise/ --result_dir ../result/SCoPE2_Specht_noise/row_noise/uncertainty_estimation/ \
        > ../result/SCoPE2_Specht_noise/row_noise/uncertainty_estimation/uncertainty_estimation.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht_noise/column_noise/uncertainty_estimation" ]; then
    mkdir -p ../result/SCoPE2_Specht_noise/column_noise/uncertainty_estimation/
    CUDA_VISIBLE_DEVICES=1 python -u 1.uncertainty_estimation.py --seed 666 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht_noise/column_noise/ --result_dir ../result/SCoPE2_Specht_noise/column_noise/uncertainty_estimation/ \
        > ../result/SCoPE2_Specht_noise/column_noise/uncertainty_estimation/uncertainty_estimation.log 2>&1 &
fi

if [ ! -d "../result/SCoPE2_Specht_noise/mosaic_noise/uncertainty_estimation" ]; then
    mkdir -p ../result/SCoPE2_Specht_noise/mosaic_noise/uncertainty_estimation/
    CUDA_VISIBLE_DEVICES=2 python -u 1.uncertainty_estimation.py --seed 666 --num_epochs 100 --patience 15 \
        --data_dir ../data/SCoPE2_Specht_noise/mosaic_noise/ --result_dir ../result/SCoPE2_Specht_noise/mosaic_noise/uncertainty_estimation/ \
        > ../result/SCoPE2_Specht_noise/mosaic_noise/uncertainty_estimation/uncertainty_estimation.log 2>&1 &
fi

wait
echo "All done!"



