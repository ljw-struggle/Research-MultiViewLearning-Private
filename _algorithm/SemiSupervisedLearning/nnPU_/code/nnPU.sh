#!/bin/bash
mkdir -p ../result/exp-mnist-tlp
CUDA_VISIBLE_DEVICES=0 nohup python nnPU.py --preset exp-mnist-tlp > ../result/exp-mnist-tlp/result.txt 2>&1 &
mkdir -p ../result/exp-mnist-mlp
CUDA_VISIBLE_DEVICES=1 nohup python nnPU.py --preset exp-mnist-mlp > ../result/exp-mnist-mlp/result.txt 2>&1 &
mkdir -p ../result/exp-cifar-cnn
CUDA_VISIBLE_DEVICES=2 nohup python nnPU.py --preset exp-cifar-cnn > ../result/exp-cifar-cnn/result.txt 2>&1 &

