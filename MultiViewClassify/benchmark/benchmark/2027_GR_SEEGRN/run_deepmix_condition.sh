#!/bin/bash
# nohup bash run_deepmix_condition.sh > run_deepmix_condition.log 2>&1 &

mkdir -p ../result/DeepMixModel/tissue_specific_data/Human-Blood-Transfer
CUDA_VISIBLE_DEVICES=0 python main_deepmix.py --config ./config/DeepMixModel/tissue_specific_data/Human-Blood-Transfer/config_mmdynamics.json \
                                              --run_id ORIGIN --seed 8 --cell 2000 > ../result/DeepMixModel/tissue_specific_data/Human-Blood-Transfer/mmdynamics.log 2>&1 &

mkdir -p ../result/DeepMixModel/tissue_specific_data/Mouse-Brain-Cortex-Transfer
CUDA_VISIBLE_DEVICES=1 python main_deepmix.py --config ./config/DeepMixModel/tissue_specific_data/Mouse-Brain-Cortex-Transfer/config_mmdynamics.json \
                                              --run_id ORIGIN --seed 8 --cell 2000 > ../result/DeepMixModel/tissue_specific_data/Mouse-Brain-Cortex-Transfer/mmdynamics.log 2>&1 &

