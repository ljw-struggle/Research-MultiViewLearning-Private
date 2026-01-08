#!/bin/bash
# nohup bash run_main_method.sh > ./run_main_method.log 2>&1 &
# ps -ef | grep run_main_method.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep main.py | grep -v grep | awk '{print $2}' | xargs kill -9
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
}
trap cleanup SIGINT SIGTERM EXIT

# mkdir -p ./result/data_bulk_multiomics/DUCMME/BRCA/
# mkdir -p ./result/data_bulk_multiomics/DUCMME/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/DUCMME/LGG/
# mkdir -p ./result/data_sc_multiomics/DUCMME/DOGMA/
# mkdir -p ./result/data_sc_multiomics/DUCMME/TEA/
# mkdir -p ./result/data_sc_multiomics/DUCMME/NEAT/
# CUDA_VISIBLE_DEVICES=0 python 4_DUCMME.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/DUCMME/BRCA/ > ./result/data_bulk_multiomics/DUCMME/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 4_DUCMME.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/DUCMME/KIPAN/ > ./result/data_bulk_multiomics/DUCMME/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 4_DUCMME.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/DUCMME/LGG/ > ./result/data_bulk_multiomics/DUCMME/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 4_DUCMME.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/DUCMME/DOGMA/ > ./result/data_sc_multiomics/DUCMME/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 4_DUCMME.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/DUCMME/TEA/ > ./result/data_sc_multiomics/DUCMME/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python 4_DUCMME.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/DUCMME/NEAT/ > ./result/data_sc_multiomics/DUCMME/NEAT/output.log 2>&1 &
# wait

mkdir -p ./result/data_bulk_multiomics/HDUR/BRCA/
mkdir -p ./result/data_bulk_multiomics/HDUR/KIPAN/
mkdir -p ./result/data_bulk_multiomics/HDUR/LGG/
mkdir -p ./result/data_sc_multiomics/HDUR/DOGMA/
mkdir -p ./result/data_sc_multiomics/HDUR/TEA/
mkdir -p ./result/data_sc_multiomics/HDUR/NEAT/
CUDA_VISIBLE_DEVICES=0 python 2_HDUR.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/HDUR/BRCA/ > ./result/data_bulk_multiomics/HDUR/BRCA/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python 2_HDUR.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/HDUR/KIPAN/ > ./result/data_bulk_multiomics/HDUR/KIPAN/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python 2_HDUR.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/HDUR/LGG/ > ./result/data_bulk_multiomics/HDUR/LGG/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python 2_HDUR.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/HDUR/DOGMA/ > ./result/data_sc_multiomics/HDUR/DOGMA/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python 2_HDUR.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/HDUR/TEA/ > ./result/data_sc_multiomics/HDUR/TEA/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python 2_HDUR.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/HDUR/NEAT/ > ./result/data_sc_multiomics/HDUR/NEAT/output.log 2>&1 &
wait