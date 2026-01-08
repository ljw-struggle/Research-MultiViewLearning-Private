#!/bin/bash
# nohup bash run_method_comparison.sh > ./run_method_comparison.log 2>&1 &
# ps -ef | grep run_method_comparison.sh | grep -v grep | awk '{print $2}' | xargs kill -9
# ps -ef | grep *.py | grep -v grep | awk '{print $2}' | xargs kill -9
cleanup() {
    echo "Cleaning up..."
    kill $(jobs -p) 2>/dev/null
}
trap cleanup SIGINT SIGTERM EXIT

# # ==================== Dataset: LGG ====================
mkdir -p ./result/data_bulk_multiomics/AE/LGG/
mkdir -p ./result/data_bulk_multiomics/DAE/LGG/
mkdir -p ./result/data_bulk_multiomics/VAE/LGG/
mkdir -p ./result/data_bulk_multiomics/ZINBVAE/LGG/
# mkdir -p ./result/data_bulk_multiomics/MOFA2/LGG/
# mkdir -p ./result/data_bulk_multiomics/MOJITOO/LGG/
# mkdir -p ./result/data_bulk_multiomics/SEURAT/LGG/
# mkdir -p ./result/data_bulk_multiomics/MOGCN/LGG/
# mkdir -p ./result/data_bulk_multiomics/SCMHNN/LGG/
# mkdir -p ./result/data_bulk_multiomics/SCMDC/LGG/
# mkdir -p ./result/data_bulk_multiomics/MATILDA/LGG/
CUDA_VISIBLE_DEVICES=0 python reference_1_ae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/AE/LGG/ --epoch_num 50 > ./result/data_bulk_multiomics/AE/LGG/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python reference_1_dae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/DAE/LGG/ --epoch_num 50 > ./result/data_bulk_multiomics/DAE/LGG/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python reference_1_vae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/VAE/LGG/ --epoch_num 50 > ./result/data_bulk_multiomics/VAE/LGG/output.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python reference_1_zinbvae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/ZINBVAE/LGG/ --epoch_num 50 > ./result/data_bulk_multiomics/ZINBVAE/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_ae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/AE/LGG/ > ./result/data_bulk_multiomics/AE/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_dae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/DAE/LGG/ > ./result/data_bulk_multiomics/DAE/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_vae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/VAE/LGG/ > ./result/data_bulk_multiomics/VAE/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_zinbvae.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/ZINBVAE/LGG/ > ./result/data_bulk_multiomics/ZINBVAE/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mofa2.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/MOFA2/LGG/ > ./result/data_bulk_multiomics/MOFA2/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mojitoo.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/MOJITOO/LGG/ > ./result/data_bulk_multiomics/MOJITOO/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_seurat.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/SEURAT/LGG/ > ./result/data_bulk_multiomics/SEURAT/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_moGCN.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/MOGCN/LGG/ > ./result/data_bulk_multiomics/MOGCN/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMHNN.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/SCMHNN/LGG/ > ./result/data_bulk_multiomics/SCMHNN/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMDC.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/SCMDC/LGG/ > ./result/data_bulk_multiomics/SCMDC/LGG/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_matilda.py --data_dir ./data/data_bulk_multiomics/LGG/ --output_dir ./result/data_bulk_multiomics/MATILDA/LGG/ > ./result/data_bulk_multiomics/MATILDA/LGG/output.log 2>&1 &
# wait

# # ==================== Dataset: BRCA ====================
# mkdir -p ./result/data_bulk_multiomics/AE/BRCA/
# mkdir -p ./result/data_bulk_multiomics/DAE/BRCA/
# mkdir -p ./result/data_bulk_multiomics/VAE/BRCA/
# mkdir -p ./result/data_bulk_multiomics/ZINBVAE/BRCA/
# mkdir -p ./result/data_bulk_multiomics/MOFA2/BRCA/
# mkdir -p ./result/data_bulk_multiomics/MOJITOO/BRCA/
# mkdir -p ./result/data_bulk_multiomics/SEURAT/BRCA/
# mkdir -p ./result/data_bulk_multiomics/MOGCN/BRCA/
# mkdir -p ./result/data_bulk_multiomics/SCMHNN/BRCA/
# mkdir -p ./result/data_bulk_multiomics/SCMDC/BRCA/
# mkdir -p ./result/data_bulk_multiomics/MATILDA/BRCA/
# CUDA_VISIBLE_DEVICES=0 python reference_1_ae.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/AE/BRCA/ > ./result/data_bulk_multiomics/AE/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_dae.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/DAE/BRCA/ > ./result/data_bulk_multiomics/DAE/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_vae.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/VAE/BRCA/ > ./result/data_bulk_multiomics/VAE/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_zinbvae.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/ZINBVAE/BRCA/ > ./result/data_bulk_multiomics/ZINBVAE/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mofa2.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/MOFA2/BRCA/ > ./result/data_bulk_multiomics/MOFA2/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mojitoo.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/MOJITOO/BRCA/ > ./result/data_bulk_multiomics/MOJITOO/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_seurat.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/SEURAT/BRCA/ > ./result/data_bulk_multiomics/SEURAT/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_moGCN.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/MOGCN/BRCA/ > ./result/data_bulk_multiomics/MOGCN/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMHNN.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/SCMHNN/BRCA/ > ./result/data_bulk_multiomics/SCMHNN/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMDC.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/SCMDC/BRCA/ > ./result/data_bulk_multiomics/SCMDC/BRCA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_matilda.py --data_dir ./data/data_bulk_multiomics/BRCA/ --output_dir ./result/data_bulk_multiomics/MATILDA/BRCA/ > ./result/data_bulk_multiomics/MATILDA/BRCA/output.log 2>&1 &
# wait

# # ==================== Dataset: KIPAN ====================
# mkdir -p ./result/data_bulk_multiomics/AE/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/DAE/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/VAE/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/ZINBVAE/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/MOFA2/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/MOJITOO/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/SEURAT/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/MOGCN/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/SCMHNN/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/SCMDC/KIPAN/
# mkdir -p ./result/data_bulk_multiomics/MATILDA/KIPAN/
# CUDA_VISIBLE_DEVICES=0 python reference_1_ae.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/AE/KIPAN/ > ./result/data_bulk_multiomics/AE/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_dae.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/DAE/KIPAN/ > ./result/data_bulk_multiomics/DAE/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_vae.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/VAE/KIPAN/ > ./result/data_bulk_multiomics/VAE/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_zinbvae.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/ZINBVAE/KIPAN/ > ./result/data_bulk_multiomics/ZINBVAE/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mofa2.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/MOFA2/KIPAN/ > ./result/data_bulk_multiomics/MOFA2/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mojitoo.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/MOJITOO/KIPAN/ > ./result/data_bulk_multiomics/MOJITOO/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_seurat.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/SEURAT/KIPAN/ > ./result/data_bulk_multiomics/SEURAT/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_moGCN.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/MOGCN/KIPAN/ > ./result/data_bulk_multiomics/MOGCN/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMHNN.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/SCMHNN/KIPAN/ > ./result/data_bulk_multiomics/SCMHNN/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMDC.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/SCMDC/KIPAN/ > ./result/data_bulk_multiomics/SCMDC/KIPAN/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_matilda.py --data_dir ./data/data_bulk_multiomics/KIPAN/ --output_dir ./result/data_bulk_multiomics/MATILDA/KIPAN/ > ./result/data_bulk_multiomics/MATILDA/KIPAN/output.log 2>&1 &
# wait

# # ==================== Dataset: DOGMA ====================
# mkdir -p ./result/data_sc_multiomics/AE/DOGMA/
# mkdir -p ./result/data_sc_multiomics/DAE/DOGMA/
# mkdir -p ./result/data_sc_multiomics/VAE/DOGMA/
# mkdir -p ./result/data_sc_multiomics/ZINBVAE/DOGMA/
# mkdir -p ./result/data_sc_multiomics/MOFA2/DOGMA/
# mkdir -p ./result/data_sc_multiomics/MOJITOO/DOGMA/
# mkdir -p ./result/data_sc_multiomics/SEURAT/DOGMA/
# mkdir -p ./result/data_sc_multiomics/MOGCN/DOGMA/
# mkdir -p ./result/data_sc_multiomics/SCMHNN/DOGMA/
# mkdir -p ./result/data_sc_multiomics/SCMDC/DOGMA/
# mkdir -p ./result/data_sc_multiomics/MATILDA/DOGMA/
# CUDA_VISIBLE_DEVICES=0 python reference_1_ae.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/AE/DOGMA/ > ./result/data_sc_multiomics/AE/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_dae.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/DAE/DOGMA/ > ./result/data_sc_multiomics/DAE/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_vae.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/VAE/DOGMA/ > ./result/data_sc_multiomics/VAE/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_zinbvae.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/ZINBVAE/DOGMA/ > ./result/data_sc_multiomics/ZINBVAE/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mofa2.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/MOFA2/DOGMA/ > ./result/data_sc_multiomics/MOFA2/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mojitoo.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/MOJITOO/DOGMA/ > ./result/data_sc_multiomics/MOJITOO/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_seurat.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/SEURAT/DOGMA/ > ./result/data_sc_multiomics/SEURAT/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_moGCN.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/MOGCN/DOGMA/ > ./result/data_sc_multiomics/MOGCN/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMHNN.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/SCMHNN/DOGMA/ > ./result/data_sc_multiomics/SCMHNN/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMDC.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/SCMDC/DOGMA/ > ./result/data_sc_multiomics/SCMDC/DOGMA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_matilda.py --data_dir ./data/data_sc_multiomics/DOGMA/ --output_dir ./result/data_sc_multiomics/MATILDA/DOGMA/ > ./result/data_sc_multiomics/MATILDA/DOGMA/output.log 2>&1 &
# wait

# # ==================== Dataset: TEA ====================
# mkdir -p ./result/data_sc_multiomics/AE/TEA/
# mkdir -p ./result/data_sc_multiomics/DAE/TEA/
# mkdir -p ./result/data_sc_multiomics/VAE/TEA/
# mkdir -p ./result/data_sc_multiomics/ZINBVAE/TEA/
# mkdir -p ./result/data_sc_multiomics/MOFA2/TEA/
# mkdir -p ./result/data_sc_multiomics/MOJITOO/TEA/
# mkdir -p ./result/data_sc_multiomics/SEURAT/TEA/
# mkdir -p ./result/data_sc_multiomics/MOGCN/TEA/
# mkdir -p ./result/data_sc_multiomics/SCMHNN/TEA/
# mkdir -p ./result/data_sc_multiomics/SCMDC/TEA/
# mkdir -p ./result/data_sc_multiomics/MATILDA/TEA/
# CUDA_VISIBLE_DEVICES=0 python reference_1_ae.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/AE/TEA/ > ./result/data_sc_multiomics/AE/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_dae.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/DAE/TEA/ > ./result/data_sc_multiomics/DAE/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_vae.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/VAE/TEA/ > ./result/data_sc_multiomics/VAE/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_zinbvae.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/ZINBVAE/TEA/ > ./result/data_sc_multiomics/ZINBVAE/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mofa2.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/MOFA2/TEA/ > ./result/data_sc_multiomics/MOFA2/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mojitoo.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/MOJITOO/TEA/ > ./result/data_sc_multiomics/MOJITOO/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_seurat.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/SEURAT/TEA/ > ./result/data_sc_multiomics/SEURAT/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_moGCN.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/MOGCN/TEA/ > ./result/data_sc_multiomics/MOGCN/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMHNN.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/SCMHNN/TEA/ > ./result/data_sc_multiomics/SCMHNN/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMDC.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/SCMDC/TEA/ > ./result/data_sc_multiomics/SCMDC/TEA/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_matilda.py --data_dir ./data/data_sc_multiomics/TEA/ --output_dir ./result/data_sc_multiomics/MATILDA/TEA/ > ./result/data_sc_multiomics/MATILDA/TEA/output.log 2>&1 &
# wait

# # ==================== Dataset: NEAT ====================
# mkdir -p ./result/data_sc_multiomics/AE/NEAT/
# mkdir -p ./result/data_sc_multiomics/DAE/NEAT/
# mkdir -p ./result/data_sc_multiomics/VAE/NEAT/
# mkdir -p ./result/data_sc_multiomics/ZINBVAE/NEAT/
# mkdir -p ./result/data_sc_multiomics/MOFA2/NEAT/
# mkdir -p ./result/data_sc_multiomics/MOJITOO/NEAT/
# mkdir -p ./result/data_sc_multiomics/SEURAT/NEAT/
# mkdir -p ./result/data_sc_multiomics/MOGCN/NEAT/
# mkdir -p ./result/data_sc_multiomics/SCMHNN/NEAT/
# mkdir -p ./result/data_sc_multiomics/SCMDC/NEAT/
# mkdir -p ./result/data_sc_multiomics/MATILDA/NEAT/
# CUDA_VISIBLE_DEVICES=0 python reference_1_ae.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/AE/NEAT/ > ./result/data_sc_multiomics/AE/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_dae.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/DAE/NEAT/ > ./result/data_sc_multiomics/DAE/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_vae.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/VAE/NEAT/ > ./result/data_sc_multiomics/VAE/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_1_zinbvae.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/ZINBVAE/NEAT/ > ./result/data_sc_multiomics/ZINBVAE/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mofa2.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/MOFA2/NEAT/ > ./result/data_sc_multiomics/MOFA2/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_mojitoo.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/MOJITOO/NEAT/ > ./result/data_sc_multiomics/MOJITOO/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_2_seurat.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/SEURAT/NEAT/ > ./result/data_sc_multiomics/SEURAT/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_moGCN.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/MOGCN/NEAT/ > ./result/data_sc_multiomics/MOGCN/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMHNN.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/SCMHNN/NEAT/ > ./result/data_sc_multiomics/SCMHNN/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_scMDC.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/SCMDC/NEAT/ > ./result/data_sc_multiomics/SCMDC/NEAT/output.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 python reference_3_matilda.py --data_dir ./data/data_sc_multiomics/NEAT/ --output_dir ./result/data_sc_multiomics/MATILDA/NEAT/ > ./result/data_sc_multiomics/MATILDA/NEAT/output.log 2>&1 &
# wait