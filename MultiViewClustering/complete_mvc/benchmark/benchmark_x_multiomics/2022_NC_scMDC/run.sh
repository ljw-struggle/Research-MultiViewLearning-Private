# Run DE analysis (LRP) based on the well-trained scMDC model and its results.
f=./data/CITESeq_GSE128639_BMNC_anno.h5
python -u main.py --n_clusters 27 --pretrain_weight_file ./result/BMNC/pretrain_checkpoint_400.pth --data_file $f --save_dir ./result/BMNC --filter_1
python -u main_lrp.py --n_clusters 27 --pretrain_weight_file ./result/BMNC/pretrain_checkpoint_400.pth \ 
--cluster_index_file ./result/BMNC/1_pred.csv --data_file $f --save_dir ./result/BMNC --filter_1 

# Here are the commands for the real data experiments. We test ten times on each dataset.
f=./data/10XMultiomics_pbmc_10k_granulocyte_plus.h5
echo "Run SMAGE-seq PBMC10K"
python -u main.py --n_clusters 12 --pretrain_weight_file ./result/PBMC/pretrain_checkpoint_400.pth --data_file $f --save_dir ./result/PBMC \ 
--filter_1 --filter_2 --filter_1_num 2000 --filter_2_num 2000 -el 256 128 64 -dl1 64 128 256 -dl2 64 128 256 -phi_1 0.005 -phi_2 0.005 -sigma_2 2.5 -tau 0.1

f=./data/CITESeq_realdata_spleen_lymph_111_anno.h5
echo "Run multi-batch CITE-seq SLN111"
python -u main_batch.py --n_clusters 35 --pretrain_weight_file ./result/SLN111/pretrain_checkpoint_400.pth --data_file $f --save_dir ./result/SLN111 --filter_1
