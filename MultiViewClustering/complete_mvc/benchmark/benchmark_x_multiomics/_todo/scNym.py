# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scnym
import time
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import random
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 
# SEED = 2021
# data = sc.read_h5ad('/aaa/jianhuayao/project2/data/Zheng68k/Zheng68k_full.h5ad')
# sc.pp.filter_cells(data, min_genes=200)
# sc.pp.normalize_total(data, target_sum=1e6)
# sc.pp.log1p(data, base=2)
# load raw data
# dataset = 'tea'
dataset = 'neat'
# dataset = 'dogma'
labeled_cell_ratio = 0.1
if dataset == 'tea':
    data_dir_rna = './datasets/MOJITOO_h5ad/PBMC-TEA_rna.h5ad'
    data_dir_adt = './datasets/MOJITOO_h5ad/PBMC-TEA_adt.h5ad'
    data_dir_atac = './datasets/MOJITOO_h5ad/PBMC-TEA_atac.h5ad'
    label_dict = {'T.DoubleNegative':0, 'Platelets':1, 'Mono.CD14':2, 'DC.Myeloid':3, 'B.Naive':4, 'T.CD4.Naive':5, 'B.Activated':6, 
                  'T.CD8.Effector':7, 'T.CD8.Naive':8, 'NK':9, 'Mono.CD16':10, 'T.CD4.Memory':11}
elif dataset == 'dogma':
    data_dir_rna = './datasets/MOJITOO_h5ad/PBMC-DOGMA_rna.h5ad'
    data_dir_adt = './datasets/MOJITOO_h5ad/PBMC-DOGMA_adt.h5ad'
    data_dir_atac = './datasets/MOJITOO_h5ad/PBMC-DOGMA_atac.h5ad'
    label_dict={'pDC':0, 'CD8 Naive':1, 'Eryth':2, 'Treg':3, 'B memory':4, 'CD8 TCM':5, 'B naive':6, 'NK':7, 'Plasmablast':8, 'CD4 CTL':9, 
                'CD14 Mono':10, 'gdT':11, 'Platelet':12, 'CD4 Naive':13, 'NK_CD56bright':14, 'CD8 TEM':15, 'ASDC':16, 'B intermediate':17, 
                'MAIT':18, 'CD16 Mono':19, 'ILC':20, 'CD4 Proliferating':21, 'CD4 TCM':22, 'cDC2':23, 'CD4 TEM':24, 'dnT':25, 'HSPC':26}
elif dataset == 'neat':
    label_dict = {'C1':0,'C2':1,'C3':2,'C4':3,'C5':4,'C6':5,'C7':6}

'''
def load_tri_modal_data(data_dir_rna,data_dir_adt,data_dir_atac):
    data_rna = sc.read_h5ad(data_dir_rna)
    # print(data_rna)
    # print('----------------')
    fts_rna = data_rna.obsm['X_rpca']
    # lbls = np.array(itemgetter(*list(data_rna.obs['celltype']))(label_dict))
    lbls = list(data_rna.obs['celltype'])
    data_adt = sc.read_h5ad(data_dir_adt)
    fts_adt = data_adt.X
    data_atac = sc.read_h5ad(data_dir_atac)
    fts_atac = data_atac.obsm['X_lsi']
    print(fts_rna.shape,fts_adt.shape,fts_atac.shape)
    # features = None
    fts = np.concatenate([fts_rna, fts_adt,fts_atac],axis=1)
    print(fts.shape)
    return fts,lbls
fts,lbls = load_tri_modal_data(data_dir_rna,data_dir_adt,data_dir_atac)
np.save('./datasets/raw_data/{}_raw_tri_modal_data.npy'.format(dataset),fts)
df1 = pd.DataFrame(data=lbls,columns=['celltype'])
df1.to_csv('./datasets/raw_data/{}_raw_data_lbls.csv'.format(dataset),index=False)
'''

if dataset == 'neat':
    fts = np.load('./datasets/neat-seq/processed_data/rna_3000hvg_pca.npy')
    lbls_ = pd.read_csv('./datasets/neat-seq/labels.csv')
    lbls = list(lbls_['labels'])
else:
    # fts = np.load('./datasets/raw_data/{}_raw_tri_modal_data.npy'.format(dataset))
    fts = np.load('./Matilda/Matilda-main/data/my_{}_data/fts_rna.npy'.format(dataset))
    # fts = np.load('./result_{}/unsupervised_embedding/x_ach.npy'.format(dataset))
    lbls_ = pd.read_csv('./datasets/raw_data/{}_raw_data_lbls.csv'.format(dataset))
    lbls = list(lbls_['celltype'])
# fts = fts[:,50:260]
print(fts.shape)
# print(fts)
fts = scaler.fit_transform(fts)
# 将data转换为scanpy的anndata形式
embedding_name = []
cell_name = []
for i in range(fts.shape[1]):
    embedding_name.append(str(i)+'_')
# print(embedding_name)
for i in range(fts.shape[0]):
    cell_name.append(str(i)+'_')
data=ad.AnnData(fts,obs=cell_name,var=embedding_name)
data = ad.AnnData(fts,dict(obs_names=cell_name),dict(var_names=embedding_name))
data.obs['celltype'] = lbls
# print(data.X)
# # print(data.var_names)
# sc.pp.normalize_total(data, target_sum=1e6)
# sc.pp.log1p(data, base=2)
# print(data.X)
# sc.pp.pca(data)
# fts_pd = pd.DataFrame(data.X)
# fts_pd.fillna(1.)
# fts = fts_pd.values
# print(fts)
# print('---------------')
# data = ad.AnnData(fts,dict(obs_names=cell_name),dict(var_names=embedding_name))
# data.obs['celltype'] = lbls
acc = []
f1 = []
f1w = []
for rep in range(1):
    # SEED += 1
    num_cell = fts.shape[0]
    indices = list(range(fts.shape[0]))
    random.shuffle(indices)
    index_train = indices[:int(num_cell*labeled_cell_ratio)]
    index_val = indices[int(num_cell*labeled_cell_ratio):]
    data_train = data[index_train]
    data_val = data[index_val]
    # print(data_train.X)
    # print(data_train.var_names)
    # print(data_val.var_names)
    # data.obs['celltype'][index_val] = 'Unlabeled'
    # print(data.obs['celltype'])
    # print('--------------')
    data_val.obs["true_celltype"] = data_val.obs["celltype"]
    data_val.obs["celltype"] = "Unlabeled"
    # print(data_train.obs['celltype'])
    # print(data_val.obs['celltype'])
    # data_train.obs['annotations'] = np.array(data_train.obs["celltype"])
    # data_val.obs['annotations'] = 'Unlabeled'
    adata = data_train.concatenate(data_val)
    ## train
    scnym.api.scnym_api(adata=adata, task="train", groupby="celltype", config="no_new_identity", out_path="./scnym_outputs")
    ## predict
    scnym.api.scnym_api(adata=adata, task='predict', trained_model='./scnym_outputs')
    # print('result:::::')
    # print(data_val.obs['true_celltype'])
    # print('-------------------')
    # print(adata.obs['scNym'])
    data_val.obs['scNym'] = np.array(adata.obs.loc[[x + '-1' for x in data_val.obs_names], 'scNym'])
    # print(list(data_val.obs['scNym']))
    y_true = data_val.obs['true_celltype']
    y_pred = data_val.obs['scNym']
    np.save('./all_methods_annotation/{}/scnym/lbls_test_scnym_{}'.format(dataset,labeled_cell_ratio),y_true)
    np.save('./all_methods_annotation/{}/scnym/preds_test_scnym_{}'.format(dataset,labeled_cell_ratio),y_pred)
    acc.append(accuracy_score(y_true, y_pred))
    f1.append(f1_score(y_true, y_pred, average='macro'))
    f1w.append(f1_score(y_true, y_pred, average='weighted'))
    print('acc:',acc)
    print('f1w:',f1w)
    # print(confusion_matrix(y_true, y_pred))
    # print(classification_report(y_true, y_pred, digits=4))
    # print(acc,f1,f1w)
# print(acc)
# print(f1)
# print(f1w)
# f = open("./result_{}/supervised_result_scnym.txt".format(dataset),"a")
# print('\n',file=f)
# print(labeled_cell_ratio,file=f)
# print('\n',file=f)
# print(acc,file=f)
# print(f1,file=f)
# print(f1w,file=f)
