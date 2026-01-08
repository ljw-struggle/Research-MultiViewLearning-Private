import os, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import scale # standardize the data, equivalent to (x - mean(x)) / std(x)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from sklearn.metrics.cluster import contingency_matrix, rand_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy import sparse

class MMDataset(Dataset):
    def __init__(self, data_dir, concat_data=False):
        # Load the multi-omics data and concatenate the modality-specific features.
        self.data_dir = data_dir  # Save data_dir as instance variable
        self.data_list = []
        modality_list = ['modality_mrna', 'modality_meth', 'modality_mirna'] if data_dir.find('bulk') != -1 else ['modality_rna', 'modality_protein', 'modality_atac']     
        for modality in modality_list:
            modality_data = pd.read_csv(os.path.join(data_dir, modality + '.csv'), header=0, index_col=0) # shape: (num_samples, num_features)
            modality_data_min = np.min(modality_data.values, axis=0, keepdims=True) # shape: (1, num_features)
            modality_data_max = np.max(modality_data.values, axis=0, keepdims=True) # shape: (1, num_features)
            modality_data_values = (modality_data.values - modality_data_min)/(modality_data_max - modality_data_min + 1e-10) # shape: (num_samples, num_features), normalize the data to [0, 1]
            self.data_list.append(modality_data_values.astype(float))
            print('{} shape: {}'.format(modality, modality_data_values.shape))
        label = modality_data.index.astype(int) # shape: (num_samples, )
        self.categories = np.unique(label).shape[0]; self.data_samples = self.data_list[0].shape[0]; # number of categories, number of samples
        self.data_views = len(self.data_list); self.data_features = [self.data_list[v].shape[1] for v in range(self.data_views)] # number of categories, number of views, number of samples, number of features in each view
        self.concat_data = concat_data
        if self.concat_data:
            self.X = [torch.from_numpy(x).float() for x in self.data_list]; self.Y = torch.tensor(label, dtype=torch.long)
            self.X = torch.cat(self.X, dim=1) # concatenate the data from different views, shape: (num_samples, sum(num_features))
        else:
            self.X = [torch.from_numpy(x).float() for x in self.data_list]; self.Y = torch.tensor(label, dtype=torch.long)

    def __getitem__(self, index):
        if self.concat_data:
            x = self.X[index] # select the data from the index
            y = self.Y[index] # select the label from the index
        else:
            x = [x[index] for x in self.X] # convert to tensor
            y = self.Y[index] # select the label from the index
        return x, y, index

    def __len__(self):
        return len(self.Y)
    
    def get_data_info(self):
        return self.data_views, self.data_samples, self.data_features, self.categories
    
    def get_label_to_name(self):
        if self.data_dir == './data/data_sc_multiomics/TEA/':
            return {0: 'B.Activated', 1: 'B.Naive', 2: 'DC.Myeloid', 3: 'Mono.CD14', 4: 'Mono.CD16', 5: 'NK', 6: 'Platelets', 7: 'T.CD4.Memory', 8: 'T.CD4.Naive', 9: 'T.CD8.Effector', 10: 'T.CD8.Naive', 11: 'T.DoubleNegative'}
        if self.data_dir == './data/data_sc_multiomics/NEAT/':
            return {0: 'C1', 1: 'C2', 2: 'C3', 3: 'C4', 4: 'C5', 5: 'C6', 6: 'C7'}
        if self.data_dir == './data/data_sc_multiomics/DOGMA/':
            return {0: 'ASDC', 1: 'B intermediate', 2: 'B memory', 3: 'B naive', 4: 'CD14 Mono', 5: 'CD16 Mono', 6: 'CD4 CTL', 7: 'CD4 Naive', 8: 'CD4 Proliferating', 9: 'CD4 TCM', 10: 'CD4 TEM', 11: 'CD8 Naive', 12: 'CD8 TCM', 13: 'CD8 TEM', 14: 'Eryth', 15: 'HSPC', 16: 'ILC', 17: 'MAIT', 18: 'NK', 19: 'NK_CD56bright', 20: 'Plasmablast', 21: 'Platelet', 22: 'Treg', 23: 'cDC2', 24: 'dnT', 25: 'gdT', 26: 'pDC'}
        if self.data_dir == './data/data_bulk_multiomics/BRCA/':
            return {0: 'Normal-like', 1: 'Basal-like', 2: 'HER2-enriched', 3: 'Luminal A', 4: 'Luminal B'}
        if self.data_dir == './data/data_bulk_multiomics/LGG/':
            return {0: 'grade II', 1: 'grade III'}
        if self.data_dir == './data/data_bulk_multiomics/KIPAN/':
            return {0: 'KICH', 1: 'KIRC', 2: 'KIRP'}
    
    def get_hyperedge_matrix(self, k_neighbors=30):
        # Construct the multi-omics hypergraph incidence matrix using k-nearest neighbors to avoid O(N²) complexity, avoid memory issues for large datasets.
        num_samples = self.data_list[0].shape[0]
        print(f'Using k-NN (k={k_neighbors}) for large dataset with {num_samples} samples')
        hyperedge_mat_list = []
        for i, data in enumerate(self.data_list):
            KNN = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto', metric='euclidean')
            KNN.fit(data)
            distances, indices = KNN.kneighbors(data) # distances: (num_samples, k_neighbors), indices: (num_samples, k_neighbors)
            distance_avg = distances.mean(axis=1, keepdims=True) # shape: (num_samples, 1)
            row_indices = np.repeat(np.arange(num_samples), k_neighbors) # shape: (num_hyperedges * k_neighbors, ); for example: [0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
            col_indices = indices.flatten() # shape: (num_hyperedges * k_neighbors, ); for example: [1, 2, 3, 0, 2, 3, 0, 1, 2, ...]
            weights = np.exp(-distances.flatten() ** 2 / ((1 * distance_avg[row_indices]) ** 2).flatten()) # gaussian kernel for computing the hyperedge weight
            hyperedge_mat = sparse.csr_matrix((weights, (row_indices, col_indices)), shape=(num_samples, num_samples)) # shape: (num_hyperedges, num_samples)
            hyperedge_mat_list.append(hyperedge_mat.T) # shape: (num_samples, num_hyperedges), transpose operation make the csr matrix to csc matrix
            print(f'Modality {i}: sparse matrix with {hyperedge_mat.nnz} non-zero elements') # number of non-zero elements
        hyperedge_mat = sparse.hstack(hyperedge_mat_list, format='csc') # shape: (num_samples, num_hyperedges * num_views)
        return hyperedge_mat

    def get_H_WE_HT(self, hyperedge_mat, dynamic_weight=True):
        # Calculate G from hyperedge incidence matrix H, where G = invDV2 * H * W_e * invDE * HT * invDV2
        assert sparse.issparse(hyperedge_mat), 'hyperedge_mat must be a sparse matrix'
        H = hyperedge_mat  # shape: (num_samples, num_hyperedges)
        DV = np.array(H.sum(axis=1)).squeeze()  # node degree
        DE = np.array(H.sum(axis=0)).squeeze()  # hyperedge degree
        DV = np.maximum(DV, 1e-10)
        DE = np.maximum(DE, 1e-10)
        invDV2 = sparse.diags(np.power(DV, -0.5), format='csc')
        invDE = sparse.diags(np.power(DE, -1), format='csc')
        return (invDV2 * H, None, invDE * H.T * invDV2) if dynamic_weight else (invDV2 * H * invDE * H.T * invDV2)

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] # accuracy


def purity_score(y_true, y_pred):
    contingency_matrix_result = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result) 
