import os, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import scale # standardize the data, equivalent to (x - mean(x)) / std(x)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from sklearn.metrics.cluster import contingency_matrix, rand_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

class MMDataset(Dataset):
    def __init__(self, data_dir, concat_data=False):
        # Load the multi-omics data and concatenate the modality-specific features.
        self.data_dir = data_dir  # Save data_dir as instance variable
        data_list = []
        modality_list = ['modality_mrna', 'modality_meth', 'modality_mirna'] if data_dir.find('bulk') != -1 else ['modality_rna', 'modality_protein', 'modality_atac']     
        for modality in modality_list:
            modality_data = pd.read_csv(os.path.join(data_dir, modality + '.csv'), header=0, index_col=0) # shape: (num_samples, num_features)
            modality_data_min = np.min(modality_data.values, axis=0, keepdims=True) # shape: (1, num_features)
            modality_data_max = np.max(modality_data.values, axis=0, keepdims=True) # shape: (1, num_features)
            modality_data_values = (modality_data.values - modality_data_min)/(modality_data_max - modality_data_min + 1e-10) # shape: (num_samples, num_features), normalize the data to [0, 1]
            data_list.append(modality_data_values.astype(float))
            print('{} shape: {}'.format(modality, modality_data_values.shape))
        label = modality_data.index.astype(int) # shape: (num_samples, )
        self.categories = np.unique(label).shape[0]; self.data_samples = data_list[0].shape[0]; # number of categories, number of samples
        self.data_views = len(data_list); self.data_features = [data_list[v].shape[1] for v in range(self.data_views)] # number of categories, number of views, number of samples, number of features in each view
        self.concat_data = concat_data
        if self.concat_data:
            self.X = [torch.from_numpy(x).float() for x in data_list]; self.Y = torch.tensor(label, dtype=torch.long)
            self.X = torch.cat(self.X, dim=1) # concatenate the data from different views, shape: (num_samples, sum(num_features))
        else:
            self.X = [torch.from_numpy(x).float() for x in data_list]; self.Y = torch.tensor(label, dtype=torch.long)

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

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] # accuracy

def purity_score(y_true, y_pred):
    contingency_matrix_result = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result) 

def overall_performance_report(multi_times_embedding_list, multi_times_label_pred_list, label, result_dir):
    times = len(multi_times_embedding_list)
    clustering_cacc = np.zeros([times]); clustering_nmi = np.zeros([times]); clustering_ri = np.zeros([times]); clustering_ari = np.zeros([times]); clustering_asw = np.zeros([times]); clustering_purity = np.zeros([times])
    for t, H in enumerate(multi_times_embedding_list):
        label_pred = KMeans(n_clusters=len(np.unique(label)), n_init=20, random_state=0).fit_predict(H) if multi_times_label_pred_list is None else multi_times_label_pred_list[t]
        clustering_cacc[t] = clustering_acc(label, label_pred) * 100
        clustering_nmi[t] = normalized_mutual_info_score(label, label_pred) * 100
        clustering_ri[t] = rand_score(label, label_pred) * 100
        clustering_ari[t] = adjusted_rand_score(label, label_pred) * 100
        clustering_asw[t] = silhouette_score(H, label_pred) * 100 if np.unique(label_pred).shape[0] > 1 else 0
        clustering_purity[t] = purity_score(label, label_pred) * 100
    clustering_result_list = pd.DataFrame({'Times': np.arange(times), 'Clustering ACC': clustering_cacc, 'Clustering NMI': clustering_nmi, 'Clustering RI': clustering_ri, 'Clustering ARI': clustering_ari, 'Clustering ASW': clustering_asw, 'Clustering Purity': clustering_purity})
    clustering_result_list.set_index('Times', inplace=True)
    clustering_result_list.to_csv(os.path.join(result_dir, 'clustering_result.csv'), index=True)
    mean_row_df = pd.DataFrame({'statistics': 'mean', 'Clustering ACC': clustering_cacc.mean(), 'Clustering NMI': clustering_nmi.mean(), 'Clustering RI': clustering_ri.mean(), 'Clustering ARI': clustering_ari.mean(), 'Clustering ASW': clustering_asw.mean(), 'Clustering Purity': clustering_purity.mean()}, index=[0])
    var_row_df = pd.DataFrame({'statistics': 'var', 'Clustering ACC': clustering_cacc.std(ddof=0), 'Clustering NMI': clustering_nmi.std(ddof=0), 'Clustering RI': clustering_ri.std(ddof=0), 'Clustering ARI': clustering_ari.std(ddof=0), 'Clustering ASW': clustering_asw.std(ddof=0), 'Clustering Purity': clustering_purity.std(ddof=0)}, index=[0])
    clustering_result_list_statistics = pd.concat([mean_row_df, var_row_df], ignore_index=True)
    clustering_result_list_statistics.to_csv(os.path.join(result_dir, 'clustering_result_statistics.csv'), index=True)
    
# import umap
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.neighbors import KNeighborsClassifier

# ### Visualization
# def tsne(H, Y):
#     embedding = TSNE(n_components=2, random_state=0).fit_transform(H)
#     plt.figure(figsize=(10, 10), dpi=80)
#     plt.scatter(embedding[:, 0], embedding[:, 1], c=Y.squeeze(), s=50, cmap=plt.cm.get_cmap('Paired', len(np.unique(Y))))
#     plt.colorbar(ticks=range(len(np.unique(Y)) + 1)); plt.axis('off'); plt.xticks([]); plt.yticks([]); plt.show()

# def umap(H, Y):
#     embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(H)
#     plt.figure(figsize=(10, 10), dpi=80)
#     plt.scatter(embedding[:, 0], embedding[:, 1], c=Y.squeeze(), s=50, cmap=plt.cm.get_cmap('Paired', len(np.unique(Y))))
#     plt.colorbar(ticks=range(len(np.unique(Y)) + 1)); plt.axis('off'); plt.xticks([]); plt.yticks([]); plt.show()

# def preprocess_data(data): # normalize total UMI counts to the mean of all rows, log2 normalization, z-score normalization
#     temp = torch.sum(data, 1, keepdim=True) # shape: (n,1), calculate the sum of each row
#     temp = temp/torch.mean(temp) # shape: (n,1), sum of each row / mean(sum of all rows)
#     data = data/temp # shape: (n, m), make the sum of each row to be the mean of all rows
#     data = torch.log2(data + 1) # shape: (n, m), log2 transformation
#     temp_mean = torch.mean(data, 1, keepdim=True) # shape: (n,1), calculate the mean of each row (every row is same)
#     temp_std = torch.std(data, 1, keepdim=True) # shape: (n,1), calculate the standard deviation of each row
#     data = (data - temp_mean)/temp_std # shape: (n, m), z-score normalization
#     return data
    
# def overall_performance_report(multi_times_embedding_list, multi_times_label_pred_list, label, result_dir):
#     times = len(multi_times_embedding_list)
#     # Clustering Performance Evaluation
#     clustering_cacc = np.zeros([times]); clustering_nmi = np.zeros([times]); clustering_ri = np.zeros([times]); clustering_ari = np.zeros([times]); clustering_asw = np.zeros([times]); clustering_purity = np.zeros([times])
#     for t, H in enumerate(multi_times_embedding_list):
#         label_pred = KMeans(n_clusters=len(np.unique(label)), n_init=20, random_state=0).fit_predict(H) if multi_times_label_pred_list is None else multi_times_label_pred_list[t]
#         clustering_cacc[t] = clustering_acc(label, label_pred) * 100
#         clustering_nmi[t] = normalized_mutual_info_score(label, label_pred) * 100
#         clustering_ri[t] = rand_score(label, label_pred) * 100
#         clustering_ari[t] = adjusted_rand_score(label, label_pred) * 100
#         clustering_asw[t] = silhouette_score(H, label_pred) * 100 if np.unique(label_pred).shape[0] > 1 else 0
#         clustering_purity[t] = purity_score(label, label_pred) * 100
#     clustering_result_list = pd.DataFrame({'Times': np.arange(times), 'Clustering ACC': clustering_cacc, 'Clustering NMI': clustering_nmi, 'Clustering RI': clustering_ri, 'Clustering ARI': clustering_ari, 'Clustering ASW': clustering_asw, 'Clustering Purity': clustering_purity})
#     clustering_result_list.set_index('Times', inplace=True)
#     clustering_result_list.to_csv(os.path.join(result_dir, 'clustering_result.csv'), index=True)
    
#     clustering_result_list_statistics = clustering_result_list.describe()
#     # Calculate the population variance of the clustering results. (ddof=0: population variance, ddof=1: sample variance)
#     # for pandas describe() function, the default is ddof=1, which is the sample variance.
#     # mean_row_df = pd.DataFrame({'statistics': 'mean', 'Clustering ACC': clustering_acc.mean(), 'Clustering NMI': clustering_nmi.mean(), 'Clustering RI': clustering_ri.mean(), 'Clustering ARI': clustering_ari.mean(), 'Clustering ASW': clustering_asw.mean(), 'Clustering Purity': clustering_purity.mean()}, index=[0])
#     # var_row_df = pd.DataFrame({'statistics': 'var', 'Clustering ACC': clustering_acc.std(ddof=0), 'Clustering NMI': clustering_nmi.std(ddof=0), 'Clustering RI': clustering_ri.std(ddof=0), 'Clustering ARI': clustering_ari.std(ddof=0), 'Clustering ASW': clustering_asw.std(ddof=0), 'Clustering Purity': clustering_purity.std(ddof=0)}, index=[0])
#     # metrics_statistics_df = pd.concat([mean_row_df, var_row_df], ignore_index=True)
#     population_std_df = pd.DataFrame({'Clustering ACC': clustering_cacc.std(ddof=0), 'Clustering NMI': clustering_nmi.std(ddof=0), 'Clustering RI': clustering_ri.std(ddof=0), 'Clustering ARI': clustering_ari.std(ddof=0), 'Clustering ASW': clustering_asw.std(ddof=0), 'Clustering Purity': clustering_purity.std(ddof=0)}, index=['p_std'])
#     clustering_result_list_statistics = pd.concat([clustering_result_list_statistics, population_std_df])
#     clustering_result_list_statistics.to_csv(os.path.join(result_dir, 'clustering_result_list_statistics.csv'), index=True)
#     print(f'Clustering ACC = {clustering_cacc.mean():.2f}±{clustering_cacc.std():.2f}\n' + 
#           f'Clustering NMI = {clustering_nmi.mean():.2f}±{clustering_nmi.std():.2f}\n' + 
#           f'Clustering RI = {clustering_ri.mean():.2f}±{clustering_ri.std():.2f}\n' + 
#           f'Clustering ARI = {clustering_ari.mean():.2f}±{clustering_ari.std():.2f}\n' + 
#           f'Clustering ASW = {clustering_asw.mean():.2f}±{clustering_asw.std():.2f}\n' + 
#           f'Clustering Purity = {clustering_purity.mean():.2f}±{clustering_purity.std():.2f}\n')

# def clustering(H, Y_pred, Y, n_clusters, count=1):
#     acc_array = np.zeros(count); nmi_array = np.zeros(count); ri_array = np.zeros(count); ari_array = np.zeros(count); asw_array = np.zeros(count); purity_array = np.zeros(count)
#     for i in range(count): # Randomly cluster the features multiple times and evaluate the clustering performance
#         if Y_pred is None:
#             Y_pred = KMeans(n_clusters=n_clusters, n_init=20, random_state=0).fit_predict(H) # KMeans is not a deterministic method and the results are random for different initializations.
#         acc_array[i] = clustering_acc(Y, Y_pred) * 100; 
#         nmi_array[i] = normalized_mutual_info_score(Y, Y_pred) * 100
#         ri_array[i] = rand_score(Y, Y_pred) * 100; 
#         ari_array[i] = adjusted_rand_score(Y, Y_pred) * 100
#         if np.unique(Y_pred).shape[0] > 1:
#             asw_array[i] = silhouette_score(H, Y_pred) * 100
#         else:
#             asw_array[i] = 0
#         purity_array[i] = purity_score(Y, Y_pred) * 100
#     return {'cacc_mean': acc_array.mean(), 'cacc_std': acc_array.std(), 
#             'nmi_mean': nmi_array.mean(), 'nmi_std': nmi_array.std(), 
#             'ri_mean': ri_array.mean(), 'ri_std': ri_array.std(), 
#             'ari_mean': ari_array.mean(), 'ari_std': ari_array.std(), 
#             'asw_mean': asw_array.mean(), 'asw_std': asw_array.std(), 
#             'purity_mean': purity_array.mean(), 'purity_std': purity_array.std()}
    
# def classification(H, Y, k=3, count=5, test_size=0.5):
#     acc_array = np.zeros(count); precision_array = np.zeros(count); recall_array = np.zeros(count); f1_macro_array = np.zeros(count); f1_micro_array = np.zeros(count); ap_array = np.zeros(count); auc_array = np.zeros(count)
#     for i in range(count): # Randomly split the data and train the classifier multiple times
#         train_data, test_data, train_label, test_label = train_test_split(H, Y, test_size=test_size, stratify=Y)
#         classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(train_data, train_label) # KNeighborsClassifier is a deterministic method.
#         test_pred = classifier.predict(test_data) # shape: (num_test, )
#         test_prob = classifier.predict_proba(test_data) # shape: (num_test, num_class)
#         num_class = test_prob.shape[1]
#         acc_array[i] = accuracy_score(test_label, test_pred)*100
#         precision_array[i] = precision_score(test_label, test_pred, average='macro', zero_division=0)*100
#         recall_array[i] = recall_score(test_label, test_pred, average='macro', zero_division=0)*100
#         f1_macro_array[i] = f1_score(test_label, test_pred, average='macro', zero_division=0)*100
#         f1_micro_array[i] = f1_score(test_label, test_pred, average='micro', zero_division=0)*100
#         ap_array[i] = average_precision_score(np.eye(num_class)[test_label], test_prob, average='macro')*100
#         auc_array[i] = roc_auc_score(np.eye(num_class)[test_label], test_prob, average='macro')*100
#     return {'acc_mean': acc_array.mean(), 'acc_std': acc_array.std(), 
#             'precision_mean': precision_array.mean(), 'precision_std': precision_array.std(), 
#             'recall_mean': recall_array.mean(), 'recall_std': recall_array.std(), 
#             'f1_macro_mean': f1_macro_array.mean(), 'f1_macro_std': f1_macro_array.std(), 
#             'f1_micro_mean': f1_micro_array.mean(), 'f1_micro_std': f1_micro_array.std(), 
#             'ap_mean': ap_array.mean(), 'ap_std': ap_array.std(), 
#             'auc_mean': auc_array.mean(), 'auc_std': auc_array.std()}

# def overall_performance_report(multi_times_embedding_list, multi_times_label_pred_list, label, result_dir):
#     times = len(multi_times_embedding_list)
    
#     # Clustering Performance Evaluation
#     clustering_acc = np.zeros([times]); clustering_nmi = np.zeros([times]); clustering_ri = np.zeros([times]); clustering_ari = np.zeros([times]); clustering_asw = np.zeros([times]); clustering_purity = np.zeros([times])
#     for t, H in enumerate(multi_times_embedding_list):
#         label_pred = None if multi_times_label_pred_list is None else multi_times_label_pred_list[t]
#         clustering_result = clustering(H, label_pred, label, len(np.unique(label)), count=5) # Evaluate using clustering
#         clustering_acc[t] = clustering_result["cacc_mean"]
#         clustering_nmi[t] = clustering_result["nmi_mean"]
#         clustering_ri[t] = clustering_result["ri_mean"]
#         clustering_ari[t] = clustering_result["ari_mean"]
#         clustering_asw[t] = clustering_result["asw_mean"]
#         clustering_purity[t] = clustering_result["purity_mean"]
#     clustering_result_list = pd.DataFrame({'Times': np.arange(times), 'Clustering ACC': clustering_acc, 'Clustering NMI': clustering_nmi, 'Clustering RI': clustering_ri, 'Clustering ARI': clustering_ari, 'Clustering ASW': clustering_asw, 'Clustering Purity': clustering_purity})
#     clustering_result_list.set_index('Times', inplace=True)
#     clustering_result_list.to_csv(os.path.join(result_dir, 'clustering_result.csv'), index=True)
#     clustering_result_list_statistics = clustering_result_list.describe()
#     # Calculate the population variance of the clustering results. (ddof=0: population variance, ddof=1: sample variance)
#     # for pandas describe() function, the default is ddof=1, which is the sample variance.
#     population_std_df = pd.DataFrame({'Clustering ACC': clustering_acc.std(ddof=0), 'Clustering NMI': clustering_nmi.std(ddof=0), 'Clustering RI': clustering_ri.std(ddof=0), 'Clustering ARI': clustering_ari.std(ddof=0), 'Clustering ASW': clustering_asw.std(ddof=0), 'Clustering Purity': clustering_purity.std(ddof=0)}, index=['p_std'])
#     clustering_result_list_statistics = pd.concat([clustering_result_list_statistics, population_std_df])
#     clustering_result_list_statistics.to_csv(os.path.join(result_dir, 'clustering_result_list_statistics.csv'), index=True)
#     print(f'Clustering ACC = {clustering_acc.mean():.2f}±{clustering_acc.std():.2f}\n' + 
#           f'Clustering NMI = {clustering_nmi.mean():.2f}±{clustering_nmi.std():.2f}\n' + 
#           f'Clustering RI = {clustering_ri.mean():.2f}±{clustering_ri.std():.2f}\n' + 
#           f'Clustering ARI = {clustering_ari.mean():.2f}±{clustering_ari.std():.2f}\n' + 
#           f'Clustering ASW = {clustering_asw.mean():.2f}±{clustering_asw.std():.2f}\n' + 
#           f'Clustering Purity = {clustering_purity.mean():.2f}±{clustering_purity.std():.2f}\n')
    
    # # Classification Performance Evaluation
    # # Test size = 0.9
    # classification_acc = np.zeros([times]); classification_precision = np.zeros([times]); classification_recall = np.zeros([times]); classification_f1_macro = np.zeros([times]); classification_f1_micro = np.zeros([times]); classification_ap = np.zeros([times]); classification_auc = np.zeros([times])
    # for t, H in enumerate(multi_times_embedding_list):
    #     classification_result = classification(H, label, count=5, test_size=0.9) # Evaluate using classification
    #     classification_acc[t] = classification_result["acc_mean"]
    #     classification_precision[t] = classification_result["precision_mean"]
    #     classification_recall[t] = classification_result["recall_mean"]
    #     classification_f1_macro[t] = classification_result["f1_macro_mean"]
    #     classification_f1_micro[t] = classification_result["f1_micro_mean"]
    #     classification_ap[t] = classification_result["ap_mean"]
    #     classification_auc[t] = classification_result["auc_mean"]
    # classification_result_list = pd.DataFrame({'Times': np.arange(times), 'Classification ACC': classification_acc, 'Classification Precision': classification_precision, 'Classification Recall': classification_recall, 'Classification F1-Macro': classification_f1_macro, 'Classification F1-Micro': classification_f1_micro, 'Classification AP': classification_ap, 'Classification AUC': classification_auc})
    # classification_result_list.set_index('Times', inplace=True)
    # classification_result_list.to_csv(os.path.join(result_dir, 'classification_result_0.1.csv'), index=True)
    # classification_result_list_statistics = classification_result_list.describe()
    # # Calculate the population variance of the classification results. (ddof=0: population variance, ddof=1: sample variance)
    # # for pandas describe() function, the default is ddof=1, which is the sample variance.
    # # for numpy std() function, the default is ddof=0, which is the population variance.
    # population_std_df = pd.DataFrame({'Classification ACC': classification_acc.std(ddof=0), 'Classification Precision': classification_precision.std(ddof=0), 'Classification Recall': classification_recall.std(ddof=0), 'Classification F1-Macro': classification_f1_macro.std(ddof=0), 'Classification F1-Micro': classification_f1_micro.std(ddof=0), 'Classification AP': classification_ap.std(ddof=0), 'Classification AUC': classification_auc.std(ddof=0)}, index=['p_std'])
    # classification_result_list_statistics = pd.concat([classification_result_list_statistics, population_std_df])
    # classification_result_list_statistics.to_csv(os.path.join(result_dir, 'classification_result_list_statistics_0.1.csv'), index=True)
    # print(f'Test size = 0.9: \n' + \
    #       f'Classification ACC = {classification_acc.mean():.2f}±{classification_acc.std():.2f}\n' + \
    #       f'Classification Precision = {classification_precision.mean():.2f}±{classification_precision.std():.2f}\n' + \
    #       f'Classification Recall = {classification_recall.mean():.2f}±{classification_recall.std():.2f}\n' + \
    #       f'Classification F1-Macro = {classification_f1_macro.mean():.2f}±{classification_f1_macro.std():.2f}\n' + \
    #       f'Classification F1-Micro = {classification_f1_micro.mean():.2f}±{classification_f1_micro.std():.2f}\n' + \
    #       f'Classification AP = {classification_ap.mean():.2f}±{classification_ap.std():.2f}\n' + \
    #       f'Classification AUC = {classification_auc.mean():.2f}±{classification_auc.std():.2f}\n')
    
    # # Test size = 0.8
    # classification_acc = np.zeros([times]); classification_precision = np.zeros([times]); classification_recall = np.zeros([times]); classification_f1_macro = np.zeros([times]); classification_f1_micro = np.zeros([times]); classification_ap = np.zeros([times]); classification_auc = np.zeros([times])
    # for t, H in enumerate(multi_times_embedding_list):
    #     classification_result = classification(H, label, count=5, test_size=0.8) # Evaluate using classification
    #     classification_acc[t] = classification_result["acc_mean"]
    #     classification_precision[t] = classification_result["precision_mean"]
    #     classification_recall[t] = classification_result["recall_mean"]
    #     classification_f1_macro[t] = classification_result["f1_macro_mean"]
    #     classification_f1_micro[t] = classification_result["f1_micro_mean"]
    #     classification_ap[t] = classification_result["ap_mean"]
    #     classification_auc[t] = classification_result["auc_mean"]
    # classification_result_list = pd.DataFrame({'Times': np.arange(times), 'Classification ACC': classification_acc, 'Classification Precision': classification_precision, 'Classification Recall': classification_recall, 'Classification F1-Macro': classification_f1_macro, 'Classification F1-Micro': classification_f1_micro, 'Classification AP': classification_ap, 'Classification AUC': classification_auc})
    # classification_result_list.set_index('Times', inplace=True)
    # classification_result_list.to_csv(os.path.join(result_dir, 'classification_result_0.2.csv'), index=True)
    # classification_result_list_statistics = classification_result_list.describe()
    # # Calculate the population variance of the classification results. (ddof=0: population variance, ddof=1: sample variance)
    # # for pandas describe() function, the default is ddof=1, which is the sample variance.
    # # for numpy std() function, the default is ddof=0, which is the population variance.
    # population_std_df = pd.DataFrame({'Classification ACC': classification_acc.std(ddof=0), 'Classification Precision': classification_precision.std(ddof=0), 'Classification Recall': classification_recall.std(ddof=0), 'Classification F1-Macro': classification_f1_macro.std(ddof=0), 'Classification F1-Micro': classification_f1_micro.std(ddof=0), 'Classification AP': classification_ap.std(ddof=0), 'Classification AUC': classification_auc.std(ddof=0)}, index=['p_std'])
    # classification_result_list_statistics = pd.concat([classification_result_list_statistics, population_std_df])
    # classification_result_list_statistics.to_csv(os.path.join(result_dir, 'classification_result_list_statistics_0.2.csv'), index=True)
    # print(f'Test size = 0.8: \n' + \
    #       f'Classification ACC = {classification_acc.mean():.2f}±{classification_acc.std():.2f}\n' + \
    #       f'Classification Precision = {classification_precision.mean():.2f}±{classification_precision.std():.2f}\n' + \
    #       f'Classification Recall = {classification_recall.mean():.2f}±{classification_recall.std():.2f}\n' + \
    #       f'Classification F1-Macro = {classification_f1_macro.mean():.2f}±{classification_f1_macro.std():.2f}\n' + \
    #       f'Classification F1-Micro = {classification_f1_micro.mean():.2f}±{classification_f1_micro.std():.2f}\n' + \
    #       f'Classification AP = {classification_ap.mean():.2f}±{classification_ap.std():.2f}\n' + \
    #       f'Classification AUC = {classification_auc.mean():.2f}±{classification_auc.std():.2f}\n')
    
    # # Test size = 0.7
    # classification_acc = np.zeros([times]); classification_precision = np.zeros([times]); classification_recall = np.zeros([times]); classification_f1_macro = np.zeros([times]); classification_f1_micro = np.zeros([times]); classification_ap = np.zeros([times]); classification_auc = np.zeros([times])
    # for t, H in enumerate(multi_times_embedding_list):
    #     classification_result = classification(H, label, count=5, test_size=0.7) # Evaluate using classification
    #     classification_acc[t] = classification_result["acc_mean"]
    #     classification_precision[t] = classification_result["precision_mean"]
    #     classification_recall[t] = classification_result["recall_mean"]
    #     classification_f1_macro[t] = classification_result["f1_macro_mean"]
    #     classification_f1_micro[t] = classification_result["f1_micro_mean"]
    #     classification_ap[t] = classification_result["ap_mean"]
    #     classification_auc[t] = classification_result["auc_mean"]
    # classification_result_list = pd.DataFrame({'Times': np.arange(times), 'Classification ACC': classification_acc, 'Classification Precision': classification_precision, 'Classification Recall': classification_recall, 'Classification F1-Macro': classification_f1_macro, 'Classification F1-Micro': classification_f1_micro, 'Classification AP': classification_ap, 'Classification AUC': classification_auc})
    # classification_result_list.set_index('Times', inplace=True)
    # classification_result_list.to_csv(os.path.join(result_dir, 'classification_result_0.3.csv'), index=True)
    # classification_result_list_statistics = classification_result_list.describe()
    # # Calculate the population variance of the classification results. (ddof=0: population variance, ddof=1: sample variance)
    # # for pandas describe() function, the default is ddof=1, which is the sample variance.
    # # for numpy std() function, the default is ddof=0, which is the population variance.
    # population_std_df = pd.DataFrame({'Classification ACC': classification_acc.std(ddof=0), 'Classification Precision': classification_precision.std(ddof=0), 'Classification Recall': classification_recall.std(ddof=0), 'Classification F1-Macro': classification_f1_macro.std(ddof=0), 'Classification F1-Micro': classification_f1_micro.std(ddof=0), 'Classification AP': classification_ap.std(ddof=0), 'Classification AUC': classification_auc.std(ddof=0)}, index=['p_std'])
    # classification_result_list_statistics = pd.concat([classification_result_list_statistics, population_std_df])
    # classification_result_list_statistics.to_csv(os.path.join(result_dir, 'classification_result_list_statistics_0.3.csv'), index=True)
    # print(f'Test size = 0.7: \n' + \
    #       f'Classification ACC = {classification_acc.mean():.2f}±{classification_acc.std():.2f}\n' + \
    #       f'Classification Precision = {classification_precision.mean():.2f}±{classification_precision.std():.2f}\n' + \
    #       f'Classification Recall = {classification_recall.mean():.2f}±{classification_recall.std():.2f}\n' + \
    #       f'Classification F1-Macro = {classification_f1_macro.mean():.2f}±{classification_f1_macro.std():.2f}\n' + \
    #       f'Classification F1-Micro = {classification_f1_micro.mean():.2f}±{classification_f1_micro.std():.2f}\n' + \
    #       f'Classification AP = {classification_ap.mean():.2f}±{classification_ap.std():.2f}\n' + \
    #       f'Classification AUC = {classification_auc.mean():.2f}±{classification_auc.std():.2f}\n')

    # # Test size = 0.6
    # classification_acc = np.zeros([times]); classification_precision = np.zeros([times]); classification_recall = np.zeros([times]); classification_f1_macro = np.zeros([times]); classification_f1_micro = np.zeros([times]); classification_ap = np.zeros([times]); classification_auc = np.zeros([times])
    # for t, H in enumerate(multi_times_embedding_list):
    #     classification_result = classification(H, label, count=5, test_size=0.6) # Evaluate using classification
    #     classification_acc[t] = classification_result["acc_mean"]
    #     classification_precision[t] = classification_result["precision_mean"]
    #     classification_recall[t] = classification_result["recall_mean"]
    #     classification_f1_macro[t] = classification_result["f1_macro_mean"]
    #     classification_f1_micro[t] = classification_result["f1_micro_mean"]
    #     classification_ap[t] = classification_result["ap_mean"]
    #     classification_auc[t] = classification_result["auc_mean"]
    # classification_result_list = pd.DataFrame({'Times': np.arange(times), 'Classification ACC': classification_acc, 'Classification Precision': classification_precision, 'Classification Recall': classification_recall, 'Classification F1-Macro': classification_f1_macro, 'Classification F1-Micro': classification_f1_micro, 'Classification AP': classification_ap, 'Classification AUC': classification_auc})
    # classification_result_list.set_index('Times', inplace=True)
    # classification_result_list.to_csv(os.path.join(result_dir, 'classification_result_0.4.csv'), index=True)
    # classification_result_list_statistics = classification_result_list.describe()
    # # Calculate the population variance of the classification results. (ddof=0: population variance, ddof=1: sample variance)
    # # for pandas describe() function, the default is ddof=1, which is the sample variance.
    # # for numpy std() function, the default is ddof=0, which is the population variance.
    # population_std_df = pd.DataFrame({'Classification ACC': classification_acc.std(ddof=0), 'Classification Precision': classification_precision.std(ddof=0), 'Classification Recall': classification_recall.std(ddof=0), 'Classification F1-Macro': classification_f1_macro.std(ddof=0), 'Classification F1-Micro': classification_f1_micro.std(ddof=0), 'Classification AP': classification_ap.std(ddof=0), 'Classification AUC': classification_auc.std(ddof=0)}, index=['p_std'])
    # classification_result_list_statistics = pd.concat([classification_result_list_statistics, population_std_df])
    # classification_result_list_statistics.to_csv(os.path.join(result_dir, 'classification_result_list_statistics_0.4.csv'), index=True)
    # print(f'Test size = 0.6: \n' + \
    #       f'Classification ACC = {classification_acc.mean():.2f}±{classification_acc.std():.2f}\n' + \
    #       f'Classification Precision = {classification_precision.mean():.2f}±{classification_precision.std():.2f}\n' + \
    #       f'Classification Recall = {classification_recall.mean():.2f}±{classification_recall.std():.2f}\n' + \
    #       f'Classification F1-Macro = {classification_f1_macro.mean():.2f}±{classification_f1_macro.std():.2f}\n' + \
    #       f'Classification F1-Micro = {classification_f1_micro.mean():.2f}±{classification_f1_micro.std():.2f}\n' + \
    #       f'Classification AP = {classification_ap.mean():.2f}±{classification_ap.std():.2f}\n' + \
    #       f'Classification AUC = {classification_auc.mean():.2f}±{classification_auc.std():.2f}\n')    
