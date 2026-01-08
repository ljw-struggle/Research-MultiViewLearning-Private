import os, snf, argparse
import pandas as pd, numpy as np
import seaborn as sns
from matplotlib.patches import Patch

def prepare_data(data_dir):
    data_train_list = []
    data_test_list = []
    for i in range(1, 4): # num_view = 3
        data_train = np.loadtxt(os.path.join(data_dir, str(i) + "_tr.csv"), delimiter=',')
        data_test = np.loadtxt(os.path.join(data_dir, str(i) + "_te.csv"), delimiter=',')
        data_train_min = np.min(data_train, axis=0, keepdims=True)
        data_train_max = np.max(data_train, axis=0, keepdims=True)
        data_train = (data_train - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_test = (data_test - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_train_list.append(data_train.astype(float))
        data_test_list.append(data_test.astype(float))
    label_train = np.loadtxt(os.path.join(data_dir, "labels_tr.csv"), delimiter=',').astype(int)
    label_test = np.loadtxt(os.path.join(data_dir, "labels_te.csv"), delimiter=',').astype(int)
    return data_train_list, data_test_list, label_train, label_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/BRCA/', help='The output dir.')
    # parser.add_argument('-p', '--path', type=str, nargs=3, default=['./data/BRCA/1_featname.csv', './data/BRCA/2_featname.csv', './data/BRCA/3_featname.csv'], help='Location of input files, must be 3 files')
    parser.add_argument('-m', '--metric', type=str, choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'], default='sqeuclidean', help='Distance metric to compute. Must be one of available metrics in :py:func scipy.spatial.distance.pdist.')
    parser.add_argument('-k', '--K', type=int, default=20, help='(0, N) int, number of neighbors to consider when creating affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 20.')
    parser.add_argument('-mu', '--mu', type=int, default=0.5, help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 0.5.')
    args = parser.parse_args()
    print('Load data files...')
    data_train_list, data_test_list, label_train, label_test = prepare_data(args.data_dir)
    data_list = [np.concatenate([data_train_list[i], data_test_list[i]], axis=0) for i in range(3)]
    label_list = [np.concatenate([label_train, label_test], axis=0)]
    print('Start similarity network fusion...')
    affinity_nets = snf.make_affinity([data_list[i] for i in range(3)], metric=args.metric, K=args.K, mu=args.mu)
    fused_net =snf.snf(affinity_nets, K=args.K) # (n_samples, n_samples)
    print('Save fused adjacency matrix...')
    fused_df = pd.DataFrame(fused_net)
    fused_df.to_csv(os.path.join(args.data_dir, 'SNF_fused_matrix.csv'), header=True, index=True)
    np.fill_diagonal(fused_df.values, 0)
    # sort by label
    label_color_list = ['orange', 'purple', 'pink', 'yellow', 'cyan', 'magenta', 'black']
    fig = sns.clustermap(fused_df.iloc[:, :], cmap='vlag', figsize=(8,8), row_colors=[label_color_list[i] for i in label_list[0]], col_colors=[label_color_list[i] for i in label_list[0]])
    # add legend
    legend_elements = [Patch(facecolor=label_color_list[0], edgecolor='black', label='Normal-like'),
                    Patch(facecolor=label_color_list[1], edgecolor='black', label='Basal-like'),
                    Patch(facecolor=label_color_list[2], edgecolor='black', label='HER2-enriched'),
                    Patch(facecolor=label_color_list[3], edgecolor='black', label='Luminal A'),
                    Patch(facecolor=label_color_list[4], edgecolor='black', label='Luminal B')]
    fig.ax_row_dendrogram.legend(handles=legend_elements, title="Classes", loc='center', frameon=False, edgecolor='black', markerscale=10.5, fontsize=10, ncol=1, columnspacing=3.5, bbox_to_anchor=(-1, 1))
    # fig.ax_col_dendrogram.legend(handles=legend_elements, title="Classes", loc='center', frameon=False, edgecolor='black', markerscale=10.5, fontsize=10, ncol=1, columnspacing=3.5)
    fig.savefig(os.path.join(args.data_dir, 'SNF_fused_clustermap.png'), dpi=300)
    print('Success! Results can be seen in result file')
    