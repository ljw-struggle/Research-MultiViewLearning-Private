import torch
import torch.nn as nn
import torch.nn.functional as F
import os, argparse, math, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from scipy.spatial.distance import cdist, pdist
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.utils import from_scipy_sparse_matrix


def load_feature_and_graph(data_dir, use_mRNA=True, use_meth=True, use_miRNA=True):
    # Load the multi-omics data and concatenate the modality-specific features.
    data_train_list = []
    data_test_list = []
    data_list = []
    for i in range(1, 4): # num_view = 3
        data_train = np.loadtxt(os.path.join(data_dir, str(i) + "_tr.csv"), delimiter=',')
        data_test = np.loadtxt(os.path.join(data_dir, str(i) + "_te.csv"), delimiter=',')
        data_train_min = np.min(data_train, axis=0, keepdims=True)
        data_train_max = np.max(data_train, axis=0, keepdims=True)
        data_train = (data_train - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_test = (data_test - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_train_list.append(data_train.astype(float))
        data_test_list.append(data_test.astype(float))
        data_list.append(np.concatenate([data_train, data_test], axis=0)) # shape: (num_train+num_test, num_feature)
    label_train = np.loadtxt(os.path.join(data_dir, "labels_tr.csv"), delimiter=',').astype(int)
    label_test = np.loadtxt(os.path.join(data_dir, "labels_te.csv"), delimiter=',').astype(int)
    label = np.concatenate([label_train, label_test], axis=0) # shape: (num_train+num_test, )
    data_train_indices = np.arange(label_train.shape[0])
    data_test_indices = np.arange(label_train.shape[0], label_train.shape[0] + label_test.shape[0])
    # data_list: list of numpy arrays, each array is a matrix with shape (num_train+num_test, num_feature)
    # label: numpy array with shape (num_train+num_test, )
    # data_train_indices: numpy array with shape (num_train, )
    # data_test_indices: numpy array with shape (num_test, )
    print('data_list mRNA shape:', data_list[0].shape)
    print('data_list meth shape:', data_list[1].shape)
    print('data_list miRNA shape:', data_list[2].shape)
    print('label shape:', label.shape)
    print('data_train_indices shape:', data_train_indices.shape)
    print('data_test_indices shape:', data_test_indices.shape)

    # Construct the multi-omics hypergraph incidence matrix and concatenate the modality-specific hyperedges.
    graph_mRNA = pd.DataFrame(data_list[0].T).corr(method='pearson').to_numpy() if use_mRNA else None
    graph_meth = pd.DataFrame(data_list[1].T).corr(method='pearson').to_numpy() if use_meth else None
    graph_miRNA = pd.DataFrame(data_list[2].T).corr(method='pearson').to_numpy() if use_miRNA else None
    
    # combine three modality-specific graphs
    combined_graph = graph_mRNA + graph_meth + graph_miRNA # shape: (num_sample, num_sample)
    mean_graph = np.mean(combined_graph) # mean value of the graph
    combined_graph = np.where(combined_graph > mean_graph, 1, 0) # make sure all values are 0 or 1 
    print('combined_graph shape:', combined_graph.shape)
    print('label shape:', label.shape)
    return data_list, label, data_train_indices, data_test_indices, combined_graph


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout, graph_model='GCN'):
        super().__init__()
        self.views = len(in_dim)
        self.num_class = num_class
        self.dropout = dropout
        
        for i in range(self.views):
            setattr(self, 'view_' + str(i) + '_fc', nn.Linear(in_dim[i], hidden_dim[0]))
            setattr(self, 'view_' + str(i) + '_act', nn.ReLU())
            setattr(self, 'view_' + str(i) + '_dropout', nn.Dropout(p=dropout))
            setattr(self, 'view_' + str(i) + '_fc_1', nn.Linear(hidden_dim[0], hidden_dim[0]))
            setattr(self, 'view_' + str(i) + '_act_1', nn.ReLU())
            setattr(self, 'view_' + str(i) + '_dropout_1', nn.Dropout(p=dropout))
        
        # self.GCN = GCN(self.views*hidden_dim[0], hidden_dim[0])
        if graph_model == 'GCN':
            self.GNN = GCNConv(self.views*hidden_dim[0], hidden_dim[0])
        elif graph_model == 'GAT':
            self.GNN = GATConv(self.views*hidden_dim[0], hidden_dim[0])
        else:
            raise ValueError("The graph model is not valid. Please choose from 'GCN' or 'GAT'.")
        
        self.MMClasifier = []
        assert len(hidden_dim) >= 1, "The length of hidden dim need to be greater than or equal to 1."
        if len(hidden_dim) == 1:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], num_class))
        else:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], hidden_dim[1]))
            self.MMClasifier.append(nn.LeakyReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
            for layer in range(1, len(hidden_dim) -1):
                self.MMClasifier.append(nn.Linear(hidden_dim[layer], hidden_dim[layer+1]))
                self.MMClasifier.append(nn.LeakyReLU())
                self.MMClasifier.append(nn.Dropout(p=dropout))
            self.MMClasifier.append(nn.Linear(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data_list, graph):
        edge_index, edge_weight = dense_to_sparse(graph)
        edge_index = edge_index.to(data_list[0].device)
        edge_weight = edge_weight.to(data_list[0].device)
        feature_list = []
        for i in range(self.views):
            feature = getattr(self, 'view_' + str(i) + '_fc')(data_list[i])
            feature = getattr(self, 'view_' + str(i) + '_act')(feature)
            feature = getattr(self, 'view_' + str(i) + '_dropout')(feature)
            feature = getattr(self, 'view_' + str(i) + '_fc_1')(feature)
            feature = getattr(self, 'view_' + str(i) + '_act_1')(feature)
            feature = getattr(self, 'view_' + str(i) + '_dropout_1')(feature)
            feature_list.append(feature)
        MMfeature = torch.cat(feature_list, dim=1) # shape: (batch_size, num_view*hidden_dim)
        MMfeature_graph = self.GNN(MMfeature, edge_index, edge_weight)
        MMlogit = self.MMClasifier(torch.cat([MMfeature, MMfeature_graph], dim=1))
        return MMlogit
    
    def forward_criterion(self, data_list, labeled_indices, unlabeled_indices, label, graph):
        MMlogit = self.forward(data_list, graph)
        criterion = torch.nn.CrossEntropyLoss()
        MMloss = criterion(MMlogit[labeled_indices], label[labeled_indices])
        return MMloss

def count_gcn_flops(module, input, output):
    """
    GCN: H' = activation(D^(-1/2) * A * D^(-1/2) * H * W)
    """
    x, edge_index, edge_weight = input
    N, F_in = x.shape  # N: node number, F_in: input feature dimension
    F_out = output.shape[1]  # output feature dimension
    flops = 0
    # 1. Linear Transformation: H * W + Bias
    flops += N * F_in * F_out # MACs = 2 flops (multiply + add), there we calculate the MACs
    # 2. Graph Convolution: D^(-1/2) * A * D^(-1/2) * H * W
    # Assume A is a sparse matrix, and each node has an average of k neighbors.
    # For each node, we need to aggregate the information of its neighbors.
    # Assume the average degree is k, then there are N * k edges.
    k = edge_index.shape[1] // N if edge_index.shape[1] > 0 else 1  # average degree
    flops += N * k * F_out  # neighbor aggregation
    # 3. Activation Function (ReLU)
    flops += N * F_out
    module.total_ops += flops

def count_gat_flops(module, input, output):
    """
    GAT: H' = activation(∑_j alpha_ij * W * H_j), alpha_ij = softmax(LeakyReLU(a^T * [W * H_i || W * H_j]))
    """
    x, edge_index, edge_weight = input
    N, F_in = x.shape  # N: node number, F_in: input feature dimension
    F_out = output.shape[1]  # output feature dimension
    num_heads = module.heads if hasattr(module, 'heads') else 1
    head_dim = F_out // num_heads  # head dimension
    flops = 0
    # 1. Linear Transformation: W * H (for each head)
    flops += N * F_in * head_dim * num_heads # MACs = 2 flops (multiply + add), there we calculate the MACs
    # 2. Attention Mechanism Calculation
    # For each edge, we need to calculate the attention score
    k = edge_index.shape[1] // N if edge_index.shape[1] > 0 else 1  # average degree
    # 2.1 Calculate Attention Score: a^T * [W * H_i || W * H_j]; a^T.shape: (num_heads, 2 * head_dim, 1)
    # Each attention score needs: 1 linear transformation
    flops += edge_index.shape[1] * 2 * head_dim * num_heads
    # 2.2 LeakyReLU Activation
    flops += edge_index.shape[1] * num_heads
    # 2.3 Softmax Normalization (for each node's neighbors)
    # Each node needs to softmax its neighbors for each head
    flops += N * k * 3 * num_heads  # exp + sum + division
    # 3. Weighted Aggregation: ∑_j α_ij * W * H_j
    flops += edge_index.shape[1] * head_dim * num_heads
    # 4. Final Activation Function
    flops += N * F_out
    module.total_ops += flops

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/data_MOGONET/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/data_MOGONET/GNN/BRCA/', help='The output dir.')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('-m', '--modality', default=6, type=int, help='The option for modality missing.') # args.modality: {0:001, 1:010, 2:011, 3:100, 4:101, 5:110, 6:111}, xx1: mRNA = True, x1x: methylation = True, 1xx: miRNA = True
    parser.add_argument('-si', '--sigma', default=0, type=float, help='The standard deviation of the Gaussian noise.')
    parser.add_argument('-v', '--verbose', default=0, type=int, help='The verbose level.')
    parser.add_argument('-g', '--graph_model', default='GCN', type=str, help='The graph model to use. Options: GCN, GAT.')
    args = parser.parse_args()
    if 'BRCA' in args.data_dir: 
        hidden_dim = [500]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 5
    if 'ROSMAP' in args.data_dir: 
        hidden_dim = [300]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 2 
    if 'KIPAN' in args.data_dir:
        hidden_dim = [500]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 3 
    if 'LGG' in args.data_dir:
        hidden_dim = [500]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 2 
    if 'TCGA-23' in args.data_dir:
        hidden_dim = [500]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 23
    use_mRNA = ((args.modality+1) % 2 == 1); use_meth = ((args.modality+1) // 2 % 2 == 1); use_miRNA = ((args.modality+1) // 4 % 2 == 1); print(f"use_mRNA: {use_mRNA}, use_meth: {use_meth}, use_miRNA: {use_miRNA}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_list, label, data_train_indices, data_test_indices, graph = load_feature_and_graph(args.data_dir, use_mRNA=use_mRNA, use_meth=use_meth, use_miRNA=use_miRNA)
    dim_list = [data.shape[1] for data in data_list]
    # -- noisy condition control
    data_list = [data_list[i] + np.random.normal(0, args.sigma, data_list[i].shape) for i in range(len(data_list))] if args.sigma > 0 else data_list    
    # -- missing modality control
    data_list[0] = np.zeros_like(data_list[0]) if not use_mRNA else data_list[0]
    data_list[1] = np.zeros_like(data_list[1]) if not use_meth else data_list[1]
    data_list[2] = np.zeros_like(data_list[2]) if not use_miRNA else data_list[2]
    data_list = [torch.FloatTensor(data).to(device) for data in data_list]
    label = torch.LongTensor(label).to(device)
    graph = torch.FloatTensor(graph).to(device)
    model = GNN(dim_list, hidden_dim, num_class, dropout=0.5, graph_model=args.graph_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    
    # -- verbose == 2: print the FLOPs and params
    if args.verbose == 2:
        from thop import profile, clever_format
        flops, params = profile(model, inputs=(data_list, graph), verbose=False, custom_ops={GCNConv: count_gcn_flops, GATConv: count_gat_flops}) # return MACs and params; flops = 2 * MACs
        flops, params = clever_format([flops*2, params], "%.3f")
        print(f'Input data shape: {data_list[0].shape}, {data_list[1].shape}, {data_list[2].shape}')
        print(f'Input graph shape: {graph.shape}')
        print(f"FLOPs: {flops}, Params: {params}")
        print(f"Params by manual calculation: {sum(p.numel() for p in model.parameters())}")
        exit()
        
    for epoch in range(1, num_epoch + 1):
        model.train()
        optimizer.zero_grad()
        loss = model.forward_criterion(data_list=data_list, labeled_indices=data_train_indices, unlabeled_indices=data_test_indices, label=label, graph=graph)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print('Training Epoch {:d}: Loss={:.5f}'.format(epoch, loss.cpu().detach().numpy()))
            model.eval()
            with torch.no_grad():
                logit = model.forward(data_list, graph)
                prob = F.softmax(logit, dim=1).data.cpu().numpy()
                label_test = label[data_train_indices]
                prob_test = prob[data_train_indices]
            if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1 = f1_score(label_test.cpu().numpy(), prob_test.argmax(1))
                auc = roc_auc_score(label_test.cpu().numpy(), prob_test[:,1])
                print('Training Epoch {:d}: Train ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(epoch, acc, f1, auc))
            if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
                print('Training Epoch {:d}: Train ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
            if 'TCGA-23' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro') 
                ap = average_precision_score(np.eye(num_class)[label_test.cpu().numpy()], prob_test, average='macro')
                auc = roc_auc_score(np.eye(num_class)[label_test.cpu().numpy()], prob_test, average='macro')
                print('Training Epoch {:d}: Train ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}, macro_AUPR={:.5f}, macro_AUC={:.5f}'.format(epoch, acc, f1_weighted, f1_macro, ap, auc))
            with torch.no_grad():
                logit = model.forward(data_list, graph)
                prob = F.softmax(logit, dim=1).data.cpu().numpy()
                label_test = label[data_test_indices]
                prob_test = prob[data_test_indices]
            if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1 = f1_score(label_test.cpu().numpy(), prob_test.argmax(1))
                auc = roc_auc_score(label_test.cpu().numpy(), prob_test[:,1])
                print('Training Epoch {:d}: Test ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(epoch, acc, f1, auc))
            if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
                print('Training Epoch {:d}: Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
            if 'TCGA-23' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
                ap = average_precision_score(np.eye(num_class)[label_test.cpu().numpy()], prob_test, average='macro')
                auc = roc_auc_score(np.eye(num_class)[label_test.cpu().numpy()], prob_test, average='macro')
                print('Training Epoch {:d}: Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}, macro_AUPR={:.5f}, macro_AUC={:.5f}'.format(epoch, acc, f1_weighted, f1_macro, ap, auc))
                
    model.eval()
    with torch.no_grad():
        logit = model.forward(data_list, graph)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
        label_test = label[data_test_indices]
        prob_test = prob[data_test_indices]
    if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
        acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
        f1 = f1_score(label_test.cpu().numpy(), prob_test.argmax(1))
        auc = roc_auc_score(label_test.cpu().numpy(), prob_test[:,1])
        print('Test ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(acc, f1, auc))

        if not os.path.exists(os.path.join(args.output_dir, 'metrics.csv')):
            metrics_df = pd.DataFrame({'seed': [args.seed], 'modality': [args.modality], 'sigma': [args.sigma], 'ACC': [acc], 'F1': [f1], 'AUC': [auc]})
        else:
            metrics_df = pd.read_csv(os.path.join(args.output_dir, 'metrics.csv'))
            metrics_df = pd.concat([metrics_df, pd.DataFrame({'seed': [args.seed], 'modality': [args.modality], 'sigma': [args.sigma], 'ACC': [acc], 'F1': [f1], 'AUC': [auc]})], ignore_index=True)
        metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
    
        if args.verbose == 1:
            if len(metrics_df) >= 10:
                mean_values = metrics_df[['ACC', 'F1', 'AUC']].mean()
                variance_values = metrics_df[['ACC', 'F1', 'AUC']].var()
                metrics_statistics_df = pd.DataFrame(columns=['statistics', 'ACC', 'F1', 'AUC'])
                mean_row_df = pd.DataFrame({'statistics': 'mean', 'ACC': mean_values['ACC'], 'F1': mean_values['F1'], 'AUC': mean_values['AUC']}, index=[0])
                var_row_df = pd.DataFrame({'statistics': 'var', 'ACC': variance_values['ACC'], 'F1': variance_values['F1'], 'AUC': variance_values['AUC']}, index=[0])
                metrics_statistics_df = pd.concat([mean_row_df, var_row_df], ignore_index=True)
                metrics_statistics_df.to_csv(os.path.join(args.output_dir, 'metrics_statistics.csv'), index=False) 
        
    if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
        acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
        f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
        f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
        print('Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(acc, f1_weighted, f1_macro))
        
        if not os.path.exists(os.path.join(args.output_dir, 'metrics.csv')):
            metrics_df = pd.DataFrame({'seed': [args.seed], 'modality': [args.modality], 'sigma': [args.sigma], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro]})
        else:
            metrics_df = pd.read_csv(os.path.join(args.output_dir, 'metrics.csv'))
            metrics_df = pd.concat([metrics_df, pd.DataFrame({'seed': [args.seed], 'modality': [args.modality], 'sigma': [args.sigma], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro]})], ignore_index=True)
        metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
        
        if args.verbose == 1:
            if len(metrics_df) >= 10:
                mean_values = metrics_df[['ACC', 'F1_weighted', 'F1_macro']].mean()
                variance_values = metrics_df[['ACC', 'F1_weighted', 'F1_macro']].var()
                metrics_statistics_df = pd.DataFrame(columns=['statistics', 'ACC', 'F1_weighted', 'F1_macro'])
                mean_row_df = pd.DataFrame({'statistics': 'mean', 'ACC': mean_values['ACC'], 'F1_weighted': mean_values['F1_weighted'], 'F1_macro': mean_values['F1_macro']}, index=[0])
                var_row_df = pd.DataFrame({'statistics': 'var', 'ACC': variance_values['ACC'], 'F1_weighted': variance_values['F1_weighted'], 'F1_macro': variance_values['F1_macro']}, index=[0])
                metrics_statistics_df = pd.concat([mean_row_df, var_row_df], ignore_index=True)
                metrics_statistics_df.to_csv(os.path.join(args.output_dir, 'metrics_statistics.csv'), index=False)

    if 'TCGA-23' in args.data_dir:
        acc = accuracy_score(label_test.cpu().numpy(), prob_test.argmax(1))
        f1_weighted = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='weighted')
        f1_macro = f1_score(label_test.cpu().numpy(), prob_test.argmax(1), average='macro')
        ap = average_precision_score(np.eye(num_class)[label_test.cpu().numpy()], prob_test, average='macro')
        auc = roc_auc_score(np.eye(num_class)[label_test.cpu().numpy()], prob_test, average='macro')
        print('Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}, macro_AUPR={:.5f}, macro_AUC={:.5f}'.format(acc, f1_weighted, f1_macro, ap, auc))
        
        if not os.path.exists(os.path.join(args.output_dir, 'metrics.csv')):
            metrics_df = pd.DataFrame({'seed': [args.seed], 'modality': [args.modality], 'sigma': [args.sigma], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro], 'macro_AUPR': [ap], 'macro_AUC': [auc]})
        else:
            metrics_df = pd.read_csv(os.path.join(args.output_dir, 'metrics.csv'))
            metrics_df = pd.concat([metrics_df, pd.DataFrame({'seed': [args.seed], 'modality': [args.modality], 'sigma': [args.sigma], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro], 'macro_AUPR': [ap], 'macro_AUC': [auc]})], ignore_index=True)
        metrics_df.to_csv(os.path.join(args.output_dir, 'metrics.csv'), index=False)
        
        if args.verbose == 1:
            if len(metrics_df) >= 10:
                mean_values = metrics_df[['ACC', 'F1_weighted', 'F1_macro', 'macro_AUPR', 'macro_AUC']].mean()
                variance_values = metrics_df[['ACC', 'F1_weighted', 'F1_macro', 'macro_AUPR', 'macro_AUC']].var()
                metrics_statistics_df = pd.DataFrame(columns=['statistics', 'ACC', 'F1_weighted', 'F1_macro', 'macro_AUPR', 'macro_AUC'])
                mean_row_df = pd.DataFrame({'statistics': 'mean', 'ACC': mean_values['ACC'], 'F1_weighted': mean_values['F1_weighted'], 'F1_macro': mean_values['F1_macro'], 'macro_AUPR': mean_values['macro_AUPR'], 'macro_AUC': mean_values['macro_AUC']}, index=[0])
                var_row_df = pd.DataFrame({'statistics': 'var', 'ACC': variance_values['ACC'], 'F1_weighted': variance_values['F1_weighted'], 'F1_macro': variance_values['F1_macro'], 'macro_AUPR': variance_values['macro_AUPR'], 'macro_AUC': variance_values['macro_AUC']}, index=[0])
                metrics_statistics_df = pd.concat([mean_row_df, var_row_df], ignore_index=True)
                metrics_statistics_df.to_csv(os.path.join(args.output_dir, 'metrics_statistics.csv'), index=False)
