import os, math, random, argparse, numpy as np, pandas as pd
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def load_feature_and_hyperedge(data_dir, view_list):
    def construct_hyperedge_with_KNN(X, k_list=[10], is_prob=True, m_prob=1, edge_type='euclid'):
        hyperedge_mat_list = []
        for k in k_list:
            if edge_type == 'euclid':
                distance_mat = cdist(X, X, 'euclid')
                hyperedge_mat = np.zeros(distance_mat.shape)
                for center_idx in range(distance_mat.shape[0]):
                    distance_mat[center_idx, center_idx] = 0
                    distance_vec = distance_mat[center_idx]
                    distance_vec_avg = np.average(distance_vec)
                    nearest_idx = np.array(np.argsort(distance_vec)).squeeze() 
                    nearest_idx[k - 1] = center_idx if not np.any(nearest_idx[:k] == center_idx) else nearest_idx[k - 1] # add the center node to the nearest neighbors if it is not in the top k
                    for node_idx in nearest_idx[:k]:
                        hyperedge_mat[node_idx, center_idx] = np.exp(-distance_vec[node_idx] ** 2 / (m_prob * distance_vec_avg) ** 2) if is_prob else 1.0 # Gaussian kernel for computing the hyperedge weight
            hyperedge_mat_list.append(hyperedge_mat)
        return np.hstack(hyperedge_mat_list)
    
    def generate_G_from_H(H, variable_weight=False):
        # Calculate G from hypgerraph incidence matrix H, where G = DV2 * H * W * invDE * HT * DV2
        H = np.array(H) # shape: N X M, N is the number of nodes, M is the number of hyperedges
        W = np.ones(H.shape[1]) # the weight of the hyperedge
        DV = np.sum(H * W, axis=1) # the degree of the node
        DE = np.sum(H, axis=0) # the degree of the hyperedge
        invDE = np.mat(np.diag(np.power(DE, -1))) # shape: M X M
        invDV2 = np.mat(np.diag(np.power(DV, -0.5))) # shape: N X N
        W = np.mat(np.diag(W)) # shape: M X M
        H = np.mat(H) # shape: N X M
        if variable_weight:
            return invDV2 * H, W, invDE * H.T * invDV2
        else:
            return invDV2 * H * W * invDE * H.T * invDV2 # shape: N X N

    # Load the multi-omics data and concatenate the modality-specific features.
    data_train_list = []
    data_test_list = []
    data_list = []
    for i in range(1, 4): # num_view = 3
        data_train = np.loadtxt(os.path.join(data_dir, str(i) + "_tr.csv"), delimiter=',')
        data_test = np.loadtxt(os.path.join(data_dir, str(i) + "_te.csv"), delimiter=',')
        # data_train_min = np.min(data_train, axis=0, keepdims=True) 
        # data_train_max = np.max(data_train, axis=0, keepdims=True)
        # data_train = (data_train - data_train_min)/(data_train_max - data_train_min + 1e-10)
        # data_test = (data_test - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_train_list.append(data_train.astype(float))
        data_test_list.append(data_test.astype(float))
    label_train = np.loadtxt(os.path.join(data_dir, "labels_tr.csv"), delimiter=',').astype(int)
    label_test = np.loadtxt(os.path.join(data_dir, "labels_te.csv"), delimiter=',').astype(int)
    label = np.concatenate([label_train, label_test], axis=0) # shape: (num_train+num_test, )
    data_train_indices = np.arange(label_train.shape[0])
    data_test_indices = np.arange(label_train.shape[0], label_train.shape[0] + label_test.shape[0])

    # Construct the multi-omics hypergraph incidence matrix and concatenate the modality-specific hyperedges.
    H_tr = []; H_te = []
    for i in range(len(data_train_list)):
        H_tr.append(construct_hyperedge_with_KNN(data_train_list[i], k_list=[3], is_prob=True, m_prob=1))
        H_te.append(construct_hyperedge_with_KNN(data_test_list[i], k_list=[3], is_prob=True, m_prob=1))
    H_train = np.concatenate(H_tr, axis=1)
    H_test = np.concatenate(H_te, axis=1)
    
    # Calculate the sample weight for the training set
    unique_labels, count = np.unique(label, return_counts=True)
    sample_weight = np.zeros(len(label))
    for i in range(len(unique_labels)):
        sample_weight[np.where(label == unique_labels[i])[0]] = count[i] / np.sum(count)
    sample_weight_train = sample_weight[data_train_indices]
    
    # Calculate the adjacency matrix for the training and test sets
    adj_train = generate_G_from_H(H_train, variable_weight=False)
    adj_test = generate_G_from_H(H_test, variable_weight=False)
    return data_train_list, data_test_list, label, data_train_indices, data_test_indices, adj_train, adj_test, sample_weight_train
    
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None # self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight) # node feature transformation
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(G, x) # hypergraph convolution
        return x
    
class HGNN(nn.Module):
    def __init__(self, in_ch, hidden_dim, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc_1 = HGNN_conv(in_ch, hidden_dim[0])
        self.hgc_2 = HGNN_conv(hidden_dim[0], hidden_dim[1])

    def forward(self, x, G):
        x = F.dropout(F.leaky_relu(self.hgc_1(x, G), 0.25), self.dropout)
        x = F.leaky_relu(self.hgc_2(x, G), 0.25)
        return x
    
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(self.xavier_init)
           
    def forward(self, x):
        return self.clf(x)
    
    @staticmethod
    def xavier_init(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, n_hidden, n_head, dropout, num_class):
        super(TransformerEncoder, self).__init__()
        self.input_data_dims = input_data_dims; self.n_hidden = n_hidden; self.n_head = n_head; self.dropout = dropout; self.n_class = num_class; self.modal_num = len(input_data_dims)
        self.w_qs = nn.ModuleList([nn.Linear(dim, self.n_head * n_hidden, bias=True) for dim in self.input_data_dims])
        self.w_ks = nn.ModuleList([nn.Linear(dim, self.n_head * n_hidden, bias=True) for dim in self.input_data_dims])
        self.w_vs = nn.ModuleList([nn.Linear(dim, self.n_head * n_hidden, bias=True) for dim in self.input_data_dims])
        self.att_dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.n_hidden * self.n_head, self.n_hidden * self.n_head)
        self.dropout = nn.Dropout(self.dropout)
        self.layer_norm = nn.LayerNorm(self.n_hidden * self.n_head, eps=1e-6)
        self.classifier = nn.Sequential(nn.Linear(self.modal_num*self.n_head*self.n_hidden, num_class))
        self.classifier.apply(self.xavier_init)
    
    def forward(self, input_data, mask=None): # input_data: (bs, modal_num, input_data_dims)
        bs = input_data[0].shape[0]; q = []; k = []; v = []
        for i in range(self.modal_num):
            q.append(self.w_qs[i](input_data[i]).view(-1, self.n_head, self.n_hidden)) # shape: (bs, n_head, n_hidden)
            k.append(self.w_ks[i](input_data[i]).view(-1, self.n_head, self.n_hidden)) # shape: (bs, n_head, n_hidden)
            v.append(self.w_vs[i](input_data[i]).view(-1, self.n_head, self.n_hidden)) # shape: (bs, n_head, n_hidden)
        q = torch.stack(q, dim=1); q = q.transpose(1, 2) # shape: (bs, n_head, modal_num, n_hidden)
        k = torch.stack(k, dim=1); k = k.transpose(1, 2) # shape: (bs, n_head, modal_num, n_hidden)
        v = torch.stack(v, dim=1); v = v.transpose(1, 2) # shape: (bs, n_head, modal_num, n_hidden)
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(k.size(-1)) # shape: (bs, n_head, modal_num, modal_num)
        attn = attn.masked_fill(mask == 0, -1e9) if mask is not None else attn # shape: (bs, n_head, modal_num, modal_num)
        attn = F.softmax(attn, dim=-1) # shape: (bs, n_head, modal_num, modal_num)
        attn = self.att_dropout(attn) # shape: (bs, n_head, modal_num, modal_num)
        output = torch.matmul(attn, v) # shape: (bs, n_head, modal_num, n_hidden)
        output = output.transpose(1, 2).contiguous().view(-1, self.modal_num, self.n_head * self.n_hidden)
        residual = output.transpose(1, 2).contiguous().view(-1, self.modal_num, self.n_head * self.n_hidden)
        output = self.dropout(self.fc(output))
        output = output + residual
        output = self.layer_norm(output)
        output = torch.reshape(output, (-1, self.modal_num * self.n_head * self.n_hidden)) # shape: (bs, modal_num*n_head*n_hidden)
        output = self.classifier(output) # shape: (bs, num_class)
        return output
    
    @staticmethod
    def xavier_init(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
def count_hgnn_flops(module, input, output):
    x, G = input
    N, F_in = x.shape  # N: node number, F_in: input feature dimension
    F_out = output.shape[1]  # output feature dimension
    flops = 0
    # 1. Linear Transformation: H * W + Bias
    flops += N * F_in * F_out # MACs = 2 flops (multiply + add), there we calculate the MACs
    # 2. Hypergraph Convolution: D^(-1/2) * H * W * D^(-1/2)
    flops += N * N  * F_out # MACs = 2 flops (multiply + add), there we calculate the MACs
    # # 3. Activation Function (ReLU)
    # flops += N * F_out
    module.total_ops += flops

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/data_MOGONET/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/data_MOGONET/MORE/BRCA/', help='The output dir.')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--n_hidden', type=int, default=10, help='Number of hidden units.')
    parser.add_argument('--n_head', type=int, default=4, help='Number of head.')
    parser.add_argument('-v', '--verbose', default=0, type=int, help='The verbose level.')
    args = parser.parse_args()
    if 'BRCA' in args.data_dir: 
        num_class = 5; view_list = [1, 2, 3]; num_epoch_pretrain = 500; num_epoch = 1500; lr_e_pretrain = 1e-3; lr_e = 5e-4; lr_c = 1e-3
    if 'ROSMAP' in args.data_dir: 
        num_class = 2; view_list = [1, 2, 3]; num_epoch_pretrain = 500; num_epoch = 1500; lr_e_pretrain = 1e-3; lr_e = 5e-4; lr_c = 1e-3
    if 'KIPAN' in args.data_dir:
        num_class = 3; view_list = [1, 2, 3]; num_epoch_pretrain = 500; num_epoch = 1500; lr_e_pretrain = 1e-3; lr_e = 5e-4; lr_c = 1e-3
    if 'LGG' in args.data_dir:
        num_class = 2; view_list = [1, 2, 3]; num_epoch_pretrain = 500; num_epoch = 1500; lr_e_pretrain = 1e-3; lr_e = 5e-4; lr_c = 1e-3
    if 'TCGA-23' in args.data_dir:
        num_class = 23; view_list = [1, 2, 3]; num_epoch_pretrain = 500; num_epoch = 1500; lr_e_pretrain = 1e-3; lr_e = 5e-4; lr_c = 1e-3
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_train_list, data_test_list, label, data_train_indices, data_test_indices, adj_train, adj_test, sample_weight_train = load_feature_and_hyperedge(args.data_dir, view_list)
    data_train_list = [torch.FloatTensor(data).to(device) for data in data_train_list]
    data_test_list = [torch.FloatTensor(data).to(device) for data in data_test_list]
    label = torch.LongTensor(label).to(device)
    adj_train = torch.FloatTensor(adj_train).to(device)
    adj_test = torch.FloatTensor(adj_test).to(device)
    sample_weight_train = torch.FloatTensor(sample_weight_train).to(device)
    
    model_dict = {}
    for i in range(len(view_list)):
        model_dict["E{:}".format(i+1)] = HGNN(data_train_list[i].shape[1], hidden_dim=[args.n_hidden, 10]).to(device)
        model_dict["C{:}".format(i+1)] = Classifier(10, num_class).to(device)
    model_dict["C"] = TransformerEncoder(input_data_dims=[10, 10, 10], n_hidden=args.n_hidden, n_head=args.n_head, dropout=args.dropout, num_class=num_class).to(device)
    
    # -- verbose == 2: print the FLOPs and params
    if args.verbose == 2:
        class MORE(nn.Module):
            def __init__(self):
                super(MORE, self).__init__()
                self.E = nn.ModuleList([HGNN(data_train_list[i].shape[1], hidden_dim=[args.n_hidden, 10]) for i in range(len(view_list))])
                self.C = nn.ModuleList([Classifier(10, num_class) for i in range(len(view_list))])
                self.C_general = TransformerEncoder(input_data_dims=[10, 10, 10], n_hidden=args.n_hidden, n_head=args.n_head, dropout=args.dropout, num_class=num_class)
                
            def forward(self, x, G):
                embeddings = [self.E[i](x[i], G) for i in range(len(x))]
                classifications = [self.C[i](embeddings[i]) for i in range(len(x))]
                final_classification = self.C_general(embeddings)
                return embeddings, classifications, final_classification
            
        from thop import profile, clever_format
        data_list = [torch.cat([data_train, data_test], dim=0) for data_train, data_test in zip(data_train_list, data_test_list)]
        params = sum([sum(p.numel() for p in model.parameters()) for model in model_dict.values()])
        flops, params = profile(MORE().to(device), inputs=(data_list, torch.ones([len(data_list[0]), len(data_list[0])]).to(device)), verbose=False, custom_ops={HGNN_conv: count_hgnn_flops}) # return MACs and params; flops = 2 * MACs
        flops, params = clever_format([flops*2, params], "%.3f")
        print(f'Input data shape: {data_list[0].shape}, {data_list[1].shape}, {data_list[2].shape}')
        print(f'Input graph shape: {torch.ones([len(data_list[0]), len(data_list[0])]).shape}')
        print(f"FLOPs: {flops}, Params: {params}")
        print(f"Params by manual calculation: {sum(p.numel() for p in MORE().parameters())}")
        exit()
    
    # 1. Pretraining Stage for the encoder
    optim_dict = {"C{:}".format(i+1): torch.optim.Adam(list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), lr=lr_e) for i in range(len(view_list))}
    optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    for epoch in range(num_epoch_pretrain):
        model_dict = {n: m.train() for n, m in model_dict.items()}
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_dict = {}
        for i in range(len(view_list)):
            optim_dict["C{:}".format(i+1)].zero_grad()
            output = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_train_list[i], adj_train))
            loss = torch.mean(torch.mul(criterion(output, label[data_train_indices]), sample_weight_train))
            loss.backward()
            optim_dict["C{:}".format(i+1)].step()
            loss_dict["C{:}".format(i+1)] = loss.detach().cpu().numpy().item()
        
    # 2. Training Stage for the classifier and the encoder
    optim_dict = {"C{:}".format(i+1): torch.optim.Adam(list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), lr=lr_e) for i in range(len(view_list))}
    optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    for epoch in range(num_epoch+1):
        model_dict = {n: m.train() for n, m in model_dict.items()}
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss_dict = {}
        # 2.1 Training Stage for the encoder and the specific classifier
        for i in range(len(view_list)):
            optim_dict["C{:}".format(i+1)].zero_grad()
            output = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_train_list[i], adj_train))
            loss = torch.mean(torch.mul(criterion(output, label[data_train_indices]), sample_weight_train))
            loss.backward()
            optim_dict["C{:}".format(i+1)].step()
            loss_dict["C{:}".format(i+1)] = loss.detach().cpu().numpy().item()
        # 2.2 Training Stage for the general classifier
        optim_dict["C"].zero_grad()
        encoder_output_list = [model_dict['E{:}'.format(i+1)](data_train_list[i], adj_train) for i in range(len(view_list))]
        output = model_dict["C"](encoder_output_list)
        loss = torch.mean(torch.mul(criterion(output, label[data_train_indices]), sample_weight_train))
        loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = loss.detach().cpu().numpy().item()
        # 2.3 Testing Stage for the general classifier
        if epoch % 50 == 0:
            model_dict = {n: m.eval() for n, m in model_dict.items()}
            with torch.no_grad():
                encoder_output_list = [model_dict['E{:}'.format(i+1)](data_test_list[i], adj_test) for i in range(len(view_list))]
                output = model_dict["C"](encoder_output_list)
                prob = F.softmax(output, dim=1).data.cpu().numpy()
            print("Test: Epoch {:d}".format(epoch))
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(label[data_test_indices].cpu().numpy(), prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(label[data_test_indices].cpu().numpy(), prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(label[data_test_indices].cpu().numpy(), prob[:,1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(label[data_test_indices].cpu().numpy(), prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(label[data_test_indices].cpu().numpy(), prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(label[data_test_indices].cpu().numpy(), prob.argmax(1), average='macro')))
                
    # 3. Testing Stage for the general classifier
    model_dict = {n: m.eval() for n, m in model_dict.items()}
    with torch.no_grad():
        encoder_output_list = [model_dict['E{:}'.format(i+1)](data_test_list[i], adj_test) for i in range(len(view_list))]
        output = model_dict["C"](encoder_output_list)
        prob = F.softmax(output, dim=1).data.cpu().numpy()
        label_test = label[data_test_indices].cpu().numpy()
        prob_test = prob # shape: (bs, num_class)
    if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
        acc = accuracy_score(label_test, prob_test.argmax(1))
        f1 = f1_score(label_test, prob_test.argmax(1))
        auc = roc_auc_score(label_test, prob_test[:,1])
        print('Test ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(acc, f1, auc))

        if not os.path.exists(os.path.join(args.output_dir, 'metrics.csv')):
            metrics_df = pd.DataFrame({'seed': [args.seed], 'ACC': [acc], 'F1': [f1], 'AUC': [auc]})
        else:
            metrics_df = pd.read_csv(os.path.join(args.output_dir, 'metrics.csv'))
            metrics_df = pd.concat([metrics_df, pd.DataFrame({'seed': [args.seed], 'ACC': [acc], 'F1': [f1], 'AUC': [auc]})], ignore_index=True)
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
        acc = accuracy_score(label_test, prob_test.argmax(1))
        f1_weighted = f1_score(label_test, prob_test.argmax(1), average='weighted')
        f1_macro = f1_score(label_test, prob_test.argmax(1), average='macro')
        print('Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(acc, f1_weighted, f1_macro))
        
        if not os.path.exists(os.path.join(args.output_dir, 'metrics.csv')):
            metrics_df = pd.DataFrame({'seed': [args.seed], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro]})
        else:
            metrics_df = pd.read_csv(os.path.join(args.output_dir, 'metrics.csv'))
            metrics_df = pd.concat([metrics_df, pd.DataFrame({'seed': [args.seed], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro]})], ignore_index=True)
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
        acc = accuracy_score(label_test, prob_test.argmax(1))
        f1_weighted = f1_score(label_test, prob_test.argmax(1), average='weighted')
        f1_macro = f1_score(label_test, prob_test.argmax(1), average='macro')
        ap = average_precision_score(np.eye(num_class)[label_test], prob_test, average='macro')
        auc = roc_auc_score(np.eye(num_class)[label_test], prob_test, average='macro')
        print('Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}, macro_AUPR={:.5f}, macro_AUC={:.5f}'.format(acc, f1_weighted, f1_macro, ap, auc))
        
        if not os.path.exists(os.path.join(args.output_dir, 'metrics.csv')):
            metrics_df = pd.DataFrame({'seed': [args.seed], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro], 'macro_AUPR': [ap], 'macro_AUC': [auc]})
        else:
            metrics_df = pd.read_csv(os.path.join(args.output_dir, 'metrics.csv'))
            metrics_df = pd.concat([metrics_df, pd.DataFrame({'seed': [args.seed], 'ACC': [acc], 'F1_weighted': [f1_weighted], 'F1_macro': [f1_macro], 'macro_AUPR': [ap], 'macro_AUC': [auc]})], ignore_index=True)
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
                