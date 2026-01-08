import os, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim)) if bias else None
        nn.init.xavier_normal_(self.weight.data)
        nn.init.zeros_(self.bias.data) if bias else None
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        return (output + self.bias) if self.bias is not None else output
    
def count_gcn_flops(module, input, output):
    """
    m: GraphConvolution module (sparse matrix)
    x: tuple(inputs) -> (x, adj)
    y: output tensor (N, F_out)
    # count the MACs, including: (the Multiplication and Addition)
    # MACs = 2 flops (multiply + add), there we calculate the MACs
    # 1) support = x @ W: N * F_in * F_out
    # 2) output = adj @ support (sparse-dense): nnz(adj) * F_out
    # 3) bias addition: N * F_out (if exists)
    """
    x, G = input # N: node number, F_in: input feature dimension
    N, F_in = x.shape
    F_out = output.shape[1] # output feature dimension
    # 1) 1. Linear Transformation: H * W + Bias / dense matmul x @ W
    macs_dense = N * F_in * F_out
    # 2) 2. Graph Convolution: D^(-1/2) * H * W * D^(-1/2) / sparse mm adj @ support
    ### torch.sparse.mm is a sparse matrix multiplication function, which is faster than torch.matmul for sparse matrices.
    # if G.layout is torch.sparse_coo or str(G.layout).startswith("torch.sparse"): # sparse tensor
    #     nnz = int(G._nnz()) # number of non-zero elements
    # else:
    #     nnz = int((G != 0).sum().item()) if not G.is_sparse else int(G._nnz()) # dense tensor, count the non-zero elements
    # macs_graphconv = nnz * F_out
    # ### torch.mm is a dense matrix multiplication function, which is faster than torch.sparse.mm for dense matrices.
    macs_graphconv = N * N * F_out
    # 3) 3. Bias Addition: Bias / bias add
    # bias_ops = (N * F_out) if (m.bias is not None) else 0
    # total_macs = macs_dense + macs_sparse + bias_ops
    MACs = macs_dense + macs_graphconv
    module.total_ops += MACs

class GCN_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hidden_dim[0])
        self.gc2 = GraphConvolution(hidden_dim[0], hidden_dim[1])
        self.gc3 = GraphConvolution(hidden_dim[1], hidden_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(F.leaky_relu(self.gc1(x, adj), 0.25), self.dropout, training=self.training)
        x = F.dropout(F.leaky_relu(self.gc2(x, adj), 0.25), self.dropout, training=self.training)
        x = F.leaky_relu(self.gc3(x, adj), 0.25)
        return x


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class VCDN(nn.Module):
    def __init__(self, num_view, num_class, hidden_dim):
        super().__init__()
        self.num_view = num_view
        self.num_class = num_class
        self.model = nn.Sequential(nn.Linear(pow(num_class, num_view), hidden_dim), nn.LeakyReLU(0.25), nn.Linear(hidden_dim, num_class))
        self.model.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, data_list):
        data_list = [torch.sigmoid(x) for x in data_list] # Multi-view Classification Results
        x = torch.reshape(torch.matmul(data_list[0].unsqueeze(-1), data_list[1].unsqueeze(1)), (-1, pow(self.num_class, 2), 1))
        for i in range(2, self.num_view):
            x = torch.reshape(torch.matmul(x, data_list[i].unsqueeze(1)), (-1, pow(self.num_class, i+1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_class, self.num_view)))
        output = self.model(vcdn_feat)
        return output
        # data_list = [torch.sigmoid(x) for x in data_list]
        # x = torch.matmul(data_list[0].unsqueeze(-1), data_list[1].unsqueeze(1)) # shape: (N, num_class, num_class)
        # for i in range(2, self.num_view):
        #     x = torch.matmul(torch.reshape(x, (-1, pow(self.num_class, i))).unsqueeze(-1), data_list[i].unsqueeze(1))
        # vcdn_feat = torch.reshape(x, (-1, pow(self.num_class, self.num_view)))
        # output = self.model(vcdn_feat)
        # return output


def prepare_data(data_dir):
    data_train_list = []
    data_test_list = []
    data_all_list = []
    for i in range(1, 4): # num_view = 3
        data_train = np.loadtxt(os.path.join(data_dir, str(i) + "_tr.csv"), delimiter=',')
        data_test = np.loadtxt(os.path.join(data_dir, str(i) + "_te.csv"), delimiter=',')
        data_train_min = np.min(data_train, axis=0, keepdims=True)
        data_train_max = np.max(data_train, axis=0, keepdims=True)
        data_train = (data_train - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_test = (data_test - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_train_list.append(data_train.astype(float))
        data_test_list.append(data_test.astype(float))
        data_all_list.append(np.concatenate((data_train_list[i-1], data_test_list[i-1]), axis=0))
    label_train = np.loadtxt(os.path.join(data_dir, "labels_tr.csv"), delimiter=',').astype(int)
    label_test = np.loadtxt(os.path.join(data_dir, "labels_te.csv"), delimiter=',').astype(int)
    label_all = np.concatenate((label_train, label_test), axis=0)
    return data_train_list, data_test_list, data_all_list, label_train, label_test, label_all


def generate_adj_mat(data_train_list, data_test_list, adj_avg_degree):
    cosine_distance = lambda x1, x2: 1 - np.dot(x1, x2.T)/(np.linalg.norm(x1, ord=2, axis=1, keepdims=True)*np.linalg.norm(x2, ord=2, axis=1, keepdims=True).T).clip(min=1e-8)
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_train_list)):
        distance = cosine_distance(data_train_list[i], data_train_list[i])
        distance_sorted = np.sort(distance.reshape(-1,))
        distance_threshold = distance_sorted[adj_avg_degree*data_train_list[i].shape[0]] # the edge will be retained when node distance is less than or equal to distance_threshold
        
        distance_train_train = cosine_distance(data_train_list[i], data_train_list[i])
        adj = (distance_train_train <= distance_threshold)
        adj = (1-distance_train_train) * adj # similarity adjacent matrix
        adj = adj + adj.T * (adj.T > adj) - adj * (adj.T > adj) # correct the adjacent matrix
        np.fill_diagonal(adj, 1)
        adj = adj / np.sum(adj, axis=1, keepdims=True) # 1-order normalization
        adj_train_list.append(adj)
        
        data_train_test = np.concatenate((data_train_list[i], data_test_list[i]), axis=0)
        adj = np.zeros((data_train_test.shape[0], data_train_test.shape[0]))
        num_train = len(data_train_list[i])
        distance_train_test = cosine_distance(data_train_list[i], data_test_list[i])
        adj_train_test = (distance_train_test <= distance_threshold)
        adj[:num_train, num_train:] = (1 - distance_train_test) * adj_train_test
        distance_test_train = cosine_distance(data_test_list[i], data_train_list[i])
        adj_test_train = (distance_test_train <= distance_threshold)
        adj[num_train:, :num_train] = (1 - distance_test_train) * adj_test_train
        adj = adj + adj.T * (adj.T > adj) - adj * (adj.T > adj) # correct the adjacent matrix, remain the max value
        np.fill_diagonal(adj, 1)
        adj = adj / np.sum(adj, axis=1, keepdims=True) # 1-order normalization
        adj_test_list.append(adj)
        
    return adj_train_list, adj_test_list


def save_model_dict(folder, model_dict):
    os.makedirs(folder, exist_ok=True)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module + '.pth'))
        
        
def load_model_dict(folder, model_dict):
    for module in model_dict:
        print(os.path.join(folder, module + '.pth'))
        if os.path.exists(os.path.join(folder, module + '.pth')):
            model_dict[module].load_state_dict(torch.load(os.path.join(folder, module+".pth")))
        else:
            print('WARNING: Module {:} from model_dict is not loaded!'.format(module))
    return model_dict


def calculate_sample_weight(label, num_class):
    count = np.array([np.sum(label==i) for i in range(num_class)]) # shape: (num_class)
    weight = count / np.sum(count) # shape: (num_class)
    sample_weight = [weight[i] for i in label] # shape: (num_sample)
    return sample_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/BRCA/', help='The output dir.')
    parser.add_argument('-v', '--verbose', default=1, type=int, help='The verbose level.')
    args = parser.parse_args()
    if 'ROSMAP' in args.data_dir: 
        num_class = 2; adj_avg_degree = 2; hidden_dim_list = [200, 200, 100]; num_epoch_pretrain = 500; num_epoch = 2500; lr_e_pretrain = 1e-3; lr_e = 5e-4; lr_c = 1e-3
    if 'BRCA' in args.data_dir: 
        num_class = 5; adj_avg_degree = 10; hidden_dim_list = [400, 400, 200]; num_epoch_pretrain = 500; num_epoch = 2500; lr_e_pretrain = 1e-3; lr_e = 5e-4; lr_c = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    
    # Load Data.
    data_train_list, data_test_list, data_all_list, label_train, label_test, label_all = prepare_data(data_dir=args.data_dir)
    adj_train_list, adj_test_list = generate_adj_mat(data_train_list, data_test_list, adj_avg_degree)
    sample_weight_train = calculate_sample_weight(label_train, num_class)
    label_train = torch.tensor(label_train, dtype=torch.long).to(device)
    onehot_label_train = F.one_hot(label_train, num_class).to(device)
    sample_weight_train = torch.tensor(sample_weight_train, dtype=torch.float).to(device)
    label_test = torch.tensor(label_test, dtype=torch.long).to(device)
    label_test = label_test.cpu().numpy()
    label_all = torch.tensor(label_all, dtype=torch.long).to(device)
    for i in range(len(data_train_list)):
        data_train_list[i] = torch.tensor(data_train_list[i], dtype=torch.float).to(device)
        data_test_list[i] = torch.tensor(data_test_list[i], dtype=torch.float).to(device)
        data_all_list[i] = torch.tensor(data_all_list[i], dtype=torch.float).to(device)
        adj_train_list[i] = torch.tensor(adj_train_list[i], dtype=torch.float).to_sparse().to(device)
        adj_test_list[i] = torch.tensor(adj_test_list[i], dtype=torch.float).to_sparse().to(device)

    # Define Model.
    num_view = len(data_train_list)
    data_train_dim_list = [x.shape[1] for x in data_train_list]
    model_dict = {}
    for i in range(num_view):
        model_dict["E{:}".format(i+1)] = GCN_Encoder(data_train_dim_list[i], hidden_dim_list, 0.5)
        model_dict["C{:}".format(i+1)] = Classifier(hidden_dim_list[-1], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, pow(num_class, num_view))
    for m in model_dict:
        model_dict[m].to(device)
        # print(next(model_dict[m].parameters()))
        
    # -- verbose == 2: print the FLOPs and params
    print(adj_train_list[0].to_dense().shape) # (num_train, num_train)
    print(adj_test_list[0].to_dense().shape) # (num_train + num_test, num_train + num_test)
    if args.verbose == 1:
        class MOGONET(nn.Module):
            def __init__(self):
                super(MOGONET, self).__init__()
                self.E = nn.ModuleList([model_dict["E{:}".format(i+1)] for i in range(num_view)])
                self.C = nn.ModuleList([model_dict["C{:}".format(i+1)] for i in range(num_view)])
                self.C_general = model_dict["C"]
                
            def forward(self, x, G):
                embeddings = [self.E[i](x[i], G[i]) for i in range(num_view)]
                classifications = [self.C[i](embeddings[i]) for i in range(num_view)]
                final_classification = self.C_general(classifications)
                return embeddings, classifications, final_classification
            
        from thop import profile, clever_format
        params = sum([sum(p.numel() for p in model.parameters()) for model in model_dict.values()])
        flops, params = profile(MOGONET().to(device), inputs=(data_all_list, adj_test_list), verbose=False, custom_ops={GraphConvolution: count_gcn_flops}) # return MACs and params; flops = 2 * MACs
        flops, params = clever_format([flops*2, params], "%.3f")
        print(f'Input data shape: {data_all_list[0].shape}, {data_all_list[1].shape}, {data_all_list[2].shape}')
        print(f'Input graph shape: {adj_test_list[0].shape}, {adj_test_list[1].shape}, {adj_test_list[2].shape}')
        print(f"FLOPs: {flops}, Params: {params}")
        print(f"Params by manual calculation: {sum(p.numel() for p in MOGONET().parameters())}")
        exit()
        
    # PreTraining Stage.
    print('Pretrain GCNs...')
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(list(model_dict["E{:}".format(i+1)].parameters()) + list(model_dict["C{:}".format(i+1)].parameters()), lr=lr_e_pretrain)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for epoch in tqdm(range(num_epoch_pretrain), ascii=True):
        for m in model_dict:
            model_dict[m].train()   
        loss_dict = {}
        for i in range(num_view):
            optim_dict["C{:}".format(i+1)].zero_grad()
            output_train = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_train_list[i], adj_train_list[i]))
            loss_train = torch.mean(torch.mul(criterion(output_train, label_train), sample_weight_train))
            loss_train.backward()
            optim_dict["C{:}".format(i+1)].step()
            loss_dict["C{:}".format(i+1)] = loss_train.detach().cpu().item()
        
    # Training Stage.
    print('Training......')
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)] = torch.optim.Adam(list(model_dict["E{:}".format(i+1)].parameters())+list(model_dict["C{:}".format(i+1)].parameters()), lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for epoch in range(num_epoch+1):
        for m in model_dict:
            model_dict[m].train()   
        loss_dict = {}
        for i in range(num_view):
            optim_dict["C{:}".format(i+1)].zero_grad()
            output_train = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_train_list[i], adj_train_list[i]))
            loss_train = torch.mean(torch.mul(criterion(output_train, label_train), sample_weight_train))
            loss_train.backward()
            optim_dict["C{:}".format(i+1)].step()
            loss_dict["C{:}".format(i+1)] = loss_train.detach().cpu().item()
        if num_view >= 2:
            optim_dict["C"].zero_grad()
            output_train_list = []
            for i in range(num_view):
                output_train_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_train_list[i], adj_train_list[i])))
            output_train = model_dict["C"](output_train_list)    
            loss_train = torch.mean(torch.mul(criterion(output_train, label_train), sample_weight_train))
            loss_train.backward()
            optim_dict["C"].step()
            loss_dict["C"] = loss_train.detach().cpu().item()
            
        for m in model_dict:
            model_dict[m].eval()
        if epoch % 50 == 0:
            num_train = len(data_train_list[0])
            output_test_list = []
            for i in range(num_view):
                output_test_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_all_list[i], adj_test_list[i])))
            output_test = model_dict["C"](output_test_list) if num_view >=2 else output_test_list[0]
            output_test = output_test[num_train:,:]
            prob_test = F.softmax(output_test, dim=1).detach().cpu().numpy()
            
            if num_class == 2:
                acc = accuracy_score(label_test, prob_test.argmax(1)); f1 = f1_score(label_test, prob_test.argmax(1)); auc = roc_auc_score(label_test, prob_test[:, 1])
                print('Training Epoch {:d}: Test ACC={:.5f}, Test F1={:.5f}, Test AUC={:.5f}'.format(epoch, acc, f1, auc))
            else:
                acc = accuracy_score(label_test, prob_test.argmax(1)); f1_weighted = f1_score(label_test, prob_test.argmax(1), average='weighted'); f1_macro = f1_score(label_test, prob_test.argmax(1), average='macro')
                print('Training Epoch {:d}: Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
    
    save_model_dict(args.output_dir, model_dict)
                