import os, time, yaml, math, copy
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist

yaml.add_constructor('!join', lambda loader, node: os.path.sep.join(loader.construct_sequence(node))) # add direction join function when parse the yaml file
yaml.add_constructor('!concat', lambda loader, node: ''.join(loader.construct_sequence(node))) # add string concatenation function when parse the yaml file

def construct_hyperedge_with_KNN(X, k_list=[10], is_prob=True, m_prob=1):
    hyperedge_mat_list = []
    for k in k_list:
        distance_mat = cdist(X, X, 'euclid')
        hyperedge_mat = np.zeros(distance_mat.shape)
        for center_idx in range(distance_mat.shape[0]):
            distance_mat[center_idx, center_idx] = 0
            distance_vec = distance_mat[center_idx]
            distance_vec_avg = np.average(distance_vec)
            nearest_idx = np.array(np.argsort(distance_vec)).squeeze()
            nearest_idx[k - 1] = center_idx if not np.any(nearest_idx[:k] == center_idx) else nearest_idx[k - 1] # add the center node to the nearest neighbors if it is not in the top k
            for node_idx in nearest_idx[:k]:
                hyperedge_mat[node_idx, center_idx] = np.exp(-distance_vec[node_idx] ** 2 / (m_prob * distance_vec_avg) ** 2) if is_prob else 1.0
        hyperedge_mat_list.append(hyperedge_mat)
    return np.hstack(hyperedge_mat_list)

def load_feature_construct_H(data_dir, k_list=[10], is_prob=True, m_prob=1, use_mvcnn_feature=False, use_gvcnn_feature=True, use_mvcnn_hyperedge=False, use_gvcnn_hyperedge=True):
    # Construct feature matrix
    data = scio.loadmat(data_dir)
    feature_mvcnn = data['X'][0].item().astype(np.float32) if use_mvcnn_feature else None
    feature_gvcnn = data['X'][1].item().astype(np.float32) if use_gvcnn_feature else None
    label = data['Y'].astype(np.int32); label = label - 1 if label.min() == 1 else label
    idx = data['indices'].item(); idx_train = np.where(idx == 1)[0]; idx_test = np.where(idx == 0)[0]
    feature_all = None
    for feature in [feature_mvcnn, feature_gvcnn]:
        if feature is not None:
            feature_all = feature if feature_all is None else np.hstack((feature_all, feature))
    print('Feature shape:', feature_all.shape, 'Label shape:', label.shape, 'Train shape:', idx_train.shape, 'Test shape:', idx_test.shape)
    # Construct hypergraph incidence matrix
    hyperedge_mvcnn = construct_hyperedge_with_KNN(feature_mvcnn, k_list=k_list, is_prob=is_prob, m_prob=m_prob) if use_mvcnn_hyperedge else None
    hyperedge_gvcnn = construct_hyperedge_with_KNN(feature_gvcnn, k_list=k_list, is_prob=is_prob, m_prob=m_prob) if use_gvcnn_hyperedge else None
    hyperedge_all = None
    for hyperedge in [hyperedge_mvcnn, hyperedge_gvcnn]:
        if hyperedge is not None:
            hyperedge_all = hyperedge if hyperedge_all is None else np.hstack((hyperedge_all, hyperedge))
    print('Hyperedge shape:', hyperedge_all.shape)
    return feature_all, label, idx_train, idx_test, hyperedge_all

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

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.register_parameter('bias', None) if not bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        x = x + self.bias if self.bias is not None else x
        x = G.matmul(x)
        return x
    
class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.dropout(F.relu(self.hgc1(x, G)), self.dropout)
        x = self.hgc2(x, G)
        return x

if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.makedirs(config['result_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = config['modelnet40_ft'] if config['on_dataset'] == 'ModelNet40' else config['ntu2012_ft']
    feature, label, idx_train, idx_test, H = load_feature_construct_H(data_dir, k_list=config['k_list'], is_prob=config['is_prob'], m_prob=config['m_prob'], 
                                                                 use_mvcnn_feature=config['use_mvcnn_feature'], use_gvcnn_feature=config['use_gvcnn_feature'], 
                                                                 use_mvcnn_hyperedge=config['use_mvcnn_hyperedge'], use_gvcnn_hyperedge=config['use_gvcnn_hyperedge'])
    G = generate_G_from_H(H)
    N_class = int(label.max()) + 1
    feature = torch.Tensor(feature).to(device)
    label = torch.Tensor(label).squeeze().long().to(device)
    G = torch.Tensor(G).to(device)
    idx_train = torch.Tensor(idx_train).long().to(device)
    idx_test = torch.Tensor(idx_test).long().to(device)
    print(f"Classification on {config['on_dataset']} dataset and class number {N_class}")
    print(f"use MVCNN feature: {config['use_mvcnn_feature']}", f"use GVCNN feature: {config['use_gvcnn_feature']}")
    print(f"use MVCNN hyperedge: {config['use_mvcnn_hyperedge']}", f"use GVCNN hyperedge: {config['use_gvcnn_hyperedge']}")
    model = HGNN(in_ch=feature.shape[1], n_class=N_class, n_hid=config['n_hid'], dropout=config['drop_out']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']) # optimizer = optim.SGD(model_ft.parameters(), lr=0.01, weight_decay=config['weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
    criterion = torch.nn.CrossEntropyLoss()
    start = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(config['max_epoch']):
        print('Epoch {}/{}'.format(epoch, config['max_epoch'] - 1)) if epoch % config['print_freq'] == 0 else None
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            idx = idx_train if phase == 'train' else idx_test
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(feature, G)
                loss = criterion(outputs[idx], label[idx])
                preds = torch.argmax(outputs, dim=1)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
            epoch_loss = loss.item()
            epoch_acc = torch.sum(preds[idx] == label.data[idx]).double() / len(idx)
            print('{} Loss: {:.4f} ACC: {:.4f}'.format(phase, epoch_loss, epoch_acc)) if epoch % config['print_freq'] == 0 else None
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
        print('Best val ACC: {:4f}'.format(best_acc)) if epoch % config['print_freq'] == 0 else None
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60),'Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model)
    