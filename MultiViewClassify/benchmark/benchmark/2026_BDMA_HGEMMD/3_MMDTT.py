import os, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class MMD(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.num_class = num_class
        self.dropout = dropout
        self.FeatureInfoEncoder = nn.ModuleList([nn.Linear(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([nn.Linear(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([nn.Linear(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([nn.Linear(hidden_dim[0], num_class) for _ in range(self.views)])
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim[0], num_heads=10, batch_first=True) # batch_first=True: (batch_size, seq_len, hidden_dim)
        self.MMClasifier = []
        assert len(hidden_dim) >= 1, "The length of hidden dim need to be greater than or equal to 1."
        if len(hidden_dim) == 1:
            self.MMClasifier.append(nn.Linear(hidden_dim[0], num_class))
        else:
            self.MMClasifier.append(nn.Linear(hidden_dim[0], hidden_dim[1]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
            for layer in range(1, len(hidden_dim) -1):
                self.MMClasifier.append(nn.Linear(hidden_dim[layer], hidden_dim[layer+1]))
                self.MMClasifier.append(nn.ReLU())
                self.MMClasifier.append(nn.Dropout(p=dropout))
            self.MMClasifier.append(nn.Linear(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data_list):
        FeatureInfo, Feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            featureinfo = torch.sigmoid(self.FeatureInfoEncoder[view](data_list[view]))
            feature = self.FeatureEncoder[view](data_list[view] * featureinfo)
            feature = F.dropout(F.relu(feature), self.dropout, training=self.training)
            tcp_logit = self.TCPClassifierLayer[view](feature)
            tcp_confidence = self.TCPConfidenceLayer[view](feature)
            feature = feature * tcp_confidence # shape: (batch_size, hidden_dim[0])
            FeatureInfo[view] = featureinfo; Feature[view] = feature
            TCPLogit[view] = tcp_logit; TCPConfidence[view] = tcp_confidence
        # Multi-head attention
        # Q.shape: (batch_size, 1, hidden_dim[0])
        Q = torch.ones(Feature[0].shape[0], 1, Feature[0].shape[1]).to(Feature[0].device)
        K = torch.stack([Feature[view] for view in range(self.views)], dim=1) # shape: (batch_size, views, hidden_dim[0])
        V = torch.stack([Feature[view] for view in range(self.views)], dim=1) # shape: (batch_size, views, hidden_dim[0])
        output = self.mha(Q, K, V)[0] # shape: (batch_size, 1, hidden_dim[0])
        MMfeature = output.squeeze(dim=1) # shape: (batch_size, hidden_dim[0])
        MMlogit = self.MMClasifier(MMfeature)
        return MMlogit, FeatureInfo, Feature, TCPLogit, TCPConfidence
    
    def forward_criterion(self, data_list, label):
        MMlogit, FeatureInfo, Feature, TCPLogit, TCPConfidence = self.forward(data_list)
        criterion = torch.nn.CrossEntropyLoss()
        MMloss = criterion(MMlogit, label)
        for view in range(self.views):
            view_pred = F.softmax(TCPLogit[view], dim=1)
            view_conf = torch.gather(input=view_pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = F.mse_loss(TCPConfidence[view].view(-1), view_conf) + criterion(TCPLogit[view], label)
            sparsity_loss = torch.mean(FeatureInfo[view])
            MMloss = MMloss + confidence_loss + sparsity_loss
        return MMloss

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

def count_mha(module, input, output):
    # print(input[0].shape); print(input[1].shape); print(input[2].shape) # input: (B, L, E) * 3
    # print(output[0].shape); print(output[1].shape) # output, attention: (B, L, E), (B, L, L)
    q, k, v = input[0], input[1], input[2]
    B, L_q, E = q.shape; H = module.num_heads
    L_k, L_v = k.shape[1], v.shape[1]
    flops, params = 0, 0
    # Q, K, V projections: B * L_q * E * E + B * L_k * E * E + B * L_v * E * E
    flops += B * L_q * (E * E) + B * L_k * (E * E) + B * L_v * (E * E) # MACs = 2 flops (multiply + add), there we calculate the MACs
    params += E * E + E
    # Q × K^T: B * H * L_q * L_k * D
    flops += B * H * L_q * L_k * (E // H)  # Q × K^T
    # Softmax: B * H * L_q * L_k * 3 (exp + sum + div)
    flops += B * H * L_q * L_k * 3 # MACs = 2 flops (multiply + add), there we calculate the MACs
    # Attention × V: B * H * L_q * L_v * D
    flops += B * H * L_q * L_v * (E // H)
    # Output projection: B * L * E * E
    flops += B * L_q * (E * E) # MACs = 2 flops (multiply + add), there we calculate the MACs
    params += E * E + E
    module.total_ops += flops
    
custom_ops = {nn.MultiheadAttention: count_mha}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/data_MOGONET/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/data_MOGONET/MMD/BRCA/', help='The output dir.')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('-m', '--modality', default=6, type=int, help='The option for modality missing.') # args.modality: {0:001, 1:010, 2:011, 3:100, 4:101, 5:110, 6:111}, xx1: mRNA = True, x1x: methylation = True, 1xx: miRNA = True
    parser.add_argument('-si', '--sigma', default=0, type=float, help='The standard deviation of the Gaussian noise.')
    parser.add_argument('-v', '--verbose', default=0, type=int, help='The verbose level.')
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
    data_train_list, data_test_list, label_train, label_test = prepare_data(args.data_dir)
    dim_list = [data_train.shape[1] for data_train in data_train_list]
    data_list = [np.concatenate([data_train, data_test], axis=0) for data_train, data_test in zip(data_train_list, data_test_list)]
    # -- noisy condition control
    data_list = [data_list[i] + np.random.normal(0, args.sigma, data_list[i].shape) for i in range(len(data_list))] if args.sigma > 0 else data_list
    # -- missing modality control
    data_list[0] = np.zeros_like(data_list[0]) if not use_mRNA else data_list[0]
    data_list[1] = np.zeros_like(data_list[1]) if not use_meth else data_list[1]
    data_list[2] = np.zeros_like(data_list[2]) if not use_miRNA else data_list[2]
    # -- split data into train and test
    data_train_list = [data_list[i][:len(label_train)] for i in range(len(data_list))]
    data_test_list = [data_list[i][len(label_train):] for i in range(len(data_list))]
    data_train_list = [torch.FloatTensor(data_train).to(device) for data_train in data_train_list]
    data_test_list = [torch.FloatTensor(data_test).to(device) for data_test in data_test_list]
    label_train = torch.LongTensor(label_train).to(device)
    label_test = torch.LongTensor(label_test).to(device)
    model = MMD(dim_list, hidden_dim, num_class, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    
    # -- verbose == 2: print the FLOPs and params
    if args.verbose == 2:
        from thop import profile, clever_format
        data_list = [torch.cat([data_train_list[i], data_test_list[i]], dim=0) for i in range(len(data_train_list))]
        flops, params = profile(model, inputs=(data_list,), verbose=False, custom_ops=custom_ops) # return MACs and params; flops = 2 * MACs
        flops, params = clever_format([flops*2, params], "%.3f")
        print(f'Input data shape: {data_list[0].shape}, {data_list[1].shape}, {data_list[2].shape}')
        print(f"FLOPs: {flops}, Params: {params}")
        print(f"Params by manual calculation: {sum(p.numel() for p in model.parameters())}")
        exit()
    
    for epoch in range(1, num_epoch + 1):
        model.train()
        optimizer.zero_grad()
        loss = model.forward_criterion(data_train_list, label_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 50 == 0:
            print('Training Epoch {:d}: Loss={:.5f}'.format(epoch, loss.cpu().detach().numpy()))
            model.eval()
            with torch.no_grad():
                logit, _, _, _, _ = model.forward(data_train_list)
                prob = F.softmax(logit, dim=1).data.cpu().numpy()
            if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
                acc = accuracy_score(label_train.cpu().numpy(), prob.argmax(1))
                f1 = f1_score(label_train.cpu().numpy(), prob.argmax(1))
                auc = roc_auc_score(label_train.cpu().numpy(), prob[:,1])
                print('Training Epoch {:d}: Train ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(epoch, acc, f1, auc))
            if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
                acc = accuracy_score(label_train.cpu().numpy(), prob.argmax(1))
                f1_weighted = f1_score(label_train.cpu().numpy(), prob.argmax(1), average='weighted')
                f1_macro = f1_score(label_train.cpu().numpy(), prob.argmax(1), average='macro')
                print('Training Epoch {:d}: Train ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
            if 'TCGA-23' in args.data_dir:
                acc = accuracy_score(label_train.cpu().numpy(), prob.argmax(1))
                f1_weighted = f1_score(label_train.cpu().numpy(), prob.argmax(1), average='weighted')
                f1_macro = f1_score(label_train.cpu().numpy(), prob.argmax(1), average='macro')
                ap = average_precision_score(np.eye(num_class)[label_train.cpu().numpy()], prob, average='macro')
                auc = roc_auc_score(np.eye(num_class)[label_train.cpu().numpy()], prob, average='macro')
                print('Training Epoch {:d}: Train ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}, macro_AUPR={:.5f}, macro_AUC={:.5f}'.format(epoch, acc, f1_weighted, f1_macro, ap, auc))
            with torch.no_grad():
                logit, _, _, _, _ = model.forward(data_test_list)
                prob = F.softmax(logit, dim=1).data.cpu().numpy()
            if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
                f1 = f1_score(label_test.cpu().numpy(), prob.argmax(1))
                auc = roc_auc_score(label_test.cpu().numpy(), prob[:,1])
                print('Training Epoch {:d}: Test ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(epoch, acc, f1, auc))
            if 'BRCA' in args.data_dir or 'KIPAN' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='macro')
                print('Training Epoch {:d}: Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
            if 'TCGA-23' in args.data_dir:
                acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
                f1_weighted = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='weighted')
                f1_macro = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='macro')
                ap = average_precision_score(np.eye(num_class)[label_test.cpu().numpy()], prob, average='macro')
                auc = roc_auc_score(np.eye(num_class)[label_test.cpu().numpy()], prob, average='macro')
                print('Training Epoch {:d}: Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}, macro_AUPR={:.5f}, macro_AUC={:.5f}'.format(epoch, acc, f1_weighted, f1_macro, ap, auc))

    model.eval()
    with torch.no_grad():
        logit, _, _, _, _ = model.forward(data_test_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    if 'ROSMAP' in args.data_dir or 'LGG' in args.data_dir:
        acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
        f1 = f1_score(label_test.cpu().numpy(), prob.argmax(1))
        auc = roc_auc_score(label_test.cpu().numpy(), prob[:,1])
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
        acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
        f1_weighted = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='weighted')
        f1_macro = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='macro')
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
        acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
        f1_weighted = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='weighted')
        f1_macro = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='macro')
        ap = average_precision_score(np.eye(num_class)[label_test.cpu().numpy()], prob, average='macro')
        auc = roc_auc_score(np.eye(num_class)[label_test.cpu().numpy()], prob, average='macro')
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
                