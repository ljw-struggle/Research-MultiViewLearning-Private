import os, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class MMDynamic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.num_class = num_class
        self.dropout = dropout
        self.FeatureInfoEncoder = nn.ModuleList([nn.Linear(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([nn.Linear(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([nn.Linear(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([nn.Linear(hidden_dim[0], num_class) for _ in range(self.views)])
        self.MMClasifier = []
        assert len(hidden_dim) >= 1, "The length of hidden dim need to be greater than or equal to 1."
        if len(hidden_dim) == 1:
            self.MMClasifier.append(nn.Linear(self.views*hidden_dim[0], num_class))
        else:
            self.MMClasifier.append(nn.Linear(self.views*hidden_dim[0], hidden_dim[1]))
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
            feature = feature * tcp_confidence
            FeatureInfo[view] = featureinfo; Feature[view] = feature
            TCPLogit[view] = tcp_logit; TCPConfidence[view] = tcp_confidence
        MMfeature = torch.cat([i for i in Feature.values()], dim=1)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/BRCA/', help='The output dir.')
    parser.add_argument('-t', '--test_only', default=False, action='store_true', help='Whether to test only.')
    args = parser.parse_args()
    if 'BRCA' in args.data_dir: 
        hidden_dim = [500]; num_epoch = 2500; lr = 1e-4; step_size = 500; num_class = 5
    if 'ROSMAP' in args.data_dir: 
        hidden_dim = [300]; num_epoch = 1000; lr = 1e-4; step_size = 500; num_class = 2
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_train_list, data_test_list, label_train, label_test = prepare_data(args.data_dir)
    dim_list = [data_train.shape[1] for data_train in data_train_list]
    data_train_list = [torch.FloatTensor(data_train).to(device) for data_train in data_train_list]
    data_test_list = [torch.FloatTensor(data_test).to(device) for data_test in data_test_list]
    label_train = torch.LongTensor(label_train).to(device)
    label_test = torch.LongTensor(label_test).to(device)
    model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    if not args.test_only:
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
                    logit, _, _, _, _ = model.forward(data_test_list)
                    prob = F.softmax(logit, dim=1).data.cpu().numpy()
                if 'ROSMAP' in args.data_dir:
                    acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
                    f1 = f1_score(label_test.cpu().numpy(), prob.argmax(1))
                    auc = roc_auc_score(label_test.cpu().numpy(), prob[:,1])
                    print('Training Epoch {:d}: Test ACC={:.5f}, Test F1={:.5f}, Test AUC={:.5f}'.format(epoch, acc, f1, auc))
                if 'BRCA' in args.data_dir:
                    acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
                    f1_weighted = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='weighted')
                    f1_macro = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='macro')
                    print('Training Epoch {:d}: Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(epoch, acc, f1_weighted, f1_macro))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint.pt"))
    else:
        best_checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint.pt'))
        model.load_state_dict(best_checkpoint)
        model.eval()
        with torch.no_grad():
            logit = model.forward(data_test_list)
            prob = F.softmax(logit, dim=1).data.cpu().numpy()
        if 'ROSMAP' in args.data_dir:
            acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
            f1 = f1_score(label_test.cpu().numpy(), prob.argmax(1))
            auc = roc_auc_score(label_test.cpu().numpy(), prob[:,1])
            print('Test ACC={:.5f}, F1={:.5f}, AUC={:.5f}'.format(acc, f1, auc))
        if 'BRCA' in args.data_dir:
            acc = accuracy_score(label_test.cpu().numpy(), prob.argmax(1))
            f1_weighted = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='weighted')
            f1_macro = f1_score(label_test.cpu().numpy(), prob.argmax(1), average='macro')
            print('Test ACC={:.5f}, F1_weighted={:.5f}, F1_macro={:.5f}'.format(acc, f1_weighted, f1_macro))
    