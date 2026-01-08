import os, math, time, numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spectralnet import SpectralNet
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from scipy.optimize import linear_sum_assignment
import muon as mu
import scanpy as sc

def normalization(input_):
    sampleMean = torch.mean(input_, dim=1).view(input_.shape[0], 1)
    sampleStd = torch.std(input_, dim=1).view(input_.shape[0], 1)
    input_ = (input_ - sampleMean) / (sampleStd + 1e-10)
    return input_

def load_data():
    data = mu.read_h5mu("data/Chen_high.h5mu.gz")
    sc.pp.normalize_total(data["rna"], target_sum=1e4)
    sc.pp.log1p(data["rna"])
    mu.atac.pp.tfidf(data['atac'])
    # sc.pp.normalize_total(data["premRNA"], target_sum=1e4)
    # sc.pp.log1p(data["premRNA"])
    # sc.pp.normalize_total(data["mRNA"], target_sum=1e4)
    # sc.pp.log1p(data["mRNA"])
    # mu.prot.pp.clr(data["adt"])
    X = []
    X.append(normalization(torch.tensor(data['rna'].X.todense()).float()))
    X.append(normalization(torch.tensor(data['atac'].X.todense()).float()))
    # X.append(normalization(torch.tensor(data['premRNA'].X.todense()).float()))
    # X.append(normalization(torch.tensor(data['mRNA'].X.todense()).float()))
    # X.append(normalization(torch.tensor(data['adt'].X.todense()).float()))
    label = (data['rna'].obs['cell_type'].cat.codes).to_numpy()
    return X, label, len(pd.unique(label))

class Cell(Dataset):
    def __init__(self, data, labels=None, GMM_labels=None):
        self.num_views = len(data)
        self.data = data
        self.labels = labels
        self.GMM_labels = GMM_labels

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        if self.labels is None:
            return [x[idx] for x in self.data]
        elif self.GMM_labels is None:
            return [x[idx] for x in self.data], torch.from_numpy(self.labels)[idx]
        else:
            return [x[idx] for x in self.data], torch.from_numpy(self.labels)[idx], self.GMM_labels[idx]

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3):
        self.patience = patience
        self.counter = 0
        self.last_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, loss):
        if self.last_loss is None:
            self.last_loss = loss
        # elif np.abs(loss - self.last_loss) / self.last_loss < self.delta :
        #     self.counter += 1
        #     self.last_loss = loss
        #     print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        elif loss > self.last_loss - self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.last_loss = loss
            self.counter = 0
            
def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / Y_pred.size, w

def purity(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]

def to_numpy(x):
    if x.is_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()

def cal_neighbors_S(S, K):
    ind = torch.argsort(S, dim=1, descending=True)
    neighbors = ind[:, 0:K + 1].view(-1, K + 1)
    return neighbors.cpu()

def cal_neighbors_D(D, K):
    n = D.shape[0]
    neighbors = torch.from_numpy(np.zeros((n, K + 1)))
    for idx in range(n):
        ind = torch.argsort(D[idx, :])
        neighbors[idx] = ind[0:K + 1].view(1, K + 1)
    neighbors[:, 0] = torch.arange(n)
    
    return neighbors.type(torch.long)

def form_data(data, label, neighbors):
    form_data = []
    for i, data_v in enumerate(data):
        K = neighbors.shape[1]
        form_data_v = []
        for idx in range(K):
            data_temp = data_v[neighbors[:, idx]].unsqueeze(2)
            form_data_v.append(data_temp)
        form_data.append(torch.cat(form_data_v, dim=-1))
    for idx in range(neighbors.shape[1]):
        acc, _ = cluster_acc(label[neighbors[:, idx]], label)
        print(acc)
    return form_data

class vae(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(vae, self).__init__()
        self.input_dim = layer_sizes[0]
        self.out_dim = layer_sizes[-1]
        self.depth = len(layer_sizes) - 1
        encoder = []
        decoder = []
        for i in range(self.depth - 1):
            encoder.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder.append(nn.BatchNorm1d(layer_sizes[i+1], affine=True))
            encoder.append(activation)
        for i in range(self.depth - 1):
            decoder.append(nn.Linear(layer_sizes[self.depth-i], layer_sizes[self.depth-i-1]))
            decoder.append(nn.BatchNorm1d(layer_sizes[self.depth-i-1], affine=True))
            decoder.append(activation)
        decoder.append(nn.Linear(layer_sizes[1], layer_sizes[0]))
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.mean = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.logvar = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def get_latent(self, inputs):
        h = self.encoder(inputs)
        mean = self.mean(h)
        logvar = self.logvar(h)
        return mean, logvar
    
    def get_recon(self, inputs):
        recon = self.decoder(inputs)
        return recon

class Multiview_VAE(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(Multiview_VAE, self).__init__()
        self.w = nn.Parameter(torch.ones(len(layer_sizes)) / len(layer_sizes))
        self.vaes = nn.ModuleList([vae(layer_size, activation) for layer_size in layer_sizes])

    def get_latent(self, inputs):
        x_mean = 0
        x_var = 0
        w = torch.exp(self.w) / torch.sum(torch.exp(self.w))
        for view in range(len(inputs)):
            mean, logvar = self.vaes[view].get_latent(inputs[view])
            x_mean += mean * w[view]
            x_var += torch.pow(torch.exp(0.5 * logvar) * w[view], 2)
        return x_mean, torch.log(x_var)

    def get_recon(self, inputs):
        recon = []
        for view in range(len(self.vaes)):
            data = self.vaes[view].get_recon(inputs)
            recon.append(data)
        return recon
    
class Classifier(nn.Module):
    def __init__(self,layer_sizes,activation=nn.ReLU()):
        super(Classifier,self).__init__()
        n_layer = len(layer_sizes)
        layers = []
        for idx in range(n_layer-1):
            layer = nn.Linear(layer_sizes[idx],layer_sizes[idx+1])
            nn.init.xavier_normal_(layer.weight)
            # layer.bias.data = torch.zeros(layer_sizes[idx + 1])
            layers.append(layer)
            if idx < n_layer-2:
                layers.append(activation)
            else:
                layers.append(nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)

    def forward(self,inputs):
        output = self.model(inputs)
        return output

class GMM_Model(nn.Module):
    def __init__(self, N, K, mean=None, var=None):
        super(GMM_Model, self).__init__()
        if mean is not None:
            self.mean = nn.Parameter(mean.view(1, N, K))
            self.std = nn.Parameter(torch.sqrt(var).view(1, N, K))
        else:
            self.mean = nn.Parameter(torch.randn(1, N, K))
            self.std = nn.Parameter(torch.ones(1, N, K))
        self.N = N # dim
        self.K = K # label

    def compute_prob(self, data):
        prob = torch.exp(torch.sum(-torch.log((self.std ** 2) * 2 * math.pi) - torch.div(torch.pow(data.view(-1, self.N, 1) - self.mean, 2), self.std ** 2), dim=1) * 0.5)
        pc = torch.div(prob, (torch.sum(prob, dim=-1)).view(-1, 1) + 1e-10)
        return pc

    def log_prob(self, data_mean, data_logvar, cond_prob, weight):
        # term1 = torch.sum(-torch.log((self.std ** 2) * 2 * math.pi), dim=1) * 0.5
        term1 = torch.sum(-torch.log(self.std ** 2), dim=1) * 0.5
        term2 = torch.sum(-torch.div(
            torch.pow(data_mean.view(-1, self.N, 1) - self.mean, 2) + torch.exp(data_logvar).view(-1, self.N, 1),
            self.std ** 2), dim=1) * 0.5
        prob = term2 + term1
        log_p1 = torch.sum(torch.mul(prob, cond_prob), dim=-1)
        log_p = torch.sum(torch.mul(log_p1, weight))
        return log_p

def gen_x(mean, std, J):
    x_samples = []
    v_size = mean.size()
    for idx in range(J):
        x_samples.append(mean + torch.mul(std, torch.randn(v_size).cuda()))
    return x_samples

def compute_weight(inputs, similarity_type='Gauss'):
    dist = 0
    for input_ in inputs:
        dist += torch.sum(torch.pow(input_ - input_[:, :, 0].unsqueeze(2), 2), dim=1)
    dist = F.normalize(dist, dim=1)
    if similarity_type == 'Gauss':
        Gauss_simi = torch.exp(-dist)
        if Gauss_simi.shape[1] == 1:
            Gauss_simi[:, 0] = 1
        else:
            Gauss_simi[:, 0] = torch.sum(Gauss_simi[:, 1:], dim=1)
        simi = torch.div(Gauss_simi, torch.sum(Gauss_simi, dim=1, keepdim=True))
    else:
        N = inputs[0].size(-1)
        simi = torch.ones(1, N) / (N - 1)
        simi[0, 0] = 1
        simi = torch.mul(torch.ones(inputs[0].size(0), 1), simi)
        simi = torch.div(simi, torch.sum(simi, dim=1, keepdim=True))
    return simi

def train_model(vae, classifier, GMM, optimizer, lr_scheduler, dataloader, dataname, epoch_num, device):
    vae = vae.to(device)
    classifier = classifier.to(device)
    GMM = GMM.to(device)
    avg_loss = []
    early_stopping = EarlyStopping(patience=10, delta=1e-3)
    t = trange(epoch_num, leave=True)
    for epoch in t:
        Total_loss = []
        label = []
        label_pred = []
        Recon_loss = []
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            N_samples = inputs[0].size(0)
            N_n = inputs[0].size(2)
            inputs = [input_.to(device) for input_ in inputs]
            all_inputs = []
            all_targets = []
            for data_v in inputs:
                temp_data1 = []
                temp_data2 = []
                for idx in range(N_n):
                    temp_data1.append(data_v[:, :, idx])
                    temp_data2.append(data_v[:, :, 0])
                temp_data1 = torch.cat(temp_data1, dim=0)
                temp_data2 = torch.cat(temp_data2, dim=0)
                all_inputs.append(temp_data1)
                all_targets.append(temp_data2)
            label.append(labels)
            weight = compute_weight(inputs)
            weight_temp = []
            for idx in range(N_n):
                weight_temp.append(weight[:, idx])
            weight = torch.cat(weight_temp)
            # Compute the prior of c
            vae.eval()
            classifier.eval()
            with torch.no_grad():
                x_mean, _ = vae.get_latent(all_inputs)
                pc = classifier(x_mean).data
            pc = torch.mul(pc, weight.view(-1, 1))
            pc_temp = 0
            for idx in range(N_n):
                pc_temp = pc_temp + pc[idx * N_samples:(idx + 1) * N_samples, :]
            pc_temp1 = []
            for idx in range(N_n):
                pc_temp1.append(pc_temp)
            pc = torch.cat(pc_temp1, dim=0)
            # Begin training
            vae.train()
            classifier.train()
            GMM.train()
            loss = 0
            J = 1
            x_mean, x_logvar = vae.get_latent(all_inputs)
            x_samples = gen_x(x_mean, torch.exp(0.5 * x_logvar), J)
            ELBO = 0
            for idx in range(J):
                x_re = vae.get_recon(x_samples[idx])
                # x_re = vae.get_recon(torch.cat((x_samples[idx], dbatch), dim=1))
                recon = 0
                for view in range(len(x_re)):
                    recon += -0.5 * torch.sum(torch.pow(all_targets[view] - x_re[view], 2), dim=1)
                ELBO = ELBO + 0.1*torch.sum(torch.mul(recon, weight))
            ELBO = ELBO / J
            Recon_loss.append((-ELBO / N_samples).item())
            cond_prob = classifier(x_samples[0])
            ELBO = ELBO + torch.sum(
                torch.mul(torch.sum(-torch.mul(cond_prob, torch.log(cond_prob + 1e-10)), dim=-1), weight))
            ELBO = ELBO + torch.sum(
                torch.mul(torch.sum(torch.mul(cond_prob, torch.log(pc + 1e-10)), dim=-1), weight))
            ELBO = ELBO + GMM.log_prob(x_mean, x_logvar, cond_prob, weight)
            ELBO = ELBO + torch.sum(torch.mul(0.5 * torch.sum(x_logvar, dim=-1), weight))
            loss = loss - ELBO / N_samples
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Total_loss.append(loss.item())
            if np.isnan(loss.item()):
                return vae, classifier, GMM, avg_loss
            vae.eval()
            classifier.eval()
            data_i = []
            for data_v in inputs:
                data_i.append(data_v[:, :, 0].data)
            with torch.no_grad():
                x_mean, _ = vae.get_latent(data_i)
                pred_l = torch.max(classifier(x_mean), dim=-1)
            label_pred.append(pred_l[-1])
        lr_scheduler.step()
        label = torch.cat(label, dim=0)
        label_pred = torch.cat(label_pred, dim=0)
        NMI = normalized_mutual_info_score(to_numpy(label), to_numpy(label_pred))
        ARI = adjusted_rand_score(to_numpy(label), to_numpy(label_pred))
        pur = purity(to_numpy(label), to_numpy(label_pred))
        acc, _ = cluster_acc(to_numpy(label_pred), to_numpy(label))
        avg_loss.append(np.mean(Total_loss))
        t.set_description('|Epoch:{} Total loss={:3f} Reconstruction Loss={:6f} PUR={:5f} ARI={:5f} NMI={:5f} ACC={:6f}'.format(
                epoch + 1, np.mean(Total_loss), np.mean(Recon_loss), pur, ARI, NMI, acc))
        t.refresh()
        early_stopping(np.mean(Total_loss))
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return vae, classifier, GMM, avg_loss

def train_opt(data, label, dataname):
    if data[0].shape[0] > 1024:
        batch_size = 128
    else:
        batch_size = 16
    learning_rate = 0.0001
    weight_decay = 1e-6
    step_size = 50
    gama = 0.1
    epoch = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Load pretrained model")
    vae = torch.load('Intermediate_data/' + dataname + '/pretrained_vae.pkl')
    classifier = torch.load('Intermediate_data/' + dataname + '/pretrained_classifier.pkl')
    GMM = torch.load('Intermediate_data/' + dataname + '/pretrained_GMM.pkl')
    print("Train the model")
    dataloader = DataLoader(Cell(data, label), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(list(vae.parameters()) + list(classifier.parameters()) + list(GMM.parameters()), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gama)
    vae, classifier, GMM, avg_loss = train_model(vae, classifier, GMM, optimizer, lr_scheduler, dataloader, dataname, epoch, device)
    vae.eval()
    classifier.eval()
    GMM.eval()
    datai = [data_[:, :, 0] for data_ in data]
    dataloader = DataLoader(Cell(datai), batch_size=1024, shuffle=False)
    x_mean = []
    x_logvar = []
    for batch_idx, (inputs) in enumerate(dataloader):
        inputs = [input_.to(device) for input_ in inputs]
        with torch.no_grad():
            x_m, x_v = vae.get_latent(inputs)
        x_mean.append(x_m)
        x_logvar.append(x_v)
    x_mean = torch.cat(x_mean, dim=0)
    x_logvar = torch.cat(x_logvar, dim=0)
    x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar), 1)[0]
    with torch.no_grad():
        pred_label = torch.max(classifier(x_sample), dim=-1)
    acc, _ = cluster_acc(to_numpy(pred_label[-1]), label)
    NMI = normalized_mutual_info_score(label, to_numpy(pred_label[-1]))
    ARI = adjusted_rand_score(label, to_numpy(pred_label[-1]))
    pur = purity(label, to_numpy(pred_label[-1]))
    FMI = fowlkes_mallows_score(label, to_numpy(pred_label[-1]))    
    vae = vae.to('cpu')
    classifier = classifier.to('cpu')
    GMM = GMM.to('cpu')
    torch.save(vae, 'result/' + dataname + '/trained_vae.pkl')
    torch.save(classifier, 'result/' + dataname + '/trained_classifier.pkl')
    torch.save(GMM, 'result/' + dataname + '/trained_GMM.pkl')
    return acc, NMI, ARI, pur, FMI

def gen_x(mean, std):
    v_size = mean.size()
    x_samples = mean + torch.mul(std, torch.randn(v_size).cuda())
    return x_samples

def pretrain_vae(vae, optimizer, lr_scheduler, dataloader, epoch_num, device):
    vae = vae.to(device)
    avg_loss = []
    t = trange(epoch_num, leave=True)
    for epoch in t:
        Total_loss = []
        vae.train()
        for batch_idx, (inputs) in enumerate(dataloader):
            inputs = [input_.to(device) for input_ in inputs]
            x_mean, x_logvar = vae.get_latent(inputs)
            x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar))
            # x_sample = torch.cat((x_sample, dbatch), dim=1)
            ELBO_rec = 0
            x_re = vae.get_recon(x_sample)
            for view in range(len(x_re)):
                ELBO_rec += 0.5 * F.mse_loss(inputs[view], x_re[view])
            loss = ELBO_rec
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Total_loss.append(loss.item())
            if np.isnan(loss.item()):
                return vae, avg_loss
        lr_scheduler.step()
        t.set_description('|Epoch:{} Total loss={:3f}'.format(epoch, np.mean(Total_loss)))
        t.refresh()
        avg_loss.append(np.mean(Total_loss))
    return vae, avg_loss

def pretrain_classifier(vae, classifier, optimizer, lr_scheduler, dataloader, epoch_num, device):
    vae.eval()
    classifier = classifier.to(device)
    classifier.train()
    avg_losses = []
    loss_f = nn.NLLLoss()
    t = trange(epoch_num, leave=True)
    for epoch in t:
        Total_loss = []
        label = []
        label_pred = []
        for batch_idx, (inputs, labels, label_train) in enumerate(dataloader):
            inputs = [input_.to(device) for input_ in inputs]
            label_train = label_train.to(device)
            label.append(labels)
            with torch.no_grad():
                x_mean, x_logvar = vae.get_latent(inputs)
            x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar))
            cond_prob = classifier(x_sample)
            loss = loss_f(torch.log(cond_prob + 1e-10), label_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Total_loss.append(loss.item())
            pred_l = torch.max(cond_prob, dim=-1)
            label_pred.append(pred_l[-1])
            if np.isnan(loss.item()):
                return classifier, avg_losses
        lr_scheduler.step()
        label = torch.cat(label, dim=0)
        label_pred = torch.cat(label_pred, dim=0)
        acc, _ = cluster_acc(to_numpy(label_pred), to_numpy(label))
        t.set_description('|Epoch:{} Total loss={:3f} ACC={:5f}'.format(epoch + 1, np.mean(Total_loss), acc))
        t.refresh()
        avg_losses.append(np.mean(Total_loss))
    return classifier, avg_losses

def pretrain_opt(data, label, c, dataname, train_vae = True):
    if data[0].shape[0] > 1024:
        batch_size = 128
    else:
        batch_size = 16
    learning_rate = 0.001
    weight_decay = 1e-6
    step_size = 50
    gama = 0.1
    epoch = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_vae:
        activation = nn.ReLU()
        encoder_sizes = []
        for _, data_v in enumerate(data):
            encoder_sizes.append([data_v.shape[1], 256, c])
        vae = Multiview_VAE(encoder_sizes, activation=activation)
        dataloader = DataLoader(Cell(data), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gama)
        print("pretrain the VAE model")
        vae, avg_loss = pretrain_vae(vae, optimizer, lr_scheduler, dataloader, epoch, device)
        vae.eval()
        dataloader = DataLoader(Cell(data), batch_size=1024, shuffle=False)
        x_mean = []
        for batch_idx, (inputs) in enumerate(dataloader):
            inputs = [input_.to(device) for input_ in inputs]
            with torch.no_grad():
                x_m, _ = vae.get_latent(inputs)
            x_mean.append(x_m)
        x_mean = torch.cat(x_mean, dim=0)
        print("| Latent range: {}/{}".format(x_mean.min(), x_mean.max()))
        kmeans = KMeans(n_clusters=c, random_state=0).fit(to_numpy(x_mean))
        cls_index = kmeans.labels_
        mean = kmeans.cluster_centers_
        mean = torch.from_numpy(mean).to(device)
        acc, _ = cluster_acc(cls_index, label)
        NMI = normalized_mutual_info_score(label, cls_index)
        ARI = adjusted_rand_score(label, cls_index)
        pur = purity(label, cls_index)
        FMI = fowlkes_mallows_score(label, cls_index)
        print('| Kmeans ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f}'.format(acc, NMI, ARI, pur, FMI))
        var = []
        for idx in range(c):
            index = np.where(cls_index == idx)
            var_g = torch.sum((x_mean[index[0], :] - mean[idx, :]) ** 2, dim=0, keepdim=True) / (len(index[0]) - 1)
            var.append(var_g)
        var = torch.cat(var, dim=0)
        GMM = GMM_Model(c, c, mean.t(), var.t())
        GMM = GMM.to(device)
        GMM.eval()
        label_pred = torch.max(GMM.compute_prob(x_mean), dim=-1)
        acc, _ = cluster_acc(to_numpy(label_pred[-1]), label)
        NMI = normalized_mutual_info_score(label, to_numpy(label_pred[-1]))
        ARI = adjusted_rand_score(label, to_numpy(label_pred[-1]))
        pur = purity(label, to_numpy(label_pred[-1]))
        FMI = fowlkes_mallows_score(label, to_numpy(label_pred[-1]))
        print('| GMM ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f}'.format(acc, NMI, ARI, pur, FMI))
        vae = vae.to('cpu')
        GMM = GMM.to('cpu')
        torch.save(vae, 'Intermediate_data/' + dataname + '/pretrained_vae.pkl')
        torch.save(GMM, 'Intermediate_data/' + dataname + '/pretrained_GMM.pkl')
    else:
        vae = torch.load('Intermediate_data/' + dataname + '/pretrained_vae.pkl')
        GMM = torch.load('Intermediate_data/' + dataname + '/pretrained_GMM.pkl')
    vae = vae.to(device)
    GMM = GMM.to(device)
    vae.eval()
    GMM.eval()
    dataloader = DataLoader(Cell(data), batch_size=1024, shuffle=False)
    x_mean = []
    for batch_idx, (inputs) in enumerate(dataloader):
        inputs = [input_.to(device) for input_ in inputs]
        with torch.no_grad():
            x_m, _ = vae.get_latent(inputs)
        x_mean.append(x_m)
    x_mean = torch.cat(x_mean, dim=0)
    with torch.no_grad():
        GMM_label = torch.max(GMM.compute_prob(x_mean), dim=-1)[-1].to('cpu')
    GMM = GMM.to('cpu')
    learning_rate = 0.01
    epoch = 70
    classifier_sizes = [c, c]
    classifier = Classifier(classifier_sizes)
    dataloader = DataLoader(Cell(data, label, GMM_label), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gama)
    print("pretrain the classifier")
    classifier, avg_loss = pretrain_classifier(vae, classifier, optimizer, lr_scheduler, dataloader, epoch, device)
    classifier.eval()
    vae.eval()
    dataloader = DataLoader(Cell(data), batch_size=1024, shuffle=False)
    x_mean = []
    x_logvar = []
    for batch_idx, (inputs) in enumerate(dataloader):
        inputs = [input_.to(device) for input_ in inputs]
        with torch.no_grad():
            x_m, x_v = vae.get_latent(inputs)
        x_mean.append(x_m)
        x_logvar.append(x_v)
    x_mean = torch.cat(x_mean, dim=0)
    x_logvar = torch.cat(x_logvar, dim=0)
    x_sample = gen_x(x_mean, torch.exp(0.5 * x_logvar))
    with torch.no_grad():
        pred_label = torch.max(classifier(x_sample), dim=-1)
    acc, _ = cluster_acc(to_numpy(pred_label[-1]), label)
    NMI = normalized_mutual_info_score(label, to_numpy(pred_label[-1]))
    ARI = adjusted_rand_score(label, to_numpy(pred_label[-1]))
    pur = purity(label, to_numpy(pred_label[-1]))
    FMI = fowlkes_mallows_score(label, to_numpy(pred_label[-1]))
    print('| classifier ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f}'.format(acc, NMI, ARI, pur, FMI))
    vae = vae.to('cpu')
    classifier = classifier.to('cpu')
    torch.save(classifier, 'Intermediate_data/' + dataname + '/pretrained_classifier.pkl')
    del vae, GMM, classifier

if __name__ == '__main__':
    def setup_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    setup_seed(0)
    raw_data, label, c = load_data()
    K = 3
    dataname = 'Chen_high'
    if not os.path.exists('Intermediate_data/' + dataname):
        os.makedirs('Intermediate_data/' + dataname)
    if not os.path.exists('result/' + dataname):
        os.makedirs('result/' + dataname)
    for t in range(1):
        setup_seed(t)
        start = time.perf_counter()
        pretrain_opt(raw_data, label, c, dataname)
        train_graph = True
        if train_graph:
            spectralNet = SpectralNet(n_clusters=c, should_use_ae=True, should_use_siamese=True, ae_hiddens=[512, c], siamese_hiddens=[256, c], spectral_hiddens=[256, c])
            data_cat = torch.cat(raw_data, dim=1)
            spectralNet.fit(data_cat)
            spectralNet_dist = spectralNet.predict(data_cat)  # Get the final assignments to clusters
            torch.save(spectralNet_dist, 'Intermediate_data/' + dataname + '/SpectralNet_D.pkl')
        else:
            spectralNet_dist = torch.load('Intermediate_data/' + dataname + '/SpectralNet_D.pkl')
        neighbors = cal_neighbors_D(spectralNet_dist, K)
        data = form_data(raw_data, label, neighbors)
        acc, NMI, ARI, pur, FMI = train_opt(data, label, dataname)
        end = time.perf_counter()
        runTime = end - start
        print('| ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f} TIME = {:6f}'.format(acc, NMI, ARI, pur, FMI, runTime))
