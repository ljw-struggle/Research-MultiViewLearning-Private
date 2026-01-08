import os, random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from _utils import load_data, evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim))

    def forward(self, x):
        return self.decoder(x)

class SCMVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim):
        super(SCMVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_fusion_module = nn.Sequential(nn.Linear(view * feature_dim, 256), nn.ReLU(), nn.Linear(256, high_feature_dim))
        self.common_information_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.view = view

    def forward(self, xs, zs_gradient=True):
        rs = []; zs = []; hs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            r = self.decoders[v](z)
            h = F.normalize(self.common_information_module(z),dim=1)
            hs.append(h); zs.append(z); rs.append(r)
        H = torch.cat(zs, dim=1) if zs_gradient else torch.cat(zs, dim=1).detach()
        H = F.normalize(self.feature_fusion_module(H), dim=1)
        return rs, zs, hs, H

class MVCLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(MVCLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

    def forward(self, h_i, h_j, weight=None):
        N =self.batch_size
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        exp_similarity_matrix = torch.exp(similarity_matrix)
        positive = torch.diag(exp_similarity_matrix) # shape: (N,), the exponential of the similarity between the positive samples (nominator)
        negative_mask = ~torch.diag(torch.ones(N), 0).bool().to(self.device)
        negative = negative_mask * exp_similarity_matrix # shape: (N, N), the exponential of the similarity between the negative samples (denominator)
        loss = -torch.log(positive / torch.sum(negative, dim=1)) # shape: (N,), infoNCE loss
        loss = loss.sum() / N # shape: (1,), average infoNCE loss
        loss = weight * loss if weight is not None else loss # shape: (1,), weighted infoNCE loss
        return loss

def validation(model, dataset, view, data_size, class_num, eval_multi_view=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        rs, zs, hs, H = model(x_list)
    zs = [z.detach().cpu().numpy() for z in zs] # shape: (batch_size, feature_dim) * view
    hs = [h.detach().cpu().numpy() for h in hs] # shape: (batch_size, feature_dim) * view
    H = H.detach().cpu().numpy() # shape: (batch_size, high_feature_dim)
    if eval_multi_view:
        print("Clustering results on low-level features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(zs[v])
            nmi, ari, acc, pur = evaluate(y, y_pred)
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v + 1, acc, nmi, ari, pur))
        print("Clustering results on view-consensus features of each view:")
        for v in range(view):
            y_pred = kmeans.fit_predict(hs[v])
            nmi, ari, acc, pur = evaluate(y, y_pred)
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v + 1, acc, nmi, ari, pur))
    # Clustering results on global features
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(H)
    nmi, ari, acc, pur = evaluate(y, y_pred)
    print('Overall, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SCMVC')
    parser.add_argument('--dataset', default='MNIST-USPS', choices=['MNIST-USPS', 'BDGP', 'CCV', 'Fashion', 'Caltech-2V', 'Caltech-3V', 'Caltech-4V', 'Caltech-5V', 'Cifar10', 'Cifar100', 'Prokaryotic', 'Synthetic3d'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--pre_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=50, type=int)
    parser.add_argument("--feature_dim", default=64, type=int)
    parser.add_argument("--high_feature_dim", default=20, type=int)
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()
    os.makedirs('./result', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset; batch_size = args.batch_size; learning_rate = args.learning_rate; weight_decay = args.weight_decay; 
    pre_epochs = args.pre_epochs; con_epochs = args.con_epochs; feature_dim = args.feature_dim; high_feature_dim = args.high_feature_dim; temperature = args.temperature; seed = args.seed
    seed_dict = {"MNIST-USPS": 10, "BDGP": 30, "CCV": 100, "Fashion": 10, "Caltech-2V": 200, "Caltech-3V": 30, "Caltech-4V": 100, "Caltech-5V": 1000000, "Cifar10": 10, "Cifar100": 10, "Prokaryotic": 10000, "Synthetic3d": 100}
    con_epochs_dict = {"MNIST-USPS": 50, "BDGP": 10, "CCV": 50, "Fashion": 50, "Caltech-2V": 100, "Caltech-3V": 100, "Caltech-4V": 100, "Caltech-5V": 100, "Cifar10": 10, "Cifar100": 20, "Prokaryotic": 20, "Synthetic3d": 100}
    con_epochs = con_epochs_dict[dataset_name]; seed = seed_dict[dataset_name]

    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = SCMVC(view, dims, feature_dim, high_feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion_mse = torch.nn.MSELoss()
    criterion_mvc = MVCLoss(batch_size, temperature, device)
    
    ## 3. Train the model.
    best_acc, best_nmi, best_pur = 0, 0, 0
    for epoch in range(pre_epochs):
        model.train()
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            rs, _, _, _ = model(x_list)
            loss = sum([criterion_mse(x_list[v], rs[v]) for v in range(view)])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Reconstruction Pre-training Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(losses)))
    # acc, nmi, pur = validation(model, dataset, view, data_size, class_num, eval_multi_view=True)
    for epoch in range(con_epochs):
        model.train()
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            rs, zs, hs, H = model(x_list)
            loss_list = []
            with torch.no_grad():
                w = [] # shape: (view,), adaptive weights for each view
                global_sim = torch.matmul(H,H.t()) # shape: (batch_size, batch_size), similarity matrix of the global features
                for v in range(view):
                    view_sim = torch.matmul(hs[v], hs[v].t()) # shape: (batch_size, batch_size), similarity matrix of the view features
                    related_sim = torch.matmul(hs[v], H.t()) # shape: (batch_size, batch_size), similarity matrix of the view features and the global features
                    w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (batch_size * batch_size) # The implementation of MMD
                    w.append(torch.exp(-w_v)) # shape: (view,), adaptive weights for each view
                w = torch.stack(w) # shape: (view,), adaptive weights for each view
                w = w / torch.sum(w) # shape: (view,), adaptive weights for each view (the more similar the view features are to the global features, the larger the weight)
            for v in range(view):
                loss_list.append(criterion_mse(x_list[v], rs[v])) # Reconstruction loss
                loss_list.append(criterion_mvc(H, hs[v], w[v])) # Self-weighted contrastive learning loss, shape: (1,) (the more similar the view features are to the global features, the larger the loss)
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Contrastive Learning Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(losses)))
        nmi, ari, acc, pur = validation(model, dataset, view, data_size, class_num, eval_multi_view=False)
        if acc > best_acc:
            best_epoch = epoch
            best_nmi, best_ari, best_acc, best_pur = nmi, ari, acc, pur
            torch.save(model.state_dict(), './result/' + dataset_name + '.pth')
    print('The best clustering performace (epoch {}): NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}'.format(best_epoch, best_nmi, best_ari, best_acc, best_pur))
    