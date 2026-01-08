import os, random, math, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from itertools import combinations
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

class MFLVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(MFLVC, self).__init__()
        # self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim) for v in range(view)])
        # self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim) for v in range(view)])
        # self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        # self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        # self.view = view
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        self.view = view

    def forward(self, x_list):
        h_list = []; z_list = []; q_list = []; r_list = []
        for v in range(self.view):
            z = self.encoders[v](x_list[v])
            h = F.normalize(self.feature_contrastive_module(z), dim=1) # L2 normalization
            q = self.label_contrastive_module(z)
            r = self.decoders[v](z)
            z_list.append(z); h_list.append(h); q_list.append(q); r_list.append(r)
        return z_list, h_list, q_list, r_list

class MVCLLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(MVCLLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2) # input two vectors, output the cosine similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # input logits and labels, output the loss

    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask

    def forward_feature(self, h_i, h_j):
        # Compute the feature contrastive loss: 1/2*(L_fc^(i,j) + L_fc^(j,i))
        # h_i and h_j are the feature vectors of the positive samples, shape: (batch_size, feature_dim) unit vectors
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0) # shape: (N, feature_dim), h is the concatenation of h_i and h_j
        sim = torch.matmul(h, h.T) / self.temperature_f # shape: (N, N), sim is the cosine similarity matrix
        sim_i_j = torch.diag(sim, diagonal=self.batch_size) # shape: (N//2), sim_i_j is the cosine similarity between h_i and h_j
        sim_j_i = torch.diag(sim, diagonal=-self.batch_size) # shape: (N//2), sim_j_i is the cosine similarity between h_j and h_i
        sim_of_positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive samples
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative samples
        sim_of_negative_samples = sim[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative samples
        labels = torch.zeros(N).to(self.device).long() # shape: (N,), labels is the labels for the positive samples
        logits = torch.cat((sim_of_positive_samples, sim_of_negative_samples), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative samples
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for logits and labels
        loss /= N # shape: (1,), loss is the average loss for logits and labels, N is the number of samples
        return loss

    def forward_label(self, q_i, q_j):
        # Compute the label contrastive loss: 1/2*(L_lc^(i,j) + L_lc^(j,i))
        # q_i and q_j are the cluster assignments of the positive samples, shape: (batch_size, class_num)
        p_i = q_i.sum(0).view(-1) # shape: (class_num,), p_i is the probability of the cluster assignments of the view i
        p_i /= p_i.sum() # shape: (class_num,), p_i is the normalized probability of the cluster assignments of the view i
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum() # shape: (1,), ne_i is the negative information entropy of the cluster assignments of the view i
        p_j = q_j.sum(0).view(-1) # shape: (class_num,), p_j is the probability of the cluster assignments of the view j
        p_j /= p_j.sum() # shape: (class_num,), p_j is the normalized probability of the cluster assignments of the view j
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum() # shape: (1,), ne_j is the negative information entropy of the cluster assignments of the view j
        entropy = ne_i + ne_j # shape: (1,), entropy is the negative information entropy of the cluster assignments of the view i and view j

        N = 2 * self.class_num
        q = torch.cat((q_i.t(), q_j.t()), dim=0) # shape: (N, batch_size), q is the concatenation of q_i and q_j
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l # shape: (2*class_num, 2*class_num), sim is the cosine similarity matrix
        sim_i_j = torch.diag(sim, diagonal=self.class_num) # shape: (class_num,), sim_i_j is the cosine similarity between q_i and q_j
        sim_j_i = torch.diag(sim, diagonal=-self.class_num) # shape: (class_num,), sim_j_i is the cosine similarity between q_j and q_i
        sim_of_positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive samples
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative samples
        sim_of_negative_samples = sim[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative samples
        labels = torch.zeros(N).to(self.device).long() # shape: (N,), labels is the labels for the positive samples
        logits = torch.cat((sim_of_positive_samples, sim_of_negative_samples), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative samples
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for the positive and negative samples
        loss /= N # shape: (1,), loss is the average loss for the positive and negative samples
        return loss + entropy

def validation(model, dataset, view, data_size, class_num, eval_multi_view=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        zs, hs, qs, rs = model.forward(x_list) 
        multi_pred = [torch.argmax(qs[v], dim=1) for v in range(view)] # shape: (batch_size,) * view
        final_pred = torch.argmax(sum(qs)/view, dim=1) # shape: (batch_size,)
    low_level_vector = [z.detach().cpu().numpy() for z in zs] # shape: (batch_size, feature_dim) * view
    high_level_vector = [h.detach().cpu().numpy() for h in hs] # shape: (batch_size, feature_dim) * view
    multi_pred = [p.detach().cpu().numpy() for p in multi_pred] # shape: (batch_size,) * view
    final_pred = final_pred.detach().cpu().numpy() # shape: (batch_size,)
    if eval_multi_view:
        print("Clustering results on low-level features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(low_level_vector[v])
            nmi, ari, acc, pur = evaluate(y, y_pred)
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v + 1, acc, nmi, ari, pur))
        print("Clustering results on high-level features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(high_level_vector[v])
            nmi, ari, acc, pur = evaluate(y, y_pred)
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v + 1, acc, nmi, ari, pur))
        print("Clustering results on cluster assignments of each view:")
        for v in range(view):
            nmi, ari, acc, pur = evaluate(y, multi_pred[v])
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v+1, acc, nmi, ari, pur))
    print("Clustering results on overall cluster assignments: " + str(class_num))
    nmi, ari, acc, pur = evaluate(y, final_pred)
    print('Overall, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))

def match_pseudo_label_to_pred(y_true, y_pred):
    # match the pseudo-labels to the predicted labels using the Hungarian algorithm
    # For example: y_true = [0, 0, 0, 1, 1, 1], y_pred = [1, 1, 0, 0, 0, 0], the matched y_true = [1, 1, 1, 0, 0, 0]
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    row_ind, col_ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, row_ind is the row indices (predicted clusters), col_ind is the column indices (true clusters)
    # Create mapping: true_label -> predicted_label using dictionary
    mapping = {int(col_ind[j]): int(row_ind[j]) for j in range(len(row_ind))}
    matched_y_true = np.array([mapping[y_true[i]] for i in range(y_true.shape[0])])
    matched_y_true = torch.from_numpy(matched_y_true).long() # shape: (num_samples,)
    return matched_y_true

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MFLVC')
    parser.add_argument('--dataset', default='MNIST-USPS', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5, type=float)
    parser.add_argument("--temperature_l", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=50, type=int)
    parser.add_argument("--tune_epochs", default=50, type=int)
    parser.add_argument("--feature_dim", default=512, type=int)
    parser.add_argument("--high_feature_dim", default=128, type=int)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()
    os.makedirs('./result', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset; batch_size = args.batch_size; temperature_f = args.temperature_f; temperature_l = args.temperature_l; learning_rate = args.learning_rate; weight_decay = args.weight_decay
    mse_epochs = args.mse_epochs; con_epochs = args.con_epochs; tune_epochs = args.tune_epochs; feature_dim = args.feature_dim; high_feature_dim = args.high_feature_dim; seed = args.seed
    seed_dict = {"MNIST-USPS": 10, "BDGP": 10, "CCV": 3, "Fashion": 10, "Caltech-2V": 10, "Caltech-3V": 10, "Caltech-4V": 10, "Caltech-5V": 5}
    con_epochs_dict = {"MNIST-USPS": 50, "BDGP": 10, "CCV": 50, "Fashion": 100, "Caltech-2V": 50, "Caltech-3V": 50, "Caltech-4V": 50, "Caltech-5V": 50}
    con_epochs = con_epochs_dict[dataset_name]; seed = seed_dict[dataset_name]

    ## 1. Set seed for reproducibility.
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MFLVC(view, dims, feature_dim, high_feature_dim, class_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss() # input logits and labels, output the loss, but the predicted probability has been normalized by softmax, so there is two softmax layers in the model. NLL loss is better than CrossEntropyLoss.
    criterion_cl = MVCLLoss(batch_size, class_num, temperature_f, temperature_l, device).to(device)
    
    ## 3. Train and evaluate the model.
    ## 3.1. Reconstruction Pre-training
    model.train()
    for epoch in range(mse_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            _, _, _, rs = model(x_list)
            loss = sum([criterion_mse(x_list[v], rs[v]) for v in range(view)])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Reconstruction Pre-training Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(losses)))
    # validation(model, dataset, view, data_size, class_num, eval_multi_view=False)
    ## 3.2. Multi-view Contrastive Learning
    model.train()
    for epoch in range(con_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            zs, hs, qs, rs = model(x_list)
            loss_list = []
            for v, w in combinations(range(view), 2):
                loss_list.append(criterion_cl.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion_cl.forward_label(qs[v], qs[w]))
            for v in range(view):
                loss_list.append(criterion_mse(x_list[v], rs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Contrastive Learning Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(losses)))
    validation(model, dataset, view, data_size, class_num, eval_multi_view=False)
    ## 3.3. Multi-view Pseudo-label Refinement
    ## 3.3.1. Get the pseudo-labels from the high-level features
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]
    model.eval()
    with torch.no_grad():
        _, hs, _, _ = model.forward(x_list) 
    hs = [MinMaxScaler().fit_transform(hs[v].detach().cpu().numpy()) for v in range(view)] # Normalize the features by MinMaxScaler
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    pseudo_label_list = [kmeans.fit_predict(hs[v]).reshape(data_size) for v in range(view)]
    ## 3.3.2. Refine by matching the pseudo-labels to the predicted clusters
    model.train()
    for epoch in range(tune_epochs):
        optimizer.zero_grad()
        _, _, qs, _ = model(x_list) # qs is the cluster assignments of each modality, shape: (batch_size, class_num) * view
        loss_list = []
        for v in range(view):
            q = torch.argmax(qs[v], dim=1).detach().cpu().numpy()
            p_matched = match_pseudo_label_to_pred(pseudo_label_list[v], q).to(device)
            loss_list.append(criterion_ce(qs[v], p_matched))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        print('Pseudo-label Refinement Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss.item()/len(dataloader)))
    validation(model, dataset, view, data_size, class_num, eval_multi_view=False)
    torch.save(model.state_dict(), './result/' + dataset_name + '.pth')
    