import random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, MiniBatchKMeans
from itertools import combinations
from _utils import evaluate, load_data

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

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []; self.decoders = []; self.feature_contrastive_modules = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
            self.feature_contrastive_modules.append(nn.Sequential(nn.Linear(feature_dim, high_feature_dim)))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_modules = nn.ModuleList(self.feature_contrastive_modules)

    def forward(self, xs):
        zs = []; hs = []; rs = []; qs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            h = F.normalize(self.feature_contrastive_modules[v](z), dim=1)
            r = self.decoders[v](z)
            zs.append(z); hs.append(h); rs.append(r)
        return zs, hs, rs

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask
    
    def forward_feature(self, z1, z2, type='InfoNCE'):
        if type == 'InfoNCE':
            return self.forward_feature_InfoNCE(z1, z2)
        elif type == 'PSCL':
            return self.forward_feature_PSCL(z1, z2)
        elif type == 'RINCE':
            return self.forward_feature_RINCE(z1, z2)

    def forward_feature_InfoNCE(self, h_i, h_j): # InfoNCE loss
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.get_correlated_mask(N)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_feature_RINCE(self, out_1, out_2, lam=0.001, q=0.5, temperature=0.5): # Robust InfoNCE loss
        # InfoNCE: = -e^(log(pos/(pos+neg)))
        # RINCE: = -e^(pos^q/q) + e^((lam*(pos+neg))^q/q)
        N = 2 * self.batch_size
        out_1_dist = out_1; out_2_dist = out_2; out = torch.cat([out_1, out_2], dim=0); out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)
        similarity = torch.exp(torch.mm(out, out_dist.t()) / temperature) # shape: (2*N, 2*N), similarity is the cosine similarity matrix
        neg_mask = self.get_correlated_mask(N)
        neg = torch.sum(similarity * neg_mask.to(self.device), dim=1) # shape: (N,), neg is the sum of the similarity between the negative samples
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature) # shape: (N,), pos is the exponential of the similarity between the positive samples
        pos = torch.cat([pos, pos], dim=0) # shape: (2*N,), pos is the concatenation of the positive samples
        # loss = -(torch.mean(torch.log(pos / (pos + neg)))) # InfoNCE loss
        pos_term = -(pos**q) / q
        neg_term = ((lam*(pos + neg))**q) / q
        loss = pos_term.mean() + neg_term.mean() # RINCE loss
        return loss
    
    def forward_feature_PSCL(self, z1, z2, r=3.0): # Population Spectral Contrastive Loss
        # z1.shape: (batch_size, feature_dim), z2.shape: (batch_size, feature_dim)
        # Project the features to the sphere with radius r.
        # - If the feature is outside the sphere, project it to the sphere.
        # - If the feature is inside the sphere, keep it.
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1) # shape: (batch_size, 1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1) # shape: (batch_size, 1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r) # shape: (batch_size, feature_dim)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r) # shape: (batch_size, feature_dim)
        # Loss part 1: maximizing the positive sample pair similarity.
        loss_part_1 = -2 * torch.mean(z1 * z2) * z1.shape[1] # shape: (1,), is equal to torch.mean(torch.sum(z1 * z2, dim=1))
        # Loss part 2: minimizing the negative sample pair similarity.
        square_term = torch.matmul(z1, z2.T) ** 2 # shape: (batch_size, batch_size)
        loss_part_2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * z1.shape[0] / (z1.shape[0] - 1) # shape: (1,)
        return loss_part_1 + loss_part_2

def validation(model, dataset, view, data_size, class_num, eval_z=True, eval_h=True, measure_type='CMI'):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        zs, hs, _ = model.forward(x_list)
    zs = [z.cpu().detach().numpy() for z in zs]; hs = [h.cpu().detach().numpy() for h in hs]
    nmi_matrix_z = np.zeros((view, view)); nmi_matrix_h = np.zeros((view, view))
    if eval_z:
        z_clusters = []
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100) if len(y) <= 10000 else MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
            z_clusters.append(kmeans.fit_predict(zs[v]))
        kmeans = KMeans(n_clusters=class_num, n_init=100) if len(y) <= 10000 else MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        y_pred = kmeans.fit_predict(np.concatenate(zs, axis=1))
        nmi, ari, acc, pur = evaluate(y, y_pred)
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
        for v in range(view):
            for w in range(view):
                if measure_type == 'CMI': # Cluster Mean Information (CMI)
                    cnmi, _, _, _ = evaluate(z_clusters[v], z_clusters[w])
                    nmi_matrix_z[v][w] = np.exp(cnmi) - 1
                if measure_type == 'JSD': # Jensen-Shannon Divergence (JSD)
                    P = torch.tensor(zs[v]); Q = torch.tensor(zs[w]); divergence = JSD(P, Q).item()
                    nmi_matrix_z[v][w] = np.exp(1 - divergence) - 1
                if measure_type == 'MMD': # Maximum Mean Discrepancy (MMD)
                    P = torch.tensor(zs[v][0: 2000]); Q = torch.tensor(zs[w][0: 2000]) # select partial samples to compute MMD as it has high complexity, otherwise might be out-of-memory
                    mmd = MMD(P, Q, kernel_mul=4, kernel_num=4)
                    nmi_matrix_z[v][w] = np.exp(-mmd)
    if eval_h:
        h_clusters = []
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100) if len(y) <= 10000 else MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
            h_clusters.append(kmeans.fit_predict(hs[v]))
        kmeans = KMeans(n_clusters=class_num, n_init=100) if len(y) <= 10000 else MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        y_pred = kmeans.fit_predict(np.concatenate(hs, axis=1))
        nmi, ari, acc, pur = evaluate(y, y_pred)
        print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
        for v in range(view):
            for w in range(view):
                if measure_type == 'CMI': # Cluster Mean Information (CMI)
                    cnmi, _, _, _ = evaluate(h_clusters[v], h_clusters[w])
                    nmi_matrix_h[v][w] = np.exp(cnmi) - 1
                if measure_type == 'JSD': # Jensen-Shannon Divergence (JSD)
                    P = torch.tensor(hs[v]); Q = torch.tensor(hs[w]); divergence = JSD(P, Q).item()
                    nmi_matrix_h[v][w] = np.exp(1 - divergence) - 1
                if measure_type == 'MMD': # Maximum Mean Discrepancy (MMD)
                    P = torch.tensor(hs[v][0: 2000]); Q = torch.tensor(hs[w][0: 2000]) # select partial samples to compute MMD as it has high complexity, otherwise might be out-of-memory
                    mmd = MMD(P, Q, kernel_mul=4, kernel_num=4)
                    nmi_matrix_h[v][w] = np.exp(-mmd)
    return acc, nmi, ari, pur, nmi_matrix_z, nmi_matrix_h

def JSD(p_output, q_output, get_softmax=True): # Jensen-Shannon Divergence
    KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)
    p_output = F.softmax(p_output, dim=1) if get_softmax else p_output
    q_output = F.softmax(q_output, dim=1) if get_softmax else q_output
    log_mean_output = ((p_output + q_output)/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

def MMD(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None): # Maximum Mean Discrepancy
    # Different from MSE, MMD is a kernel-based method that measures the similarity between two distributions. 
    # MMD don't need paired data, so it is more robust to the noise and outliers.
    # MMD^2(x, y) = MMD^2(source, target) = (1/n^2) * sum(K_ss) + (1/m^2) * sum(K_tt) - (2/n*m) * sum(K_st)
    def guassian_kernel_mmd(source, target, kernel_mul=2, kernel_num=4, fix_sigma=None): # Gram matrix of the Gaussian kernel
        # source: sample_size_1 * feature_size; target: sample_size_2 * feature_size;kernel_mul: bandwith of kernels; kernel_num: number of kernels
        # return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2); shape: [[K_ss, K_st], [K_ts, K_tt]]
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0) # shape: (n_samples_source + n_samples_target, feature_size)
        total_0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # shape: (n_samples_source + n_samples_target, n_samples_source + n_samples_target, feature_size)
        total_1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # shape: (n_samples_source + n_samples_target, n_samples_source + n_samples_target, feature_size)
        l2_distance = ((total_0 - total_1) ** 2).sum(dim=2) # shape: (n_samples_source + n_samples_target, n_samples_source + n_samples_target)
        bandwidth = fix_sigma if fix_sigma is not None else torch.sum(l2_distance.detach().data) / (n_samples ** 2 - n_samples) # shape: (1,)
        bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2)) # shape: (1,)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)] # shape: (kernel_num,), bandwidth_list is the list of the bandwidths
        kernel_val = [torch.exp(-l2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list] # shape: (kernel_num,), kernel_val is the list of the kernel values
        return sum(kernel_val) # shape: (sample_size_1 + sample_size_2, sample_size_1 + sample_size_2)
    n = int(source.size()[0]); m = int(target.size()[0])
    kernels = guassian_kernel_mmd(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]; YY = kernels[n:, n:]; XY = kernels[:n, m:]; YX = kernels[m:, :n]; 
    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)   # K_ss, Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st, Source<->Target
    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts, Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)   # K_tt, Target<->Target
    loss = XX.sum() + XY.sum() + YX.sum() + YY.sum()
    return loss
    
def generate_mask(rows, cols, p): # Generate the mask for the samples and features
    mask = torch.ones(rows * cols) # shape: (rows * cols), mask is the mask for the samples and features
    mask_indices = torch.randperm(rows * cols)[:int(rows * cols * p)] # shape: (int(rows * cols * p)), mask_indices is the indices of the masked samples and features
    mask[mask_indices] = 0 # set the masked samples and features to 0
    return mask.reshape(rows, cols) # shape: (rows, cols), mask is the mask for the samples and features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SEM')
    parser.add_argument('--dataset', default='Caltech', type=str)
    parser.add_argument("--feature_dim", default=512, type=int)
    parser.add_argument("--high_feature_dim", default=128, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--temperature_f", default=1.0, type=float)
    parser.add_argument("--reconstruction_training_method", default='AE', type=str, choices=['AE', 'DAE', 'MAE'])
    parser.add_argument("--contrastive_training_method", default='InfoNCE', type=str, choices=['InfoNCE', 'PSCL', 'RINCE'])
    parser.add_argument("--mse_epochs", default=100, type=int)
    parser.add_argument("--con_epochs", default=100, type=int)
    parser.add_argument("--lambda_reconstruction", default=1, type=float)
    parser.add_argument("--measurement", default='CMI', type=str, choices=['CMI', 'JSD', 'MMD'])
    parser.add_argument("--weight_update_iteration", default=4, type=int)
    parser.add_argument("--seed", default=5, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset; feature_dim = args.feature_dim; high_feature_dim = args.high_feature_dim; batch_size = args.batch_size; learning_rate = args.learning_rate; weight_decay = args.weight_decay; temperature_f = args.temperature_f; 
    reconstruction_training_method = args.reconstruction_training_method; contrastive_training_method = args.contrastive_training_method; mse_epochs = args.mse_epochs; con_epochs = args.con_epochs; 
    lambda_reconstruction = args.lambda_reconstruction; measurement = args.measurement; weight_update_iteration = args.weight_update_iteration; seed = args.seed
    if args.dataset == "DHA": con_epochs = 300; weight_update_iteration = 1
    if args.dataset == "CCV": con_epochs = 50; weight_update_iteration = 4
    if args.dataset == "YoutubeVideo": con_epochs = 25; weight_update_iteration = 1
    if args.dataset == "NUSWIDE": con_epochs = 25; weight_update_iteration = 4
    if args.dataset == "Caltech": con_epochs = 100; weight_update_iteration = 3
    
    # # 1. Set seed for reproducibility.
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = Network(view, dims, feature_dim, high_feature_dim, class_num, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = Loss(batch_size, class_num, temperature_f, device).to(device)
    criterion_mse = torch.nn.MSELoss()
    
    ## 3.1 Reconstruction Training
    for epoch in range(mse_epochs):
        tot_loss_list = []
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            xs = [x.to(device) for x in xs]
            optimizer.zero_grad()
            if reconstruction_training_method == 'AE':
                _, _, rs = model(xs)
            if reconstruction_training_method == 'DAE':
                _, _, rs = model([torch.randn(x.shape).to(device) + x for x in xs])
            if reconstruction_training_method == 'MAE':
                _, _, rs = model([generate_mask(x.shape[0], x.shape[1], 0.3).to(device) * x for x in xs])
            loss = sum([criterion_mse(xs[v], rs[v]) for v in range(view)]) # Reconstruction loss
            loss.backward()
            optimizer.step()
            tot_loss_list.append(loss.item())
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(tot_loss_list)))

    ## 3.2 Self-Weighted Multi-view Contrastive Learning with Reconstruction Regularization
    acc, nmi, ari, pur, weight_matrix, _ = validation(model, dataset, view, data_size, class_num, eval_z=True, eval_h=False, measure_type=measurement)
    for epoch in range(con_epochs * weight_update_iteration):
        tot_loss_list = []
        for batch_idx, (xs, _, _) in enumerate(data_loader):
            xs = [x.to(device) for x in xs]
            optimizer.zero_grad()
            # if reconstruction_training_method == 'AE':
            _, hs, rs = model(xs)
            if reconstruction_training_method == 'DAE':
                _, _, rs = model([torch.randn(x.shape).to(device) + x for x in xs])
            if reconstruction_training_method == 'MAE':
                _, _, rs = model([generate_mask(x.shape[0], x.shape[1], 0.3).to(device) * x for x in xs])
            loss_list = []
            for v in range(view):
                for w in range(view):
                    loss_list.append(criterion.forward_feature(hs[v], hs[w], type=contrastive_training_method) * weight_matrix[v][w])
            for v in range(view):
                loss_list.append(lambda_reconstruction * criterion_mse(xs[v], rs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss_list.append(loss.item())
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(tot_loss_list)))
        # This code matters, to re-initialize the optimizers.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if (epoch + 1) % con_epochs == 0:
            acc, nmi, ari, pur, _, weight_matrix = validation(model, dataset, view, data_size, class_num, eval_z=False, eval_h=True, measure_type=measurement)

    # from sklearn import svm, model_selection
    # from sklearn.metrics import accuracy_score, recall_score
    # model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    # x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    # with torch.no_grad():
    #     zs, hs, _ = model.forward(x_list)
    # zs = [z.cpu().detach().numpy() for z in zs]; hs = [h.cpu().detach().numpy() for h in hs]
    # H = np.concatenate(hs, axis=1) # shape: (batch_size, feature_dim * view)
    # ACC_ALLH = []; REC_ALLH = []
    # for seed in range(10):
    #     data_train, data_test, labels_train, labels_test = model_selection.train_test_split(H, y, random_state=seed, test_size=0.7)
    #     clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr') # One-vs-Rest (OvR) strategy for multi-class classification
    #     clf.fit(data_train, labels_train.ravel()) # Train the SVM model, ravel() is used to flatten the labels to a 1D array
    #     labels_pred = clf.predict(data_test) # Predict the labels of the test data
    #     acc = accuracy_score(labels_test, labels_pred) # Calculate the accuracy
    #     recall = recall_score(labels_test, labels_pred, average='macro') # Calculate the recall
    #     ACC_ALLH.append(acc/0.01); REC_ALLH.append(recall/0.01) # Normalize the accuracy and recall by 0.01
    # print('ACC: {:.4f} ± {:.4f}'.format(np.mean(ACC_ALLH), np.std(ACC_ALLH)))
    # print('REC: {:.4f} ± {:.4f}'.format(np.mean(REC_ALLH), np.std(REC_ALLH)))
