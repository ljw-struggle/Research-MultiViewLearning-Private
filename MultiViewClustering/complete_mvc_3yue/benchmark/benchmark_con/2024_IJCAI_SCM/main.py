import random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from scipy.spatial import distance
from _utils import load_data, evaluate, scale_normalize_matrix

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
    def __init__(self, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.encoders = Encoder(input_size, feature_dim).to(device)
        self.decoders = Decoder(input_size, feature_dim).to(device)
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.label_contrastive_module = nn.Sequential(nn.Linear(high_feature_dim, 64), nn.Softmax(dim=1))

    def forward(self, x):
        z = self.encoders(x)
        h = F.normalize(self.feature_contrastive_module(z), dim=1) # L2 normalization
        q = self.label_contrastive_module(h)
        r = self.decoders(z)
        return r, z, h, q
    
class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
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

def validation(model, dataset, view, data_size, class_num, eval_q=False, eval_h=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    xs_all = torch.cat(x_list, dim=1)
    with torch.no_grad():
        _, _, h, q = model.forward(xs_all)
    h = h.cpu().detach().numpy()
    q = q.cpu().detach().numpy()
    if eval_q == True:
        q_pred = np.argmax(np.array(q), axis=1)
        nmi, ari, acc, pur = evaluate(y, q_pred)
        print('Evaluation on cluster assignments: ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
    if eval_h == True:
        h_pred = KMeans(n_clusters=class_num, n_init='auto').fit_predict(h)
        nmi, ari, acc, pur = evaluate(y, h_pred)
        print('Evaluation on feature vectors: ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))

def missing_mask(sample_num, view_num, missing_rate):
    mask = np.ones((sample_num, view_num)) # shape: (sample_num, view_num), mask is the mask for the samples
    num_partial_mask = int(sample_num * missing_rate) # num_partial_mask is the number of samples to be masked
    for i in range(num_partial_mask): # generate the partial mask for the first num_partial_mask samples
        num_zero_indices = np.random.randint(1, view_num) # 1 < num_zero_indices < view_num, num_zero_indices is the number of zeros to be set to 0
        zero_indices = np.random.choice(view_num, size=num_zero_indices, replace=False) # randomly choose num_zero_indices indices from view_num to set to 0
        mask[i, zero_indices] = 0 # set the zeros to 0
    np.random.shuffle(mask) # shuffle the mask to ensure the randomness across samples
    return mask.astype(np.float32) # shape: (sample_num, view_num), mask is the mask for the samples

def noise_addition(sample_num, feature_num, std, noise_rate):
    noise_matrix = []
    for i in range(sample_num):
        if np.random.random() < noise_rate:
            noise = np.random.randn(feature_num) * std
            noise_matrix.append(noise)
        else:
            noise_matrix.append(np.zeros(feature_num))
    noise_matrix = np.array(noise_matrix)
    return noise_matrix.astype(np.float32) # shape: (sample_num, feature_num), noise_matrix is the noise matrix

def destiny_peak(model, device, gamma, alpha, beta, metric='euclidean'): # density peak clustering !!!!!
    xs_all, y_all, _ = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)))
    xs_all = [xs_all[v].to(device) for v in range(view)]
    xs_all_concat = torch.cat(xs_all, dim=1)
    with torch.no_grad():
        _, _, h, _ = model.forward(xs_all_concat)
        h = h.cpu().detach().numpy() # shape: (data_size, feature_dim), h is the feature vectors of the original data
    ## 1. Calculate the density rho of each sample. (The more neighbors, the higher the density)
    condensed_distance = distance.pdist(h, metric=metric) # shape: (data_size * (data_size - 1) / 2), condensed_distance is the condensed distance matrix
    threshold = np.sort(condensed_distance)[int(len(condensed_distance) * gamma)] # shape: (1,), threshold is the threshold for the condensed distance
    redundant_distance = distance.squareform(condensed_distance) # shape: (data_size, data_size), redundant_distance is the redundant distance matrix
    # ie: condensed_distance is [1.0, 2.0, 3.0]; redundant_distance is [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    rho = np.sum(np.exp(-(redundant_distance / threshold) ** 2), axis=1) # shape: (data_size,), rho is the density of the data
    ## 2. Calculate the delta and the nearest neighbor
    order_distance = np.argsort(redundant_distance, axis=1) # shape: (data_size, data_size), order_distance is the order of the distance (from small to large) along the rows
    delta = np.zeros_like(rho) # shape: (data_size,), delta is the delta of the data
    nearest_neighbor = np.zeros_like(rho).astype(int) # shape: (data_size,), nearest_neighbor is the nearest neighbor of the data
    for i in range(len(delta)):
        mask = rho[order_distance[i]] > rho[i] # shape: (data_size,), mask is the mask for the neighbors with higher density than the current sample
        if mask.sum() > 0:
            nearest_neighbor[i] = order_distance[i][mask][0] # shape: (1,), nearest_neighbor[i] is the nearest neighbor of the sample i with higher density than the current sample
            delta[i] = redundant_distance[i, nearest_neighbor[i]] # shape: (1,), delta[i] is the delta of the data (the distance between the sample i and its nearest neighbor with higher density than the current sample)
        else:
            nearest_neighbor[i] = order_distance[i, -1] # shape: (1,), nearest_neighbor[i] is the most distant neighbor of the sample i when there are no neighbors with higher density than the current sample
            delta[i] = redundant_distance[i, nearest_neighbor[i]] # shape: (1,), delta[i] is the delta of the data (the distance between the sample i and its most distant neighbor when there are no neighbors with higher density than the current sample)
    ## 3. Calculate the centers of the clusters.
    rho_c = min(rho) + (max(rho) - min(rho)) * alpha # shape: (1,), rho_c is the threshold for the density
    delta_c = min(delta) + (max(delta) - min(delta)) * beta # shape: (1,), delta_c is the threshold for the delta
    # centers are the samples with density higher density and far away from other high-density samples
    centers = np.where(np.logical_and(rho > rho_c, delta > delta_c))[0] # shape: (num_clusters,), centers is the centers of the clusters
    num_clusters = len(centers) # shape: (1,), num_clusters is the number of clusters
    cluster_points = h[centers] # shape: (num_clusters, feature_dim), cluster_points is the points of the clusters
    ## 4. Calculate the probabilities of the data to the clusters.
    probabilities = np.zeros((h.shape[0], num_clusters)) # shape: (data_size, num_clusters), probabilities is the probabilities of the data
    for i in range(h.shape[0]):
        for j in range(num_clusters):
            # probability = exp(-||h[i] - cluster_points[j]||), not Gaussian kernel (which used the squared distance)
            probabilities[i, j] = np.exp(-np.linalg.norm(h[i] - cluster_points[j])) # shape: (1,), probabilities[i, j] is the probability of the data i to the cluster j
    probabilities /= probabilities.sum(axis=1, keepdims=True) # shape: (data_size, num_clusters), probabilities is the probabilities of the data
    ## 5. Calculate the predicted labels of the data and the accuracy.
    y_pred = torch.from_numpy(probabilities) # shape: (data_size, num_clusters), y_pred is the predicted labels of the data
    y_pred = torch.argmax(y_pred, dim=1) # shape: (data_size,), y_pred is the predicted labels of the data
    confusion = confusion_matrix(y_pred, y_all) # shape: (num_clusters, num_clusters), confusion is the confusion matrix
    per = np.sum(np.max(confusion, axis=0)) / np.sum(confusion) # shape: (1,), per is the accuracy (purity score) of the predicted labels
    additional_columns = 64 - probabilities.shape[1] # shape: (1,), additional_columns is the number of additional columns
    zero_columns = np.zeros((probabilities.shape[0], additional_columns)) # shape: (data_size, additional_columns), zero_columns is the zero columns
    probabilities = np.hstack((probabilities, zero_columns)) # shape: (data_size, 64), probabilities is the probabilities of the data
    probabilities = torch.from_numpy(probabilities) # shape: (data_size, 64), probabilities is the probabilities of the data
    print('num:{}'.format(num_clusters), 'accuracy:{:.6f}'.format(per)) # print the number of clusters and the accuracy
    return probabilities # shape: (data_size, 64), probabilities is the probabilities of the data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCM')
    parser.add_argument('--dataset', default='MNIST-USPS')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--stage_1_iterations", default=50, type=int)
    parser.add_argument("--stage_2_iterations", default=200, type=int)
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--high_feature_dim", default=128, type=int)
    parser.add_argument('--miss_rate', default=0.25, type=float)
    parser.add_argument('--noise_rate', default=0.25, type=float)
    parser.add_argument('--noise_std', default=0.4, type=float)
    parser.add_argument('--mode', default='SCM_ETC', type=str, choices=['SCM', 'SCM_REC', 'SCM_REC_ETC', 'SCM_ETC'])
    parser.add_argument('--seed', default=10, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset; batch_size = args.batch_size; temperature_f = args.temperature_f; learning_rate = args.learning_rate; weight_decay = args.weight_decay; 
    stage_1_iterations = args.stage_1_iterations; stage_2_iterations = args.stage_2_iterations; feature_dim = args.feature_dim; high_feature_dim = args.high_feature_dim; 
    miss_rate = args.miss_rate; noise_rate = args.noise_rate; noise_std = args.noise_std; mode = args.mode; seed = args.seed
    if dataset_name == "MNIST-USPS": stage_1_iterations = 1000; stage_2_iterations = 1000; gamma = 0.02; alpha = 0.5; beta = 0.5; seed = 1
    if dataset_name == "BDGP": stage_1_iterations = 400; stage_2_iterations = 3000; gamma = 0.1; alpha = 0.5; beta = 0.5; seed = 4
    if dataset_name == "Fashion": stage_1_iterations = 20000; stage_2_iterations = 2500; gamma = 0.003; alpha = 0.2; beta = 0.81; seed = 1
    if dataset_name == "DHA": stage_1_iterations = 500; stage_2_iterations = 700; gamma = 0.02; alpha = 0.2; beta = 0.5; seed = 4
    if dataset_name == "WebKB": stage_1_iterations = 200; stage_2_iterations = 200; gamma = 0.001; alpha = 0.6; beta = 0.6; seed = 2
    if dataset_name == "NGs": stage_1_iterations = 200; stage_2_iterations = 800; gamma = 0.00005; alpha = 0.5; beta = 0.5; seed = 5
    if dataset_name == "VOC": stage_1_iterations = 200; stage_2_iterations = 900; gamma = 0.002; alpha = 0.01; beta = 0.37; seed = 9
    if dataset_name == "Fc_COIL_20": stage_1_iterations = 2000; stage_2_iterations = 400; gamma = 0.031; alpha = 0.2; beta = 0.5; seed = 1
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = Network(sum(dims), feature_dim, high_feature_dim, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = Loss(batch_size, class_num, temperature_f, device).to(device)
    criterion_mse = torch.nn.MSELoss()
    
    for epoch in range(stage_1_iterations):
        xs, y, _ = next(iter(dataloader))
        xs = [xs[v].to(device) for v in range(view)]
        mask = missing_mask(batch_size, view, miss_rate)
        xs_masked = [torch.from_numpy(np.expand_dims(mask[:, v], axis=1) * xs[v].cpu().detach().numpy()).to(device) for v in range(view)]
        xs_noised = [torch.from_numpy(noise_addition(xs[v].shape[0], xs[v].shape[1], noise_std, noise_rate) + xs[v].cpu().detach().numpy()).to(device) for v in range(view)]
        xs_origin_concat = torch.cat(xs, dim=1); xs_masked_concat = torch.cat(xs_masked, dim=1); xs_noised_concat = torch.cat(xs_noised, dim=1)
        optimizer.zero_grad()
        r_origin, z_origin, h_origin, q_origin = model(xs_origin_concat)
        r_masked, z_masked, h_masked, q_masked = model(xs_masked_concat)
        r_noised, z_noised, h_noised, q_noised = model(xs_noised_concat)
        loss_reconstruction_origin = criterion_mse(xs_origin_concat, r_origin)
        loss_reconstruction_masked = criterion_mse(xs_masked_concat, r_masked)
        loss_reconstruction_noised = criterion_mse(xs_noised_concat, r_noised)
        loss_contrastive_1 = criterion.forward_feature(h_noised, h_masked)
        loss_contrastive_2 = criterion.forward_feature(h_masked, h_noised)
        if mode == 'SCM' or mode == 'SCM_ETC': loss = loss_contrastive_1 + loss_contrastive_2
        if mode =='SCM_REC' or mode == 'SCM_REC_ETC': loss = loss_reconstruction_origin + loss_reconstruction_masked + loss_reconstruction_noised + loss_contrastive_1 + loss_contrastive_2
        loss.backward()
        optimizer.step()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss))
    validation(model, dataset, view, data_size, class_num, eval_h=True)
    if mode == 'SCM_ETC' or mode =='SCM_REC_ETC':
        probability_matrix = destiny_peak(model, device, gamma, alpha, beta) # shape: (data_size, 64), probability_matrix is the probabilities of the data
        for epoch in range(stage_2_iterations):
            xs, y, idx = next(iter(dataloader))
            xs = [xs[v].to(device) for v in range(view)]
            mask = missing_mask(batch_size, view, miss_rate)
            xs_masked = [torch.from_numpy(np.expand_dims(mask[:, v], axis=1) * xs[v].cpu().detach().numpy()).to(device) for v in range(view)]
            xs_noised = [torch.from_numpy(noise_addition(xs[v].shape[0], xs[v].shape[1], noise_std, noise_rate) + xs[v].cpu().detach().numpy()).to(device) for v in range(view)]
            xs_origin_concat = torch.cat(xs, dim=1); xs_masked_concat = torch.cat(xs_masked, dim=1); xs_noised_concat = torch.cat(xs_noised, dim=1)
            optimizer.zero_grad()
            r_origin, z_origin, h_origin, q_origin = model(xs_origin_concat)
            r_masked, z_masked, h_masked, q_masked = model(xs_masked_concat)
            r_noised, z_noised, h_noised, q_noised = model(xs_noised_concat)
            # get the probability matrix for the original data
            select_rows = probability_matrix[idx]; qs = np.vstack(select_rows); # shape: (batch_size, 64), qs is the probabilities of the data
            qs = torch.from_numpy(qs).float().to(device); qs = scale_normalize_matrix(qs) # shape: (batch_size, 64), qs is the normalized probabilities of the data
            loss_reconstruction_origin = criterion_mse(xs_origin_concat, r_origin)
            loss_reconstruction_masked = criterion_mse(xs_masked_concat, r_masked)
            loss_reconstruction_noised = criterion_mse(xs_noised_concat, r_noised)
            loss_contrastive_1 = criterion.forward_feature(h_noised, h_masked)
            loss_contrastive_2 = criterion.forward_feature(h_masked, h_noised)
            loss_contrastive = criterion_mse(qs, q_origin) # qs.shape: (batch_size, class_num), q_origin.shape: (batch_size, class_num)
            if mode == 'SCM_REC_ETC': loss = loss_reconstruction_origin + loss_reconstruction_masked + loss_reconstruction_noised + loss_contrastive_1 + loss_contrastive_2 + loss_contrastive
            if mode == 'SCM_ETC': loss =  loss_contrastive_1 + loss_contrastive_2 + loss_contrastive
            loss.backward()
            optimizer.step()
            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss))
        validation(model, dataset, view, data_size, class_num, eval_q =True)
