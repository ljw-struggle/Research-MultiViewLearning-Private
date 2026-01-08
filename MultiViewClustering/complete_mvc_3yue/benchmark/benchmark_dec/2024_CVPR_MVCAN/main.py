import os, time, random, argparse, itertools, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
from _utils import evaluate, load_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, activation='relu', n_clusters=10, alpha=1.0):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim; self.feature_dim = feature_dim; self.activation = activation; self.n_clusters = n_clusters; self.alpha = alpha
        activation_dict = {'sigmoid': nn.Sigmoid(), 'leakyrelu': nn.LeakyReLU(0.2, inplace=True), 'tanh': nn.Tanh(), 'relu': nn.ReLU()}
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), activation_dict[self.activation], nn.Linear(500, 500), activation_dict[self.activation], nn.Linear(500, 2000), activation_dict[self.activation], nn.Linear(2000, feature_dim))
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), activation_dict[self.activation], nn.Linear(2000, 500), activation_dict[self.activation], nn.Linear(500, 500), activation_dict[self.activation], nn.Linear(500, input_dim))
        self.cluster_centers = Parameter(torch.Tensor(self.n_clusters, self.feature_dim))
        torch.nn.init.xavier_normal_(self.cluster_centers.data)
    
    def forward(self, x):
        x_latent = self.encoder(x)
        x_reconstructed = self.decoder(x_latent)
        q = 1.0 / (1.0 + torch.sum((x_latent.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize q to sum to 1 across clusters
        return x_reconstructed, x_latent, q
    
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, dim=0) # shape: [batch_size, n_clusters]
        p = weight / torch.sum(weight, dim=1, keepdim=True)  # Normalize p to sum to 1 across clusters
        return p

    @staticmethod
    def calculate_t_distribution(inputs, centers, alpha=1): # calculate the t-distribution matrix
        q = 1.0 / (1.0 + np.sum((np.expand_dims(inputs, axis=1) - centers) ** 2, axis=2) / alpha) # shape: [batch_size, n_clusters]
        q = q ** ((alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / np.sum(q, axis=1, keepdims=True) # shape: [batch_size, n_clusters]
        return q
    
    @staticmethod # match the pseudo-labels to the predicted labels using the Hungarian algorithm
    def match_pseudo_label_to_pred(y_true, y_pred): # shape: (num_samples,)
        # match the pseudo-labels to the predicted labels using the Hungarian algorithm
        # For example: y_true = [0, 0, 0, 1, 1, 1], y_pred = [1, 1, 0, 0, 0, 0], the matched y_true = [1, 1, 1, 0, 0, 0]
        y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
        w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
        row_ind, col_ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, row_ind is the row indices (predicted clusters), col_ind is the column indices (true clusters)
        # Create mapping: true_label -> predicted_label using dictionary
        mapping = {int(col_ind[j]): int(row_ind[j]) for j in range(len(row_ind))}
        matched_y_true = np.array([mapping[y_true[i]] for i in range(y_true.shape[0])])
        matched_y_true = torch.from_numpy(matched_y_true).long() # shape: (num_samples,)
        matched_matrix = np.zeros((y_pred.max()+1, y_true.max()+1), dtype=np.int64) # shape: [num_pred_clusters, num_true_clusters]
        matched_matrix[row_ind, col_ind] = 1
        return matched_matrix # shape: [num_pred_clusters, num_true_clusters]

class MVCAN(nn.Module):
    def __init__(self, view_num, view_size, latent_dim=10, n_clusters=20):
        super(MVCAN, self).__init__()
        self.view_num = view_num
        self.view_size = view_size
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.autoencoders = nn.ModuleList([AutoEncoder(view_size[i], self.latent_dim, activation='relu', n_clusters=self.n_clusters) for i in range(view_num)])

    def forward(self, x_list):
        r_list = []; z_list = []; q_list = []
        for v in range(self.view_num):
            x_reconstructed, x_latent, q = self.autoencoders[v](x_list[v])
            r_list.append(x_reconstructed); z_list.append(x_latent); q_list.append(q)
        return r_list, z_list, q_list

def get_default_config(data_name):
    match data_name:
        case 'DIGIT': # [1, 32, 32] -> [1024]
            dataset = 'DIGIT'; seed = 0; batch_size = 256; pre_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 10;
        case 'NoisyDIGIT': # [1, 32, 32] -> [1024]
            dataset = 'NoisyDIGIT'; seed = 0; batch_size = 256; pre_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 10;
        case 'COIL': # [1, 32, 32] -> [1024]
            dataset = 'COIL'; seed = 3; batch_size = 256; pre_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 10;
        case 'NoisyCOIL': # [1, 32, 32] -> [1024]
            dataset = 'NoisyCOIL'; seed = 3; batch_size = 256; pre_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 10;
        case 'Amazon': # [3, 32, 32] -> [3072]
            dataset = 'Amazon'; seed = 1; batch_size = 256; pre_epochs = 200; train_epochs = 3000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 100;
        case 'NoisyAmazon': # [3, 32, 32] -> [3072]
            dataset = 'NoisyAmazon'; seed = 1; batch_size = 256; pre_epochs = 200; train_epochs = 3000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 100;
        case 'BDGP':
            dataset = 'BDGP'; seed = 1; batch_size = 256; pre_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 10;
        case 'NoisyBDGP':
            dataset = 'NoisyBDGP'; seed = 1; batch_size = 256; pre_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 10;
        case 'DHA':
            dataset = 'DHA'; seed = 1; batch_size = 256; pre_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 0.1;
        case 'RGB-D':
            dataset = 'RGB-D'; seed = 13; batch_size = 256; pre_epochs = 200; train_epochs = 200; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 0.1;
        case 'Caltech-6V':
            dataset = 'Caltech-6V'; seed = 5; batch_size = 256; pre_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; lr = 0.0001; lambda_clu = 0.01;
        case 'YoutubeVideo':
            dataset = 'YoutubeVideo'; seed = 2; batch_size = 256; pre_epochs = 50; train_epochs = 50; T_1 = 2; T_2 = 10; lr = 0.0001; lambda_clu = 10.0;
        case _:
            raise Exception(f"Dataset '{data_name}' is not supported")
    return dataset, seed, batch_size, pre_epochs, train_epochs, T_1, T_2, lr, lambda_clu

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # BDGP, NoisyBDGP, DIGIT, NoisyDIGIT, COIL, NoisyCOIL, Amazon, NoisyAmazon, DHA, RGB-D, Caltech-6V, YoutubeVideo
    parser.add_argument('--dataset', type=str, default='BDGP')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, seed, batch_size, pre_epochs, train_epochs, T_1, T_2, lr, lambda_clu = get_default_config(args.dataset)
    
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    model = MVCAN(view, dims, latent_dim=10, n_clusters=class_num).to(device)
    optimizer = [torch.optim.Adam(model.autoencoders[v].parameters(), lr=lr) for v in range(view)]
    criterion_mse = torch.nn.MSELoss()

    ## 3. Reconstruction Pre-training.
    print('Stage 1: Reconstruction Pre-training:')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(pre_epochs):
        losses = [] # shape: [iteration_per_epoch]
        for x_list, y, idx in dataloader:
            model.train()
            x_list = [x_list[v].to(device) for v in range(view)]; loss_list = [] # shape: [view]
            for v in range(view):
                r_v, z_v, q_v = model.autoencoders[v](x_list[v])
                loss_v = criterion_mse(r_v, x_list[v])
                optimizer[v].zero_grad()
                loss_v.backward()
                optimizer[v].step()
                loss_list.append(loss_v.item())
            losses.append(np.mean(loss_list)) # shape: [iteration_per_epoch]
        print(f'Reconstruction Pre-training Epoch {epoch}', f'Loss:{np.mean(losses):.6f}')
    print('Stage 1: Evaluation after Reconstruction Pre-training:')
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list_all, y_all, idx_all = next(iter(dataloader)); x_list_all = [x_list_all[v].to(device) for v in range(view)]
    r_list, z_list, q_list = model(x_list_all)
    for v in range(view):
        z_v = z_list[v].cpu().detach().numpy() # shape: [batch_size, feature_dim]
        y_pred_on_z_v = kmeans.fit_predict(z_v) # shape: [batch_size]
        model.autoencoders[v].cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)) # shape: [n_clusters, feature_dim]
        nmi, ari, acc, pur = evaluate(y_all.cpu().detach().numpy(), y_pred_on_z_v)
        print(f'KMeans on View {v+1}: NMI = {nmi:.4f}, ARI = {ari:.4f}, ACC = {acc:.4f}, PUR = {pur:.4f}')
    print('Stage 1: Evaluation after Setting Cluster Centers:')
    r_list, z_list, q_list = model(x_list_all)
    for v in range(view):
        y_pred_on_q_v = q_list[v].cpu().detach().numpy().argmax(1)
        nmi, ari, acc, pur = evaluate(y_all.cpu().detach().numpy(), y_pred_on_q_v)
        print(f'DEC on View {v+1}: NMI = {nmi:.4f}, ARI = {ari:.4f}, ACC = {acc:.4f}, PUR = {pur:.4f}')
    
    ## 4. Multi-view Clustering.
    print('Stage 2: Multi-view Clustering:')
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list_all, y_all, idx_all = next(iter(dataloader)); x_list_all = [x_list_all[v].to(device) for v in range(view)]
    weight_for_each_view = [1.0 for v in range(view)]
    for epoch in range(train_epochs+1):
        if epoch % T_2 == 0:
            for updata_weight_for_each_view in range(T_1): # update weight_for_each_view for each view (T_1 times in total)
                r_list, z_list, q_list = model(x_list_all)
                latent_list = []; y_pred_list = []
                for v in range(view):
                    r_v = r_list[v]; z_v = z_list[v]; q_v = q_list[v]
                    z_v = MinMaxScaler().fit_transform(z_v.cpu().detach().numpy()) # shape: [batch_size, feature_dim], min-max normalization on feature space
                    latent_v = z_v * weight_for_each_view[v] # shape: [batch_size, feature_dim]
                    latent_list.append(latent_v) # shape: [view, batch_size, feature_dim]
                    y_pred_v = q_v.cpu().detach().numpy().argmax(1) # shape: [batch_size]
                    y_pred_list.append(y_pred_v)
                latent_fusion_based_on_weight_for_each_view = np.hstack(latent_list) # shape: [batch_size, feature_dim * view]
                y_pred = kmeans.fit_predict(latent_fusion_based_on_weight_for_each_view) # shape: [batch_size]
                for v in range(view):
                    nmi, ari, acc, pur = evaluate(y_pred, y_pred_list[v])
                    weight_for_each_view[v] = np.exp(np.round(nmi, 5)) # update weight_for_each_view for each view
                print(f'Update weight_for_each_view for each view: {weight_for_each_view}')
            final_k_means_cluster_centers = kmeans.cluster_centers_ # shape: [n_clusters, feature_dim]
            final_latent_fusion = latent_fusion_based_on_weight_for_each_view
            final_y_pred = y_pred
            final_y_pred_list = y_pred_list
            matched_matrix_list = [] # shape: [view, D, D]
            for v in range(view):
                matched_matrix_v = AutoEncoder.match_pseudo_label_to_pred(final_y_pred_list[v], final_y_pred) # shape: [num_pred_clusters, num_true_clusters]
                matched_matrix_list.append(matched_matrix_v) # shape: [D, D], row indices are the y_pred labels, column indices are the y_pred_list[v] labels
            q = AutoEncoder.calculate_t_distribution(final_latent_fusion, final_k_means_cluster_centers) # shape: [batch_size, n_clusters]
            q = torch.from_numpy(q).to(device) # shape: [batch_size, n_clusters]
            p = AutoEncoder.target_distribution(q) # shape: [batch_size, n_clusters]
            nmi, ari, acc, pur = evaluate(y_all.cpu().detach().numpy(), final_y_pred)
            print(f'Matched Matrix List: {matched_matrix_list}')
            print(f'Multi-view Clustering Epoch {epoch}: NMI = {nmi:.4f}, ARI = {ari:.4f}, ACC = {acc:.4f}, PUR = {pur:.4f}')
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            loss_list = []
            for v in range(view):
                r_v, z_v, q_v = model.autoencoders[v](x_list[v])
                REC_loss = F.mse_loss(r_v, x_list[v])
                CLU_loss = F.mse_loss(q_v, torch.mm(p[idx], torch.from_numpy(matched_matrix_list[v]).float().to(device)))
                loss = REC_loss + lambda_clu * CLU_loss
                print(f'REC_loss: {REC_loss}, CLU_loss: {CLU_loss}, loss: {loss}')
                optimizer[v].zero_grad()
                loss.backward()
                optimizer[v].step()
                loss_list.append(loss.item())
            losses.append(np.mean(loss_list))
        print(f'Multi-view Clustering Epoch {epoch}: Loss = {np.mean(losses):.6f}')
