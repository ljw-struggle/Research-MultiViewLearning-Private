import os, sys, random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate
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
        # Calculate the similarity matrix q using t-distribution.
        q = 1.0 / (1.0 + torch.sum((x_latent.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize q to sum to 1 across clusters
        return x_reconstructed, x_latent, q # shape: [batch_size, n_clusters]

class MVCAN(nn.Module):
    def __init__(self, view_num, view_size, latent_dim=10, n_clusters=20):
        super(MVCAN, self).__init__()
        self.view_num = view_num
        self.autoencoders = nn.ModuleList([AutoEncoder(view_size[i], latent_dim, activation='relu', n_clusters=n_clusters) for i in range(view_num)])

    def forward(self, x_list):
        r_list = []; z_list = []; q_list = []
        for v in range(self.view_num):
            x_reconstructed, x_latent, q = self.autoencoders[v](x_list[v])
            r_list.append(x_reconstructed); z_list.append(x_latent); q_list.append(q)
        return r_list, z_list, q_list
    
    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, dim=0) # shape: [batch_size, n_clusters]
        p = weight / torch.sum(weight, dim=1, keepdim=True)  # Normalize p to sum to 1 across clusters
        return p # shape: [batch_size, n_clusters]

    @staticmethod
    def calculate_t_distribution(inputs, centers, alpha=1): # calculate the t-distribution matrix
        q = 1.0 / (1.0 + np.sum((np.expand_dims(inputs, axis=1) - centers) ** 2, axis=2) / alpha) # shape: [batch_size, n_clusters]
        q = q ** ((alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / np.sum(q, axis=1, keepdims=True) # shape: [batch_size, n_clusters]
        return q # shape: [batch_size, n_clusters]

    @staticmethod
    def get_matched_matrix(y_true, y_pred): # shape: (num_samples,)
        # get the matched matrix using the Hungarian algorithm
        # For example: y_true = [0, 0, 0, 1, 1, 1], y_pred = [1, 1, 0, 0, 0, 0]
        # The matched matrix is: [[0, 1], [1, 0]], where the row indices are the predicted clusters, and the column indices are the true clusters.
        y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
        w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
        row_ind, col_ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, row_ind is the row indices (predicted clusters), col_ind is the column indices (true clusters)
        matched_matrix = np.zeros((y_pred.max()+1, y_true.max()+1), dtype=np.int64) # shape: [num_pred_clusters, num_true_clusters]
        matched_matrix[row_ind, col_ind] = 1
        return matched_matrix # shape: [num_pred_clusters, num_true_clusters]

def get_default_config(data_name):
    match data_name:
        case 'DIGIT': # [1, 32, 32] -> [1024]
            dataset_name = 'DIGIT'; random_state = 0; batch_size = 256; pretrain_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 10;
        case 'NoisyDIGIT': # [1, 32, 32] -> [1024]
            dataset_name = 'NoisyDIGIT'; random_state = 0; batch_size = 256; pretrain_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 10;
        case 'COIL': # [1, 32, 32] -> [1024]
            dataset_name = 'COIL'; random_state = 3; batch_size = 256; pretrain_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 10;
        case 'NoisyCOIL': # [1, 32, 32] -> [1024]
            dataset_name = 'NoisyCOIL'; random_state = 3; batch_size = 256; pretrain_epochs = 500; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 10;
        case 'Amazon': # [3, 32, 32] -> [3072]
            dataset_name = 'Amazon'; random_state = 1; batch_size = 256; pretrain_epochs = 200; train_epochs = 3000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 100;
        case 'NoisyAmazon': # [3, 32, 32] -> [3072]
            dataset_name = 'NoisyAmazon'; random_state = 1; batch_size = 256; pretrain_epochs = 200; train_epochs = 3000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 100;
        case 'BDGP':
            dataset_name = 'BDGP'; random_state = 1; batch_size = 256; pretrain_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 10;
        case 'NoisyBDGP':
            dataset_name = 'NoisyBDGP'; random_state = 1; batch_size = 256; pretrain_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 10;
        case 'DHA':
            dataset_name = 'DHA'; random_state = 1; batch_size = 256; pretrain_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 0.1;
        case 'RGB-D':
            dataset_name = 'RGB-D'; random_state = 13; batch_size = 256; pretrain_epochs = 200; train_epochs = 200; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 0.1;
        case 'Caltech-6V':
            dataset_name = 'Caltech-6V'; random_state = 5; batch_size = 256; pretrain_epochs = 200; train_epochs = 1000; T_1 = 2; T_2 = 100; learning_rate = 0.0001; lambda_clustering = 0.01;
        case 'YoutubeVideo':
            dataset_name = 'YoutubeVideo'; random_state = 2; batch_size = 256; pretrain_epochs = 50; train_epochs = 50; T_1 = 2; T_2 = 10; learning_rate = 0.0001; lambda_clustering = 10.0;
        case _:
            raise Exception(f"Dataset '{data_name}' is not supported")
    return dataset_name, random_state, batch_size, pretrain_epochs, train_epochs, T_1, T_2, learning_rate, lambda_clustering
    
def benchmark_2024_CVPR_MVCAN(dataset_name='BDGP', batch_size=256, pretrain_epochs=200, train_epochs=1000, T_1=2, T_2=100, learning_rate=0.0001, lambda_clustering=10, verbose=False, random_state=42):
    # MVCAN: every view perform deep embedded clustering globally with matched matrix and Adaptive Weighting.
    if dataset_name in ['DIGIT', 'NoisyDIGIT', 'COIL', 'NoisyCOIL', 'Amazon', 'NoisyAmazon', 'BDGP', 'NoisyBDGP', 'DHA', 'RGB-D', 'Caltech-6V', 'YoutubeVideo']: # get the default config for the dataset in the paper
        dataset_name, _, batch_size, pretrain_epochs, train_epochs, T_1, T_2, learning_rate, lambda_clustering = get_default_config(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True   

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    model = MVCAN(view, dims, latent_dim=10, n_clusters=class_num).to(device)
    optimizer = [torch.optim.Adam(model.autoencoders[v].parameters(), lr=learning_rate) for v in range(view)]
    criterion_mse = torch.nn.MSELoss()

    ## 3. Reconstruction Pre-training.
    print('Stage 1: Reconstruction Pre-training:') if verbose else None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(pretrain_epochs):
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
        print(f'Reconstruction Pre-training Epoch {epoch}', f'Loss:{np.mean(losses):.6f}') if verbose else None
    # print('Stage 1: Evaluation after Reconstruction Pre-training:')
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list_all, y_all, idx_all = next(iter(dataloader)); x_list_all = [x_list_all[v].to(device) for v in range(view)]
    r_list, z_list, q_list = model(x_list_all)
    for v in range(view):
        z_v = z_list[v].cpu().detach().numpy() # shape: [batch_size, feature_dim]
        y_pred_on_z_v = kmeans.fit_predict(z_v) # shape: [batch_size]
        model.autoencoders[v].cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(device)) # shape: [n_clusters, feature_dim]
        nmi, ari, acc, pur = evaluate(y_all.cpu().detach().numpy(), y_pred_on_z_v)
        print(f'KMeans on View {v+1}: NMI = {nmi:.4f}, ARI = {ari:.4f}, ACC = {acc:.4f}, PUR = {pur:.4f}') if verbose else None
    
    ## 4. Multi-view Clustering.
    print('Stage 2: Multi-view Clustering:') if verbose else None
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
                print(f'Update weight_for_each_view for each view: {weight_for_each_view}') if verbose else None
                
            final_k_means_cluster_centers = kmeans.cluster_centers_ # shape: [n_clusters, feature_dim]
            final_latent_fusion = latent_fusion_based_on_weight_for_each_view
            final_y_pred = y_pred
            final_y_pred_list = y_pred_list
            matched_matrix_list = [] # shape: [view, D, D]
            for v in range(view):
                matched_matrix_v = model.get_matched_matrix(final_y_pred_list[v], final_y_pred) # shape: [num_pred_clusters, num_true_clusters]
                matched_matrix_list.append(matched_matrix_v) # shape: [D, D], row indices are the final_y_pred labels, column indices are the final_y_pred_list[v] labels
            q = model.calculate_t_distribution(final_latent_fusion, final_k_means_cluster_centers) # shape: [batch_size, n_clusters]
            q = torch.from_numpy(q).to(device) # shape: [batch_size, n_clusters]
            p = model.target_distribution(q).float().detach() # shape: [batch_size, n_clusters]
            nmi, ari, acc, pur = evaluate(y_all.cpu().detach().numpy(), final_y_pred)
            print(f'Matched Matrix List: {matched_matrix_list}') if verbose else None
            print(f'Multi-view Clustering Epoch {epoch}: NMI = {nmi:.4f}, ARI = {ari:.4f}, ACC = {acc:.4f}, PUR = {pur:.4f}') if verbose else None
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            loss_list = []
            for v in range(view):
                r_v, z_v, q_v = model.autoencoders[v](x_list[v])
                # IDEA: q_v = sum(q_views_list) / view, shape: [batch_size, class_num]
                reconstruction_loss = F.mse_loss(r_v, x_list[v])
                # Here we use the matched matrix to match the global pseudo-labels to the predicted labels of each view, because the different views has different mapping to the global pseudo-labels.
                # After matching, the different views have same distribution of global pseudo-labels but different class labels as the view specific pseudo-labels.
                clustering_loss = F.mse_loss(q_v, torch.mm(p[idx], torch.from_numpy(matched_matrix_list[v]).float().to(device)))
                loss = reconstruction_loss + lambda_clustering * clustering_loss
                print(f'Reconstruction Loss: {reconstruction_loss}, Clustering Loss: {clustering_loss}, Total Loss: {loss}') if verbose else None
                optimizer[v].zero_grad()
                loss.backward()
                optimizer[v].step()
                loss_list.append(loss.item())
            losses.append(np.mean(loss_list))
        print(f'Multi-view Clustering Epoch {epoch}: Loss = {np.mean(losses):.6f}') if verbose else None
    print(f'Final acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, pur: {pur:.4f}.') if verbose else None
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVCAN")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--pretrain_epochs", default=200, type=int)
    parser.add_argument("--train_epochs", default=1000, type=int)
    parser.add_argument("--T_1", default=2, type=int)
    parser.add_argument("--T_2", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--lambda_clustering", default=10, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    
    nmi, ari, acc, pur = benchmark_2024_CVPR_MVCAN(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        pretrain_epochs=args.pretrain_epochs,
        train_epochs=args.train_epochs,
        T_1=args.T_1,
        T_2=args.T_2,
        learning_rate=args.learning_rate,
        lambda_clustering=args.lambda_clustering,
        verbose=False,
        random_state=args.seed
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
    