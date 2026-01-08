import os, gc, csv, logging, argparse, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
from _utils import load_data, evaluate    
from sklearn.cluster import KMeans
from tabulate import tabulate
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "5"

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(500, 500), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(500, 2000), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(2000, feature_dim))

    def forward(self, x):
        sparse_act_list = []
        for layer in self.encoder:
            x = layer(x); sparse_act_list.append(x) if isinstance(layer, nn.ReLU) else None
        return x, sparse_act_list

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dropout_rate=0.0):
        super(Decoder, self).__init__()  
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(2000, 500), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(500, 500), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(500, input_dim))

    def forward(self, x):
        return self.decoder(x)  
    
class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.query_layer = Linear(feature_dim, feature_dim, bias=False)
        self.key_layer = Linear(feature_dim, feature_dim, bias=False)
        self.value_layer = Linear(feature_dim, feature_dim, bias=False)
        self.scale = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))
        
    def forward(self, z_all, zs):
        Q = self.query_layer(z_all) # [batch_size, feature_dim]
        K = torch.stack([self.key_layer(z) for z in zs], dim=1) # [batch_size, view_count, feature_dim]
        scores = torch.einsum('bf,bvf->bv', Q, K) / self.scale # [batch_size, view_count]
        attention_weights = F.softmax(scores, dim=1) # [batch_size, view_count]
        return attention_weights # [batch_size, view_count]

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super(Network, self).__init__()
        self.view = view
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.device = device
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim, 0.2).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim, 0.2).to(device))
        self.concat_encoder = Encoder(sum(input_size), feature_dim, 0.2).to(device)
        self.concat_decoder = Decoder(sum(input_size), feature_dim, 0.2).to(device)
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_fusion_module = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, high_feature_dim))
        self.common_information_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.cycle_transfer_module = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, high_feature_dim))

    # def forward(self, xs):
    #     statistics = self.multi_view_zero_value_proportion_statistics_once(xs)
    #     means = [mean for mean, _ in statistics] # shape: (view_count,)
    #     # print(f'Multi-view sparsity ratio(zero_value_proportion mean):{means}')
    #     hs = []; rs = []; zs = []; activations = []
    #     concat_xs = torch.cat(xs, dim=1) # [batch_size, sum(input_size)]
    #     z_concat, hidden_activation_concat = self.concat_encoder(concat_xs) # [batch_size, feature_dim]
    #     r_concat = self.concat_decoder(z_concat) # [batch_size, input_size]
    #     activations.append(hidden_activation_concat) # [[batch_size, hidden_dim_1], [batch_size, hidden_dim_4], ..., [batch_size, hidden_dim_7]]
    #     for v in range(self.view):
    #         z, hidden_activation = self.encoders[v](xs[v]) # [batch_size, feature_dim]
    #         r = self.decoders[v](z) # [batch_size, input_size]
    #         h = normalize(self.common_information_module(z), dim=1) # [batch_size, high_feature_dim]
    #         zs.append(z); # [batch_size, feature_dim]
    #         activations.append(hidden_activation) # [[batch_size, hidden_dim_1], [batch_size, hidden_dim_4], ..., [batch_size, hidden_dim_7]]
    #         rs.append(r) # [batch_size, input_size]
    #         hs.append(h) # [batch_size, high_feature_dim]

    #     attention = Attention(self.feature_dim).to(self.device) # This is not trainable during training stage.
    #     weights = attention(z_concat, zs) # [batch_size, view_count]
    #     weights_expanded = weights.unsqueeze(-1)  # [batch_size, view_count, 1]
    #     stacked_zs = torch.stack(zs) # [view_count, batch_size, feature_dim]
    #     stacked_zs_transposed = stacked_zs.permute(1, 0, 2)  # [batch_size, view_count, feature_dim]
    #     z_weighted_sum = torch.sum(stacked_zs_transposed * weights_expanded, dim=1)  # [batch_size, feature_dim]
    #     H = normalize(self.feature_fusion_module(z_weighted_sum), dim=1)  # [batch_size, high_feature_dim]
    #     return rs, zs, hs, H, r_concat, z_concat, activations, means

    # NOTE: SparseMVC original code
    def forward(self, xs):
        statistics = self.multi_view_zero_value_proportion_statistics_once(xs)
        means = [mean for mean, _ in statistics] # shape: (view_count,)
        # print(f'Multi-view sparsity ratio(zero_value_proportion mean):{means}')
        hs = []; rs = []; zs = []; activations = []
        concat_xs = torch.cat(xs, dim=1) # [batch_size, sum(input_size)]
        z_concat, hidden_activation_concat = self.concat_encoder(concat_xs) # [batch_size, feature_dim]
        activations.append(hidden_activation_concat) # [[batch_size, hidden_dim_1], [batch_size, hidden_dim_4], ..., [batch_size, hidden_dim_7]]
        for v in range(self.view):
            z, hidden_activation = self.encoders[v](xs[v]) # [batch_size, feature_dim]
            zs.append(z); # [batch_size, feature_dim]
            activations.append(hidden_activation) # [[batch_size, hidden_dim_1], [batch_size, hidden_dim_4], ..., [batch_size, hidden_dim_7]]
            
        attention = Attention(self.feature_dim).to(self.device)
        weights = attention(z_concat, zs) # [batch_size, view_count]

        for v in range(self.view):
            r = self.decoders[v](zs[v]) # [batch_size, input_size]
            h = normalize(self.common_information_module(zs[v]), dim=1) # [batch_size, high_feature_dim]
            rs.append(r) # [batch_size, input_size]
            hs.append(h) # [batch_size, high_feature_dim]
            
        r_concat = self.concat_decoder(z_concat) # [batch_size, input_size]
        weights_expanded = weights.unsqueeze(-1)  # [batch_size, view_count, 1]
        stacked_zs = torch.stack(zs) # [view_count, batch_size, feature_dim]
        stacked_zs_transposed = stacked_zs.permute(1, 0, 2)  # [batch_size, view_count, feature_dim]
        z_weighted_sum = torch.sum(stacked_zs_transposed * weights_expanded, dim=1)  # [batch_size, feature_dim]
        H = normalize(self.feature_fusion_module(z_weighted_sum), dim=1)  # [batch_size, high_feature_dim]
        return rs, zs, hs, H, r_concat, z_concat, activations, means
    
    @staticmethod
    def multi_view_zero_value_proportion_statistics_once(xs):
        multi_view_zero_value_proportion_statistics = []
        for view in xs:
            # zero_value_counts = (torch.abs(view) < 10 ** -6).sum(dim=1).float()
            zero_value_counts = (view == 0.0).sum(dim=1).float() # Count the number of zero values in each sample, shape: (batch_size,)
            total_elements = view.shape[1] # Calculate the total number of elements in each sample, shape: (batch_size,)
            proportion = zero_value_counts / total_elements # Calculate the proportion of zero values in each sample, shape: (batch_size,)
            mean = round(torch.mean(proportion).item(), 4) # Calculate the mean of the proportion of zero values in each sample, shape: (1,)
            variance = round(torch.var(proportion, unbiased=False).item(), 4) # Calculate the variance of the proportion of zero values in each sample, shape: (1,)
            multi_view_zero_value_proportion_statistics.append((mean, variance)) # shape: (view_count, 2)
        return multi_view_zero_value_proportion_statistics # shape: (view_count, 2)

class MVCLoss(nn.Module): # CDA: Cross-View Distribution Alignment Loss
    def __init__(self, batch_size, temperature, device):
        super(MVCLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
    
    def forward(self, h_i, h_j, weight=None):
        N = self.batch_size
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature # shape: (N, N), the similarity between the positive samples (nominator)
        exp_similarity_matrix = torch.exp(similarity_matrix) # shape: (N, N), the exponential of the similarity between the positive samples (nominator)
        positive = torch.diag(exp_similarity_matrix) # shape: (N,), the exponential of the similarity between the positive samples (nominator)
        negative_mask = ~torch.diag(torch.ones(N), 0).bool().to(self.device) # shape: (N, N), the mask of the negative samples
        negative = negative_mask * exp_similarity_matrix # shape: (N, N), the exponential of the similarity between the negative samples (denominator)
        loss = -torch.log(positive / torch.sum(negative, dim=1)) # shape: (N,), infoNCE loss
        loss = loss.sum() / N # shape: (1,), average infoNCE loss
        loss = weight * loss if weight is not None else loss # shape: (1,), weighted infoNCE loss if weight is not None
        return loss

class SparseAELoss(nn.Module): # SAA: Sparse Autoencoder with Adaptive Encoding Loss
    def __init__(self, rho=0.05, beta=1.0):
        super(SparseAELoss, self).__init__()
        self.rho = rho # Sparse ratio
        self.beta = beta # Sparse strength
        self.criterion_mse = torch.nn.MSELoss()
    
    # def forward(self, mean, hidden_layer_activation, reconstructed_x, x):
    def forward(self, mean, hidden_layer_activation, reconstructed_x, x):
        sparse_coefficient = np.where(mean <= 0.01, 0, (mean - 0.01) / (1 - 0.01)) # shape: (N,), the sparse coefficient, when mean <= 0.01, sparse_coefficient = 0, otherwise, sparse_coefficient = (mean - 0.01) / (1 - 0.01) (range: [0, 1])
        sparse_beta = self.beta * sparse_coefficient # SSA Loss = beta * sparse_coefficient * KL Sparse Loss
        kl_sparse_loss = self.kl_sparse_loss(hidden_layer_activation, self.rho, sparse_beta) if sparse_beta > 0 else 0# KL Sparse Loss
        reconstruction_loss = self.criterion_mse(reconstructed_x, x) # Reconstruction Loss
        loss = reconstruction_loss + kl_sparse_loss # SSA Loss = Reconstruction Loss + KL Sparse Loss
        return loss
    
    def kl_sparse_loss(self, hidden_layer_activation, rho, sparse_beta):
        kl_total = 0.0
        for layer in range(len(hidden_layer_activation)):
            rho_hat = torch.mean(hidden_layer_activation[layer], dim=0) # shape: (feature_dim_of_the_activation_of_the_layer)
            rho_hat = torch.clamp(rho_hat, 0 + 1e-6, 1 - 1e-6) # clamp the rho_hat to avoid log(0) and inf, shape: (feature_dim_of_the_activation_of_the_layer)
            kl_loss = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)) # shape: (feature_dim_of_the_activation_of_the_layer)
            kl_loss = kl_loss.mean() # shape: (1,)
            kl_total += kl_loss
        return sparse_beta * kl_total / len(hidden_layer_activation) # KL Sparse Loss = beta * sparse_coefficient * KL Sparse Loss

def validation(model, dataset, view, data_size, class_num, pre_train=False, con_train=False):
    # model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        rs, zs, hs, H, r_concat, z_concat, activations, means = model(x_list)
    if pre_train:
        print("\nPre-train: The Sparse Autoencoder with Adaptive Encoding (SAA)")
        zs_results = []
        for v in range(view):
            nmi, ari, acc, pur = evaluate(y, KMeans(n_clusters=class_num, n_init=100).fit_predict(zs[v].cpu().numpy()))
            acc, nmi, ari, pur = float(acc), float(nmi), float(ari), float(pur)
            zs_results.append([f"View {v + 1}", acc, nmi, ari, pur])
        nmi, ari, acc, pur = evaluate(y, KMeans(n_clusters=class_num, n_init=100).fit_predict(z_concat.cpu().numpy()))
        acc, nmi, ari, pur = float(acc), float(nmi), float(ari), float(pur)
        zs_results.append(["Global", acc, nmi, ari, pur])
        # print(f"\n Pre-train: Early-fused Feature Clustering")
        # print(tabulate(zs_results, headers=["Feature", "ACC", "NMI", "ARI", "Purity"], tablefmt="grid", floatfmt=".4f"))
        print("Validation results: ACC:{:.4f}, NMI:{:.4f}, ARI:{:.4f}, Purity:{:.4f}".format(acc, nmi, ari, pur))
        return acc, nmi, pur, ari
    if con_train:
        print("\n Con-train: The Sparse Autoencoder with Adaptive Encoding (SAA) + Mean Squared Error (MSE) + Cross-View Distribution Alignment (CDA)")
        rs_results = []
        for v in range(view):
            nmi, ari, acc, pur = evaluate(y, KMeans(n_clusters=class_num, n_init=100).fit_predict(hs[v].cpu().numpy()))
            acc, nmi, ari, pur = float(acc), float(nmi), float(ari), float(pur)
            rs_results.append([f"View {v + 1}", acc, nmi, ari, pur])
        nmi, ari, acc, pur = evaluate(y, KMeans(n_clusters=class_num, n_init=100).fit_predict(H.cpu().numpy()))
        acc, nmi, ari, pur = float(acc), float(nmi), float(ari), float(pur)
        rs_results.append(["Global (H)", acc, nmi, ari, pur])
        # print(f"\nLate-fused Feature Clustering")
        # print(tabulate(rs_results, headers=["Feature", "ACC", "NMI", "ARI", "Purity"], tablefmt="grid", floatfmt=".4f"))
        print("Validation results: ACC:{:.4f}, NMI:{:.4f}, ARI:{:.4f}, Purity:{:.4f}".format(acc, nmi, ari, pur))
        return acc, nmi, pur, ari

def save_lists_to_file(epoch_list, acc_list, nmi_list, pur_list, ari_list, dataset_name, iteration, seed, learning_rate):
    file_path = os.path.join('./result', f'dataset_{dataset_name}_iter_{iteration}_seed_{seed}_lr_{learning_rate}.csv')
    with open(file_path, 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'acc', 'nmi', 'pur', 'ari'])
        for epoch, acc, nmi, pur, ari in zip(epoch_list, acc_list, nmi_list, pur_list, ari_list):
            acc = round(acc, 4); nmi = round(nmi, 4); pur = round(pur, 4); ari = round(ari, 4)
            writer.writerow([epoch, acc, nmi, pur, ari])
    print(f'Metrics have been saved at {file_path}')
    
def plot_metrics(epoch_list, acc_list, nmi_list, pur_list, ari_list, dataset_name, iteration, seed, learning_rate):
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_list, acc_list, label='ACC')
    plt.plot(epoch_list, nmi_list, label='NMI')
    plt.plot(epoch_list, pur_list, label='Purity')
    plt.plot(epoch_list, ari_list, label='ARI')
    plt.legend()
    plt.savefig(f'result/dataset_{dataset_name}_iter_{iteration}_seed_{seed}_lr_{learning_rate}.png')
    plt.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SparseMVC')
    parser.add_argument('--dataset', default='MSRCV1')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--pre_epochs", default=300)
    parser.add_argument("--con_epochs", default=300)
    parser.add_argument("--feature_dim", default=64)
    parser.add_argument("--high_feature_dim", default=20)
    parser.add_argument("--seed", default=50)
    parser.add_argument("--weight_decay", default=0.0)
    args = parser.parse_args()
    os.makedirs('result', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset; batch_size = args.batch_size; learning_rate = args.learning_rate; pre_epochs = args.pre_epochs; con_epochs = args.con_epochs;
    feature_dim = args.feature_dim; high_feature_dim = args.high_feature_dim; seed = args.seed; weight_decay = args.weight_decay
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    con_epochs = 1000 if data_size >= 2500 else 300
    pre_check_num = 100 if data_size >= 2500 else 100
    valid_check_num = 100 if data_size >= 2500 else 10
    batch_size = data_size
    
    max_iteration = 1
    for iteration in range(max_iteration):
        ## 1. Set reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.deterministic = True

        ## 2. Load data and initialize model
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model = Network(view, dims, feature_dim, high_feature_dim, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion_mvc = MVCLoss(batch_size, temperature=1.0, device=device)
        criterion_sparse = SparseAELoss(rho=0.05, beta=1.0)

        ## 3. Training
        ## 3.1 Pre-train with SAA + MSE
        print("Pre-train with SAA + MSE...")
        for epoch in range(pre_epochs):
            model.train()
            tot_loss_list = []
            for xs, _, _ in data_loader:
                loss_list = []
                xs = [x.to(device) for x in xs]; xs_concat = torch.cat(xs, dim=1)
                optimizer.zero_grad()
                rs, zs, hs, H, r_concat, z_concat, activation, means = model(xs)
                loss_list.append(criterion_sparse(sum(means)/len(means), activation[0], xs_concat, r_concat))
                for v in range(view):
                    loss_list.append(criterion_sparse(means[v], activation[v + 1], xs[v], rs[v]))
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss_list.append(loss.item())
            pretrain_loss = sum(tot_loss_list) / len(tot_loss_list)
            print('Pre Epochs[{}]'.format(epoch + 1), 'Loss:{:.6f}'.format(pretrain_loss))
            if (epoch + 1) % pre_check_num == 0:
                acc, nmi, pur, ari = validation(model, dataset, view, data_size, class_num, pre_train=True, con_train=False)
        ## 3.2 Con-train with SAA + MSE + CDA
        print("Con-train with SAA + MSE + CDA...")
        model.train()
        epoch_list = []; acc_list = []; nmi_list = []; pur_list = []; ari_list = []
        for epoch in range(con_epochs):
            tot_loss_list = []
            for xs, _, _ in data_loader:  # Iterate over the dataset
                loss_list = []
                xs = [x.to(device) for x in xs]; xs_concat = torch.cat(xs, dim=1)
                optimizer.zero_grad()
                rs, zs, hs, H, r_concat, z_concat, activation, means = model(xs)
                w_contrain = [1 / view for v in range(view)]
                w_contrain = torch.tensor(w_contrain).to(device)
                loss_list.append(criterion_sparse(sum(means)/len(means), activation[0], xs_concat, r_concat))
                for v in range(view): # If sparse ratio is low, use MSE loss is better; otherwise, use SAA loss is better
                    loss_list.append(criterion_sparse(means[v], activation[v + 1], xs[v], rs[v]))
                    loss_list.append(criterion_mvc(H, hs[v], w_contrain[v]))
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss_list.append(loss.item())
            con_loss = sum(tot_loss_list) / len(tot_loss_list)
            print('Con Epochs[{}]'.format(epoch + 1), 'Loss:{:.6f}'.format(con_loss))
            if (epoch + 1) % valid_check_num == 0:
                acc, nmi, pur, ari = validation(model, dataset, view, data_size, class_num, pre_train=False, con_train=True)
                epoch_list.append(epoch + 1); acc_list.append(acc); nmi_list.append(nmi); pur_list.append(pur); ari_list.append(ari)
        # torch.save(model.state_dict(), f'result/dataset_{dataset_name}_iter_{iteration}_seed_{seed}_lr_{learning_rate}.pth')
        save_lists_to_file(epoch_list, acc_list, nmi_list, pur_list, ari_list, dataset_name, iteration, seed, learning_rate)
        plot_metrics(epoch_list, acc_list, nmi_list, pur_list, ari_list, dataset_name, iteration, seed, learning_rate)
        
        # Update seed and learning rate
        seed = int(abs(seed + random.uniform(-100000, 100000)))
        learning_rate = abs(learning_rate + random.uniform(-0.0001, 0.0001))

