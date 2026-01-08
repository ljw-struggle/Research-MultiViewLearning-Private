import sys, math, random, itertools, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from utils import load_data

activation_dict = {'sigmoid': lambda: nn.Sigmoid(), 'leakyrelu': lambda: nn.LeakyReLU(0.2, inplace=True), 'tanh': lambda: nn.Tanh(), 'relu': lambda: nn.ReLU()}

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, activation='relu', batchnorm=True):
        super(Autoencoder, self).__init__()
        encoder_dim = [input_dim] + [1024, 1024, 1024] + [latent_dim]
        decoder_dim = [latent_dim] + [1024, 1024, 1024] + [input_dim]
        encoder_layers = []
        for i in range(len(encoder_dim) - 1):
            encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < len(encoder_dim) - 2:
                encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1])) if batchnorm else None
                encoder_layers.append(activation_dict[activation]())
            if i == len(encoder_dim) - 2:
                encoder_layers.append(nn.Softmax(dim=1))
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for i in range(len(decoder_dim) - 1):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1])) if batchnorm else None
            decoder_layers.append(activation_dict[activation]())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x) # shape: (batch_size, latent_dim)

    def decode(self, latent):
        return self.decoder(latent) # shape: (batch_size, input_dim)

class Prediction(nn.Module):
    def __init__(self, latent_dim, activation='relu', batchnorm=True):
        super(Prediction, self).__init__()
        encoder_dim = [latent_dim] + [128, 256, 128]
        decoder_dim = [128, 256, 128] + [latent_dim]
        encoder_layers = []
        for i in range(len(encoder_dim) - 1):
            encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1])) if batchnorm else None
            encoder_layers.append(activation_dict[activation]())
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for i in range(len(decoder_dim) - 1):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if i < len(decoder_dim) - 2:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1])) if batchnorm else None
                decoder_layers.append(activation_dict[activation]())
            if i == len(decoder_dim) - 2:
                decoder_layers.append(nn.Softmax(dim=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

class CrossViewContrastiveLoss(nn.Module): # Cross-view Contrastive Loss: Maximize the consistency between two views
    def __init__(self, alpha=9.0):
        super(CrossViewContrastiveLoss, self).__init__()
        self.alpha = alpha # when alpha is bigger, the loss is more sensitive to the consistency between two views
        self.EPS = sys.float_info.epsilon # 2.220446049250313e-16, a small number to avoid log(0)
    
    def compute_joint(self, latent_view_1, latent_view_2): # Compute the joint probability matrix P, shape: (k, k)
        p_i_j = latent_view_1.unsqueeze(2) * latent_view_2.unsqueeze(1)  # shape: (batch_size, latent_dim, latent_dim)
        p_i_j = p_i_j.sum(dim=0)  # shape: (latent_dim, latent_dim)
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # (p_i_j + p_i_j.t()) / 2. is the symmetric matrix of p_i_j
        p_i_j = p_i_j / p_i_j.sum()  # p_i_j is the normalized joint probability matrix; equal to (p_i_j / batch_size)
        return p_i_j # shape: (latent_dim, latent_dim), sum(p_i_j) = 1
    
    def forward(self, latent_view_1, latent_view_2): # Maximize the consistency between two views, shape: (1,)
        # Cross-view Contrastive Loss: Loss = - sum(p_i_j * (log(p_i_j) - (alpha + 1) * log(p_j) - (alpha + 1) * log(p_i)))
        _, k = latent_view_1.size() # shape: (batch_size, latent_dim)
        p_i_j = self.compute_joint(latent_view_1, latent_view_2)  # shape: (latent_dim, latent_dim), sum(p_i_j) = 1
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)  # shape: (latent_dim, latent_dim), sum(p_i) = latent_dim
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # shape: (latent_dim, latent_dim), sum(p_j) = latent_dim
        p_i_j = torch.where(p_i_j < self.EPS, torch.tensor(self.EPS, device=p_i_j.device), p_i_j)  # shape: (latent_dim, latent_dim)
        p_j = torch.where(p_j < self.EPS, torch.tensor(self.EPS, device=p_j.device), p_j)  # shape: (latent_dim, latent_dim)
        p_i = torch.where(p_i < self.EPS, torch.tensor(self.EPS, device=p_i.device), p_i)  # shape: (latent_dim, latent_dim), sum(p_i) = 1
        loss = - p_i_j * (torch.log(p_i_j) - (self.alpha + 1) * torch.log(p_j) - (self.alpha + 1) * torch.log(p_i)) # shape: (latent_dim, latent_dim)
        loss = loss.sum() # shape: (1,), mutual information between two views
        return loss # shape: (1,)

def get_mask(view_num, data_size, missing_rate, seed):
    np.random.seed(seed)
    # 1. missing rate >= view_num - 1, every sample select one views to be missing
    if missing_rate >= view_num - 1:
        mask = OneHotEncoder().fit_transform(np.random.randint(0, view_num, size=(data_size, 1))).toarray()
        return mask # shape: (data_size, view_num), mask is the mask for the samples
    # 2. missing rate = 0, every sample select all views to be complete
    if missing_rate == 0:
        mask = np.ones((data_size, view_num))
        return mask # shape: (data_size, view_num), mask is the mask for the samples
    # 3. missing rate is between 0 and view_num - 1, every sample select some views to be missing
    # For example: data_size = 10, view_num = 3, missing_rate = 0.3, then the mask is: 
    # Example 1: incomplete sample number = 2
    # [[1, 1, 1],  # sample 1 
    #  [1, 1, 1],  # sample 2
    #  [1, 1, 1],  # sample 3
    #  [1, 1, 1],  # sample 4
    #  [1, 1, 1],  # sample 5
    #  [1, 1, 1],  # sample 6
    #  [1, 1, 1],  # sample 7
    #  [1, 1, 1],  # sample 8
    #  [1, 0, 1],  # sample 9
    #  [0, 1, 0]]  # sample 10
    # Example 2: incomplete sample number = 3
    # [[1, 1, 1],  # sample 1
    #  [1, 1, 1],  # sample 2
    #  [1, 1, 1],  # sample 3
    #  [1, 1, 1],  # sample 4
    #  [1, 1, 1],  # sample 5
    #  [1, 1, 1],  # sample 6
    #  [1, 1, 1],  # sample 7
    #  [1, 1, 0],  # sample 8
    #  [1, 0, 1],  # sample 9
    #  [0, 1, 1]]  # sample 10
    error = 1
    one_ratio_target = 1.0 - missing_rate / view_num
    while error >= 0.005:
        view_preserve = OneHotEncoder().fit_transform(np.random.randint(0, view_num, size=(data_size, 1))).toarray()
        # 1. First try to add one_num_to_add ones to the matrix
        one_num_to_add = (view_num - missing_rate - 1) * data_size; ratio = one_num_to_add / (view_num * data_size)
        matrix_first_try = (np.random.randint(0, 100, size=(data_size, view_num)) < int(ratio * 100)).astype(np.int64) # shape: (data_size, view_num)
        # 2. Then try to add one_num_to_add_updated ones to the matrix, one_num_to_add_updated is updated based on the error_num
        error_num = np.sum(((matrix_first_try + view_preserve) > 1).astype(np.int64)) # shape: (1,)
        one_num_to_add_updated = (one_num_to_add / (one_num_to_add - error_num)) * one_num_to_add; ratio = one_num_to_add_updated / (view_num * data_size) # shape: (1,)
        matrix_second_try = (np.random.randint(0, 100, size=(data_size, view_num)) < int(ratio * 100)).astype(np.int64) # shape: (data_size, view_num)
        # 3. Finally, get the mask
        matrix = ((matrix_second_try + view_preserve) > 0).astype(np.int64) # shape: (data_size, view_num)
        one_ratio_final = np.sum(matrix) / (view_num * data_size) # shape: (1,)
        error = abs(one_ratio_target - one_ratio_final)
    return matrix # shape: (data_size, view_num), mask matrix for the samples

def next_batch(x_1, x_2, batch_size): # batch data generator
    sample_num = x_1.shape[0]
    num_batches = math.ceil(sample_num / batch_size)
    for i in range(num_batches):
        start_idx = i * batch_size; end_idx = min(sample_num, start_idx + batch_size)
        batch_x_1 = x_1[start_idx: end_idx, ...]; batch_x_2 = x_2[start_idx: end_idx, ...]
        yield (batch_x_1, batch_x_2, (i + 1))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Caltech101-20', choices=['Caltech101-20', 'Scene_15', 'LandUse_21', 'NoisyMNIST'], help='dataset name')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension (default: based on dataset, 128 or 64)')
    parser.add_argument('--start_dual_prediction', type=int, default=100, help='Epoch to start dual prediction')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=9, help='Alpha parameter for contrastive loss')
    parser.add_argument('--lambda_1', type=float, default=0.1, help='Lambda1 for dual prediction loss')
    parser.add_argument('--lambda_2', type=float, default=0.1, help='Lambda2 for reconstruction loss')
    parser.add_argument('--missing_rate', type=float, default=0.5, help='Missing rate')
    parser.add_argument('--seed_mask', type=int, default=1, help='Seed for mask')
    parser.add_argument('--seed', type=int, default=10, help='Seed')
    args = parser.parse_args()
    dataset_name = args.dataset; latent_dim = args.latent_dim; start_dual_prediction = args.start_dual_prediction; batch_size = args.batch_size
    epochs = args.epochs; lr = args.lr; alpha = args.alpha; lambda_1 = args.lambda_1; lambda_2 = args.lambda_2; missing_rate = args.missing_rate
    seed_mask = args.seed_mask; seed = args.seed
    seed_dict = {'Caltech101-20': 4, 'Scene_15': 8, 'LandUse_21': 3, 'NoisyMNIST': 0}; seed = seed_dict[dataset_name] if dataset_name in seed_dict else seed
    latent_dim_dict = {'Caltech101-20': 128, 'Scene_15': 128, 'LandUse_21': 64, 'NoisyMNIST': 64}; latent_dim = latent_dim_dict[dataset_name] if dataset_name in latent_dim_dict else latent_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load data.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, idx = next(iter(dataloader)); x_1, x_2 = x_list[0].numpy(), x_list[1].numpy(); y_numpy = y.numpy()
    mask = get_mask(view_num=2, data_size=data_size, missing_rate=missing_rate, seed=seed_mask)
    x_1_masked = x_1 * mask[:, 0][:, np.newaxis]; x_2_masked = x_2 * mask[:, 1][:, np.newaxis]
    x_1_masked = torch.from_numpy(x_1_masked).float().to(device); x_2_masked = torch.from_numpy(x_2_masked).float().to(device); mask = torch.from_numpy(mask).long().to(device) # shape: (data_size, 2), mask is the mask for the samples
    x_1_complete = x_1_masked[mask.sum(dim=1) == 2]; x_2_complete = x_2_masked[mask.sum(dim=1) == 2] # shape: (data_size_complete, dims[0]), shape: (data_size_complete, dims[1]), x_1_complete and x_2_complete are the complete data of view 1 and view 2
        
    # 2. Set seed for model training.
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True
    
    # 3. Initialize models.
    model_ae_1 = Autoencoder(dims[0], latent_dim, 'relu', True).to(device)
    model_ae_2 = Autoencoder(dims[1], latent_dim, 'relu', True).to(device)
    model_p_1_2 = Prediction(latent_dim, 'relu', True).to(device)
    model_p_2_1 = Prediction(latent_dim, 'relu', True).to(device)
    criterion_contrastive = CrossViewContrastiveLoss(alpha=alpha).to(device)
    optimizer = torch.optim.Adam(itertools.chain(model_ae_1.parameters(), model_ae_2.parameters(), model_p_1_2.parameters(), model_p_2_1.parameters()), lr=lr)
    
    # 4. Train the model.
    for epoch in range(epochs):
        x_1_complete_shuffled, x_2_complete_shuffled = shuffle(x_1_complete, x_2_complete)
        model_ae_1.train(); model_ae_2.train(); model_p_1_2.train(); model_p_2_1.train()
        loss_total = 0
        for batch_x_1, batch_x_2, _ in next_batch(x_1_complete_shuffled, x_2_complete_shuffled, batch_size):
            z_1 = model_ae_1.encode(batch_x_1) # shape: [batch_size, latent_dim]
            z_2 = model_ae_2.encode(batch_x_2) # shape: [batch_size, latent_dim]
            # 1. Within-view Reconstruction Loss
            reconstruction_loss = F.mse_loss(model_ae_1.decode(z_1), batch_x_1) + F.mse_loss(model_ae_2.decode(z_2), batch_x_2)
            # 2. Cross-view Contrastive_Loss
            contrastive_loss = criterion_contrastive(z_1, z_2)
            # 3. Cross-view Dual-Prediction Loss
            z_2_pred = model_p_1_2(z_1) # shape: [batch_size, latent_dim]
            z_1_pred = model_p_2_1(z_2) # shape: [batch_size, latent_dim]
            dual_prediction_loss = F.mse_loss(z_2_pred, z_2) + F.mse_loss(z_1_pred, z_1)
            loss = reconstruction_loss * lambda_2 + contrastive_loss + dual_prediction_loss * lambda_1 if epoch >= start_dual_prediction else reconstruction_loss * lambda_2 + contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                model_ae_1.eval(); model_ae_2.eval(); model_p_1_2.eval(); model_p_2_1.eval()
                latent_view_1 = torch.zeros(data_size, latent_dim).to(device)
                latent_view_2 = torch.zeros(data_size, latent_dim).to(device)
                view_1_complete_idx = mask[:, 0] == 1
                view_2_complete_idx = mask[:, 1] == 1
                latent_view_1[view_1_complete_idx] = model_ae_1.encode(x_1_masked[view_1_complete_idx])
                latent_view_2[view_2_complete_idx] = model_ae_2.encode(x_2_masked[view_2_complete_idx])
                if missing_rate > 0:
                    view_1_missing_idx = mask[:, 0] == 0
                    view_2_missing_idx = mask[:, 1] == 0
                    latent_view_1[view_1_missing_idx] = model_p_2_1(model_ae_2.encode(x_2_masked[view_1_missing_idx]))
                    latent_view_2[view_2_missing_idx] = model_p_1_2(model_ae_1.encode(x_1_masked[view_2_missing_idx]))
                latent_fusion = torch.cat([latent_view_1, latent_view_2], dim=1).cpu().numpy()
                cluster_assignments = KMeans(class_num, n_init=10).fit_predict(latent_fusion)
                ami = np.round(metrics.adjusted_mutual_info_score(y_numpy, cluster_assignments), 4)
                nmi = np.round(metrics.normalized_mutual_info_score(y_numpy, cluster_assignments), 4)
                ari = np.round(metrics.adjusted_rand_score(y_numpy, cluster_assignments), 4)
            print("Epoch : {:.0f}/{:.0f}; Total Loss = {:.4f}".format(epoch + 1, epochs, loss_total))
            print("Epoch : {:.0f}/{:.0f}; AMI = {:.4f}; NMI = {:.4f}; ARI = {:.4f}".format(epoch + 1, epochs, ami, nmi, ari))
