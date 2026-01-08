import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import thrC, post_proC, err_rate

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_sizes):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_channels, hidden_channels[0], kernel_size=kernel_sizes[0], stride=2, padding=1), nn.ReLU(),
                                     nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=kernel_sizes[1], stride=2, padding=1),nn.ReLU(),
                                     nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=kernel_sizes[2], stride=2, padding=1),nn.ReLU())
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, hidden_channels, input_channels, kernel_sizes):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.ConvTranspose2d(hidden_channels[2], hidden_channels[1], kernel_size=kernel_sizes[2], stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(hidden_channels[1], hidden_channels[0], kernel_size=kernel_sizes[1], stride=2, padding=1, output_padding=1), nn.ReLU(),
                                     nn.ConvTranspose2d(hidden_channels[0], input_channels, kernel_size=kernel_sizes[0], stride=2, padding=1, output_padding=1), nn.ReLU())
        
    def forward(self, x):
        return self.decoder(x)
    
class ConvAE(nn.Module):
    def __init__(self, input_dims, hidden_dim, kernels, samples, views, device):
        super(ConvAE, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dim
        self.view_number = views
        self.encoders = []
        self.decoders = []
        self.batch_size = samples
        for v in range(views):
            self.encoders.append(Encoder(input_dims[v], hidden_dim, kernels).to(device))
            self.decoders.append(Decoder(hidden_dim, input_dims[v], kernels).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, multi_view_data):
        rec_multi_data = []
        low_level_features = []
        for v in range(self.view_number):
            one_view_data = multi_view_data[v]
            one_view_low_level_feature = self.encoders[v](one_view_data)
            one_view_rec_data = self.decoders[v](one_view_low_level_feature)
            rec_multi_data.append(one_view_rec_data)
            low_level_features.append(one_view_low_level_feature)
        return rec_multi_data, low_level_features
    
class ConvAEwithSR(nn.Module):
    def __init__(self, input_dims, hidden_dim, kernels, samples, views, device):
        super(ConvAEwithSR, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dim
        self.view_number = views
        self.encoders = []
        self.decoders = []
        self.batch_size = samples
        for v in range(views):
            self.encoders.append(Encoder(input_dims[v], hidden_dim, kernels).to(device))
            self.decoders.append(Decoder(hidden_dim, input_dims[v], kernels).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.common_exp = nn.Parameter(1.0e-8 * torch.ones(samples, samples, dtype=torch.float32).cuda(), requires_grad=True)

    def forward(self, multi_view_data):
        low_level_features = []
        low_level_rec_features = []
        rec_multi_data = []
        z_common = self.common_exp - torch.diag(torch.diag(self.common_exp))
        for v in range(self.view_number):
            one_view_data = multi_view_data[v]
            one_view_low_level_feature = self.encoders[v](one_view_data)
            low_level_features.append(one_view_low_level_feature)
            # Learn view-specific self-expressive coefficients based on low-level features
            rec_latent = torch.matmul(z_common, one_view_low_level_feature.view(self.batch_size, -1))
            rec_latent = torch.reshape(rec_latent, shape=one_view_low_level_feature.size())
            low_level_rec_features.append(rec_latent)
            # Reconstruct data based on the low-dimensional representations reconstructed by self-expression
            one_view_rec_data = self.decoders[v](rec_latent)
            rec_multi_data.append(one_view_rec_data)
        return low_level_features, low_level_rec_features, rec_multi_data, z_common
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = sio.loadmat('Data/EYB_fc.mat')
    num_views =  int(data['num_modalities'])
    dims = []
    X = []
    Img = {}
    for i in range(0, num_views):
        I = []
        modality = str(i)
        modality_data = data['modality_' + modality].astype(np.float32)
        for j in range(modality_data.shape[1]):
            temp = np.reshape(modality_data[:, j], [1, 32, 32])
            I.append(temp)
        Img[modality] = np.transpose(np.array(I), [0, 1, 3, 2])
        X.append(Img[modality])
        dims.append(X[i].shape[1])
    labels = data['Label'][0]
    labels = np.array(labels)
    num_classes = labels.max()
    batch_size = X[0].shape[0]
    lr = 3e-4
    kernel_size = [3, 3, 3]
    n_hidden = [10, 20, 30]
    dims = [1, 1, 1, 1, 1]
    all_views_tensors = [torch.from_numpy(arr).to(device) for arr in X]
    
    # 1. Pretrain AE
    model = ConvAE(input_dims=dims, hidden_dim=n_hidden, kernels=kernel_size, samples=batch_size, views=num_views, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    epoch_iter = tqdm(range(10000))
    for epoch in epoch_iter:
        total_loss = 0
        mse = torch.nn.MSELoss()
        model.train()
        optimizer.zero_grad()
        rec_data, _ = model.forward(all_views_tensors)
        loss_list = []
        for v in range(num_views):
            if v == 0:
                loss_list.append(1 * mse(rec_data[v], all_views_tensors[v]))
            else:
                loss_list.append(1 * mse(rec_data[v], all_views_tensors[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        epoch_iter.set_description(f"# Epoch {epoch}, train_loss: {total_loss:.4f}, rec loss: {loss.item():.4f}")
    plt.imshow(Img['0'][100][0])
    plt.savefig('result/original_data.png')
    plt.imshow(rec_data[0].detach().cpu().numpy()[100][0])
    plt.savefig('result/reconstructed_data.png')

    # 2. Train AE with Self-Expression
    reg1 = 1.0
    reg2 = 1.0 * 10 ** (num_classes / 10.0 - 3.0)
    epoch_iter = tqdm(range(2000))
    sr_model = ConvAEwithSR(input_dims=dims, hidden_dim=n_hidden, kernels=kernel_size, samples=batch_size, views=num_views, device=device).to(device)
    optimizer = torch.optim.Adam(sr_model.parameters(), lr=lr, weight_decay=0.)
    criterion_mse = torch.nn.MSELoss()
    parameters_initAE = dict([(name, param) for name, param in model.named_parameters()])
    for name, param in sr_model.named_parameters():
        if name in parameters_initAE:
            param_pre = parameters_initAE[name]
            param.data = param_pre.data
    for epoch in epoch_iter:
        total_loss = 0
        sr_model.train()
        latent_features, rec_latent_features, rec_data, z_common = sr_model.forward(all_views_tensors)
        rec_loss_list = []
        rec_latent_list = []
        for v in range(num_views):
            if v == 0:
                rec_loss_list.append(1 * criterion_mse(rec_data[v], all_views_tensors[v]))
                rec_latent_list.append(1 * criterion_mse(latent_features[v], rec_latent_features[v]))
            else:
                rec_loss_list.append(1 * criterion_mse(rec_data[v], all_views_tensors[v]))
                rec_latent_list.append(1 * criterion_mse(latent_features[v], rec_latent_features[v]))
        rec_data_loss = sum(rec_loss_list)
        rec_latent_loss = sum(rec_latent_list)
        reg_z_loss = torch.sum(torch.pow(z_common, 2.0))
        loss = (reg2 / 2) * rec_data_loss + 1 * reg_z_loss + (reg1 / 2) * rec_latent_loss
        if (epoch + 1) % 100 == 0:
            alpha = max(0.4 - (num_classes - 1) / 10 * 0.1, 0.1)
            Coef = thrC(z_common.detach().cpu().numpy(), alpha)
            y_hat, L = post_proC(Coef, labels.max())
            missrate_x, nmi, ari = err_rate(labels, y_hat)
            acc = 1 - missrate_x
            print("accuracy: %.4f" % acc, "NMI: %.4f" % nmi, "ARI: %.4f" % ari)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        epoch_iter.set_description(f"# Epoch {epoch}, rec_loss: {rec_data_loss:.4f}, self_exp_loss: {rec_latent_loss.item():.4f}, total_loss: {loss.item()}")
    print(f"{num_classes} subjects: ACC: {acc * 100:.4f}% NMI: {nmi * 100:.4f}% ARI: {ari * 100:.4f}%")
    fig = plt.figure(figsize=(10,8))
    ax = plt.gca()
    cax = plt.imshow(z_common.detach().cpu().numpy())
    cbar = plt.colorbar(cax, extend='both', drawedges = False)
    cbar.set_label('Intensity',size=36, weight='bold')
    cbar.ax.tick_params(labelsize=18)
    plt.savefig('result/coef.png')
    plt.imshow(rec_data[0].detach().cpu().numpy()[100][0])
    plt.savefig('result/reconstructed_data_with_sr.png')
    