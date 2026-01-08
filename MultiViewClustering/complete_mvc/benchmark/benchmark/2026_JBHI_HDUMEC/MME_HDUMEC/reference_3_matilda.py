import os, h5py, scipy, random, argparse
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from _utils import MMDataset, overall_performance_report

class matilda(nn.Module):
    def __init__(self, embed_dim=200, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[185, 30, 185]):
        super(matilda, self).__init__()
        self.embed_dim = embed_dim; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims
        self.weights_modality_list = nn.ParameterList([nn.Parameter(torch.rand((1, self.feature_dims[i])) * 0.001, requires_grad=True) for i in range(num_views)])
        self.encoder_list = nn.ModuleList([nn.Sequential(nn.Linear(feature_dims[i], hidden_dims[i]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[i]), nn.Dropout(0.2)) for i in range(num_views)])
        self.fusion_net = nn.Sequential(nn.Linear(sum(hidden_dims), embed_dim), nn.ReLU(), nn.BatchNorm1d(embed_dim), nn.Dropout(0.2))
        self.fusion_net_mean = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.BatchNorm1d(embed_dim))
        self.fusion_net_logvar = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.BatchNorm1d(embed_dim))
        self.decoder_list = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, feature_dims[i]), nn.ReLU(), nn.BatchNorm1d(feature_dims[i])) for i in range(num_views)])
    
    def forward(self, x):
        encoded_output_list = [self.encoder_list[i](x[i]*self.weights_modality_list[i]) for i in range(self.num_views)]
        fusion_output = self.fusion_net(torch.cat(encoded_output_list, dim=1))
        mean = self.fusion_net_mean(fusion_output)
        logvar = self.fusion_net_logvar(fusion_output)
        z = self.reparameterize(mean, logvar)
        decoded_output_list = [self.decoder_list[i](z) for i in range(self.num_views)]
        return decoded_output_list, mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def reconstruction_loss(self, x):
        decoded_output_list, mean, logvar = self.forward(x)
        reconstruction_loss = sum([F.mse_loss(decoded_output_list[v], x[v], reduction='mean') for v in range(self.num_views)])
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kld_loss
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/Matilda/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', default=100, type=int, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', default=64, type=int, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', default=[30], type=int, help='number of epochs to train [default: [200, 100, 100]]')
    parser.add_argument('--learning_rate', default=[0.02], type=float, help='learning rate [default: [5e-3, 1e-3, 1e-3]]')
    parser.add_argument('--log_interval', default=10, type=int, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--times', default=5, type=int, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    for t in range(args.times):
        dataset = MMDataset(args.data_dir, concat_data=False); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
        data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = matilda(embed_dim=args.latent_dim, num_views=data_views, feature_dims=data_features, hidden_dims=[185, 30, 185]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
        for epoch in range(args.epoch_num[0]):
            model.train()
            losses = []
            for batch_idx, (x, y, _) in enumerate(dataloader):
                x = [x[i].to(device) for i in range(len(x))]; y = y.to(device)
                optimizer.zero_grad()
                loss = model.reconstruction_loss(x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f'Training Epoch: {epoch} Loss: {np.mean(losses):.4f}')
        model.eval()
        _, mean, _ = model.forward(data)
        multi_times_embedding_list.append(mean.detach().cpu().numpy())
    
    overall_performance_report(multi_times_embedding_list, None, label, args.output_dir)

    