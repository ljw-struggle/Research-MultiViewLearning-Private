import os, argparse, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from _utils import MMDataset, overall_performance_report
    
class HDUA(nn.Module):
    def __init__(self, embed_dim=200, num_samples=10000, num_views=3, feature_dims=[1000, 1000, 500], hidden_dims=[512, 512, 512]):
        super(HDUA, self).__init__()
        self.embed_dim = embed_dim; self.num_samples = num_samples; self.num_views = num_views; self.feature_dims = feature_dims; self.hidden_dims = hidden_dims
        self.H = torch.nn.Parameter(torch.normal(mean=torch.zeros([num_samples, embed_dim]), std=0.01), requires_grad=True) # initialize latent space H (trainable)
        # self.register_parameter('H', self.H) # register the latent space H as a parameter of the model
        self.reconstruct_net_list = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(), 
                                                                 nn.Linear(hidden_dims[i], feature_dims[i])) for i in range(num_views)]) # reconstruct each view
        self.uncertainty_net_list = nn.ModuleList([nn.Sequential(nn.Linear(embed_dim, hidden_dims[i]), nn.BatchNorm1d(hidden_dims[i]), nn.ReLU(), 
                                                                 nn.Linear(hidden_dims[i], feature_dims[i])) for i in range(num_views)]) # predict uncertainty for each view
    
    def forward(self, idx):
        reconstruct_output_list = [self.reconstruct_net_list[i](self.H[idx]) for i in range(self.num_views)] # reconstruct each view
        uncertainty_output_list = [self.uncertainty_net_list[i](self.H[idx]) for i in range(self.num_views)] # predict uncertainty for each view
        return reconstruct_output_list, uncertainty_output_list
    
    def get_embedding(self, idx):
        return self.H[idx].detach().cpu().numpy() # get the embedding of the latent space H
    
    def pretraining_loss(self, x, idx):
        x_rec, _ = self.forward(idx) # reconstruct each view
        return sum([F.mse_loss(x_rec[v], x[v], reduction='mean') for v in range(self.num_views)]) # sum the losses from all views
    
    def fine_tuning_loss(self, x, idx):
        x_rec, log_sigma_2 = self.forward(idx) # reconstruct each view and predict uncertainty
        return sum([0.5 * torch.mean((x_rec[v] - x[v])**2 * torch.exp(-log_sigma_2[v]) + log_sigma_2[v]) for v in range(self.num_views)]) # sum the losses from all views

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/HDUA/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', type=float, default=100, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', type=int, nargs='+', default=[200, 100], help='number of epochs to train [default: 500]')
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[5e-3, 1e-3], help='learning rate [default: 1e-3]')
    parser.add_argument('--log_interval', type=int, default=1, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--times', type=int, default=5, help='number of times to run the experiment [default: 30]')
    # parser.add_argument('--verbose', default=0, type=int, help='Whether to do the statistics.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed) # Set random seed for reproducibility
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    multi_times_embedding_list = []
    for t in range(args.times):
        dataset = MMDataset(args.data_dir); data = [x.clone().to(device) for x in dataset.X]; label = dataset.Y.clone().numpy()
        data_views = dataset.data_views; data_samples = dataset.data_samples; data_features = dataset.data_features; data_categories = dataset.categories
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = HDUA(embed_dim=args.latent_dim, num_samples=data_samples, num_views=data_views, feature_dims=data_features, hidden_dims=[512, 512, 512]).to(device)
        # 1\ Pretraining: optimize the latent space H to reconstruct each view
        optimizer_pre = torch.optim.Adam(list(model.reconstruct_net_list.parameters()) + [model.H], lr=args.learning_rate[0])
        for epoch_pre in range(args.epoch_num[0]):
            model.train()
            loss_pre_epoch = 0.0
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                x = [x[i].to(device) for i in range(len(x))]; y = y.to(device)
                optimizer_pre.zero_grad()
                loss_pre = model.pretraining_loss(x, idx) # sum the losses from all views
                loss_pre.backward()
                optimizer_pre.step()
                loss_pre_epoch += loss_pre.item()
            print(f'Pretraining Epoch: {epoch_pre} Loss: {loss_pre_epoch:.6f}') if epoch_pre % args.log_interval == 0 else None
        # 2\ Fine-tuning: optimize the latent space H to reconstruct each view and predict uncertainty
        optimizer = torch.optim.Adam(list(model.reconstruct_net_list.parameters()) + list(model.uncertainty_net_list.parameters()) + [model.H], lr=args.learning_rate[1])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9) # learning rate = 1e-3 * 0.9^(10/10) = 9e-4
        for epoch in range(args.epoch_num[1]):
            model.train()
            loss_epoch = 0.0
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                x = [x[i].to(device) for i in range(len(x))]; y = y.to(device)
                optimizer.zero_grad()
                loss = model.fine_tuning_loss(x, idx) # sum the losses from all views
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            scheduler.step()
            print(f'Training Epoch: {epoch} Loss: {loss_epoch:.6f}') if epoch % args.log_interval == 0 else None
        # 3\ Evaluation: evaluate the latent space H using clustering and classification
        H = model.H.detach().cpu().numpy() # get the embedding of the latent space H
        multi_times_embedding_list.append(H)
    
    overall_performance_report(multi_times_embedding_list, None, dataset.Y, args.output_dir) # Evaluate the latent space H using clustering and classification
    