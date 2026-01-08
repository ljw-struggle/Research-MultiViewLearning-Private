#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 14:01
# @Author  : Li Xiao
# @File    : autoencoder_model.py
import os, argparse, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from _utils import MMDataset, overall_performance_report
from torch import nn
from matplotlib import pyplot as plt

class MMAE(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, a=0.4, b=0.3, c=0.3):
        super(MMAE, self).__init__()
        self.a = a; self.b = b; self.c = c; self.in_feas = in_feas_dim; self.latent = latent_dim
        self.encoder_omics_1 = nn.Sequential(nn.Linear(self.in_feas[0], self.latent), nn.BatchNorm1d(self.latent), nn.Sigmoid())
        self.encoder_omics_2 = nn.Sequential(nn.Linear(self.in_feas[1], self.latent), nn.BatchNorm1d(self.latent), nn.Sigmoid())
        self.encoder_omics_3 = nn.Sequential(nn.Linear(self.in_feas[2], self.latent), nn.BatchNorm1d(self.latent), nn.Sigmoid())
        self.decoder_omics_1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        self.decoder_omics_2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        self.decoder_omics_3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0, std=0.1)
            elif name.endswith('bias'):
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_1, omics_2, omics_3):
        encoded_omics_1 = self.encoder_omics_1(omics_1); encoded_omics_2 = self.encoder_omics_2(omics_2); encoded_omics_3 = self.encoder_omics_3(omics_3)
        latent_data = torch.mul(encoded_omics_1, self.a) + torch.mul(encoded_omics_2, self.b) + torch.mul(encoded_omics_3, self.c)
        decoded_omics_1 = self.decoder_omics_1(latent_data); decoded_omics_2 = self.decoder_omics_2(latent_data); decoded_omics_3 = self.decoder_omics_3(latent_data)
        return latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3
    
    def training_loss(self, x):
        loss_fn = nn.MSELoss()
        latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = self.forward(x[0], x[1], x[2])
        loss = self.a*loss_fn(decoded_omics_1, x[0])+ self.b*loss_fn(decoded_omics_2, x[1]) + self.c*loss_fn(decoded_omics_3, x[2])
        return loss
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data_bulk_multiomics/BRCA/', help='The data dir.')
    parser.add_argument('--output_dir', default='./result/data_bulk_multiomics/moGCN/BRCA/', help='The output dir.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--latent_dim', type=float, default=100, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', type=int, default=[100], help='number of epochs to train [default: 500]')
    parser.add_argument('--learning_rate', type=float, default=[1e-3], help='learning rate [default: 1e-3]')
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
        model = MMAE(in_feas_dim=data_features, latent_dim=args.latent_dim, a=0.4, b=0.3, c=0.3).to(device)
        # 1\ Training: optimize the latent space H to reconstruct each view
        optimizer_pre = torch.optim.Adam(model.parameters(), lr=args.learning_rate[0])
        for epoch_pre in range(args.epoch_num[0]):
            loss_pre_epoch = 0.0
            for batch_idx, (x, y, _) in enumerate(dataloader):
                x = [x[i].to(device) for i in range(len(x))]; y = y.to(device)
                optimizer_pre.zero_grad()
                loss_pre = model.training_loss(x) # sum the losses from all views
                loss_pre.backward()
                optimizer_pre.step()
            loss_pre_epoch += loss_pre.item()
            print(f'Pretraining Epoch: {epoch_pre} Loss: {loss_pre_epoch:.6f}') if epoch_pre % args.log_interval == 0 else None
        # 2\ Evaluation: evaluate the latent space H using clustering and classification
        embedding, _, _, _ = model.forward(data[0], data[1], data[2])
        multi_times_embedding_list.append(embedding.detach().cpu().numpy())
    
    overall_performance_report(multi_times_embedding_list, None, label, args.output_dir) # Evaluate the latent space H using clustering and classification
    