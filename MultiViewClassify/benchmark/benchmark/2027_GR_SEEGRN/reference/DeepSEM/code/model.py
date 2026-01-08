# -*- coding: utf-8 -*-
# Reference: (GMVAE) https://github.com/jariasf/GMVAE/tree/master/pytorch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional  as F


class Gaussian(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Gaussian, self).__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.logvar = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        mean = self.mean(input) # shape: (batch_size, n_gene, output_dim)
        logvar = self.logvar(input) # shape: (batch_size, n_gene, output_dim)
        return mean, logvar


class Inference(nn.Module):
    def __init__(self):
        super(Inference, self).__init__()
        self.inference = torch.nn.ModuleList([nn.Linear(1, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), Gaussian(128, 1)])

    def forward(self, x, one_minus_adj_t):
        # x.shape: (batch_size, n_gene)
        # one_minus_adj_t.shape: (n_gene, n_gene)
        x = x.unsqueeze(2) # shape: (batch_size, n_gene, 1)
        x = self.inference[0](x) # shape: (batch_size, n_gene, 128)
        x = self.inference[1](x) # shape: (batch_size, n_gene, 128)
        x = self.inference[2](x) # shape: (batch_size, n_gene, 128)
        x = self.inference[3](x) # shape: (batch_size, n_gene, 128)
        mean, logvar = self.inference[4](x) # mean.shape: (batch_size, n_gene, 1) \ logvar.shape: (batch_size, n_gene, 1)

        mean = mean.squeeze(2) # shape: (batch_size, n_gene)
        logvar = logvar.squeeze(2) # shape: (batch_size, n_gene)
        mean = torch.matmul(mean, one_minus_adj_t) # shape: (batch_size, n_gene)
        logvar = torch.matmul(logvar, one_minus_adj_t) # shape: (batch_size, n_gene)
        z = self.reparameterize(mean, logvar) # shape: (batch_size, n_gene)
        return z, mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)
        noise = torch.randn_like(std)
        z = mean + noise * std
        return z


class Generation(nn.Module):
    def __init__(self):
        super(Generation, self).__init__()
        self.generation = torch.nn.ModuleList([nn.Linear(1, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1)])

    def forward(self, z, one_minus_adj_t_inv):
        # z.shape: (batch_size, n_gene)
        # one_minus_adj_t_inv.shape: (n_gene, n_gene)
        z = torch.matmul(z, one_minus_adj_t_inv) # shape: (batch_size, n_gene)

        z = z.unsqueeze(2) # shape: (batch_size, n_gene, 1)
        z = self.generation[0](z) # shape: (batch_size, n_gene, 128)
        z = self.generation[1](z) # shape: (batch_size, n_gene, 128)
        z = self.generation[2](z) # shape: (batch_size, n_gene, 128)
        z = self.generation[3](z) # shape: (batch_size, n_gene, 128)
        x_recon = self.generation[4](z) # shape: (batch_size, n_gene, 1)
        x_recon = x_recon.squeeze(2) # shape: (batch_size, n_gene)
        return x_recon


class DeepSEM(nn.Module):
    def __init__(self, n_gene, alpha, beta):
        super(DeepSEM, self).__init__()
        self.n_gene = n_gene
        self.alpha = alpha
        self.beta = beta
        mask = torch.tensor(np.ones((n_gene, n_gene)) - np.eye(n_gene), dtype=torch.float32, requires_grad=False).cuda()
        self.adj_parameter = nn.Parameter(torch.tensor(np.ones((n_gene, n_gene)), dtype=torch.float32).cuda(), requires_grad=True).cuda()
        torch.nn.init.xavier_normal_(self.adj_parameter)
        self.adj = self.adj_parameter * mask
        self.inference = Inference()
        self.generation = Generation()
        self.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias.data is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, dropout_mask):
        # x.shape: (batch_size, n_gene)
        # dropout_mask.shape: (batch_size, n_gene)
        one_minus_adj_t = torch.tensor(np.eye(self.n_gene), dtype=torch.float32, requires_grad=False).cuda() - self.adj.transpose(0, 1).cuda()
        one_minus_adj_t_inv = torch.inverse(one_minus_adj_t)

        z, mean, logvar = self.inference(x, one_minus_adj_t)
        x_recon = self.generation(z, one_minus_adj_t_inv)

        loss_reconstruction = self.loss_reconstruction(x, x_recon, dropout_mask)
        loss_kld = self.loss_kld(mean, logvar) * self.beta
        loss_constraint = self.loss_constraint(self.adj) * self.alpha
        loss = loss_reconstruction + loss_kld + loss_constraint
        return loss, x_recon, mean, logvar
    
    def loss_reconstruction(self, label, pred, dropout_mask):
        loss = torch.sum((label - pred).pow(2) * dropout_mask) / torch.sum(dropout_mask) # mean square error
        return loss

    def loss_kld(self, mean, logvar):
        loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return loss
    
    def loss_constraint(self, adj):
        loss = torch.mean(torch.abs(adj))
        return loss
    