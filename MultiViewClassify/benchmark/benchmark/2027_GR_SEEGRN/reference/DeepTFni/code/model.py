# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training) # shape: (n_nodes, in_features)
        output = torch.mm(input, self.weight) # shape: (n_nodes, out_features)
        output = torch.spmm(adj, output) # shape: (n_nodes, out_features)
        output = self.activation(output) # shape: (n_nodes, out_features)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, dropout, activation=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, input):
        input = F.dropout(input, self.dropout, training=self.training) # shape: (n_nodes, hidden_dim_2)
        dot_product = torch.mm(input, input.t()) # shape: (n_nodes, n_nodes)
        adj = self.activation(dot_product) # shape: (n_nodes, n_nodes)
        return adj
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.dropout) + ')'


class GCNModelVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim_1=32, hidden_dim_2=16, dropout=0):
        super(GCNModelVAE, self).__init__()
        self.gc_1 = GraphConvolution(input_dim, hidden_dim_1, dropout, activation=F.relu) # shape: (n_nodes, hidden_dim_1)
        self.gc_2 = GraphConvolution(hidden_dim_1, hidden_dim_2, dropout, activation=lambda x: x) # shape: (n_nodes, hidden_dim_2)
        self.gc_3 = GraphConvolution(hidden_dim_1, hidden_dim_2, dropout, activation=lambda x: x) # shape: (n_nodes, hidden_dim_2)
        self.decoder = InnerProductDecoder(dropout, activation=lambda x: x) # shape: (n_nodes, n_nodes)

    def forward(self, node_vectors, adj):
        mean, logvar = self.encoder(node_vectors, adj) # shape: (n_nodes, hidden_dim_2)
        z = self.reparameterize(mean, logvar) # shape: (n_nodes, hidden_dim_2)
        adj_dec = self.decoder(z) # shape: (n_nodes, n_nodes)
        return adj_dec, mean, logvar 
    
    def encoder(self, node_vectors, adj):
        temp = self.gc_1(node_vectors, adj)
        mean = self.gc_2(temp, adj)
        logvar = self.gc_3(temp, adj)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(logvar / 2) # shape: (n_nodes, hidden_dim_2)
            eps = torch.randn_like(std)
            z = mean + eps * std # shape: (n_nodes, hidden_dim_2)
            return z
        return mean
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.dropout) + ')'


def loss_function(preds, labels, mean, logvar, n_nodes, norm, pos_weight):
    # preds.shape: (n_samples)
    # labels.shape: (n_samples)
    # mean.shape: (n_nodes, hidden_dim_2)
    # logvar.shape: (n_nodes, hidden_dim_2)
    LOSS = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # Appendix B from VAE paper: KLD_Loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)  
    # KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) # shape: (1)

    # mean.shape: (n_nodes, hidden_dim_2)
    # logvar.shape: (n_nodes, hidden_dim_2)
    # KLD = -0.5 * torch.mean(1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2)) # shape: (1)
    logstd = logvar
    KLD = -0.5 / n_nodes * torch.sum(torch.mean(1 + 2 * logstd - mean.pow(2) - logstd.exp().pow(2), 1), 0) # shape: (1)
    return LOSS + KLD
