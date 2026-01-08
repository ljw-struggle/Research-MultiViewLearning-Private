# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding module.
    Reference: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # shape: (d_model / 2)
        pe = torch.zeros(max_len, d_model) # shape: (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) # shape: (max_len, d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term) # shape: (max_len, d_model / 2)
        pe = pe.unsqueeze(0).transpose(0, 1) # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe) 

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class STGRNS(nn.Module):
    def __init__(self, d_model=200, dropout=0.1):
        super().__init__()
        self.pe_layer = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=256, dropout=dropout)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 2)
        )

    def forward(self, input):
        # input: (batch_size, seq_len, input_dim)
        out = input.permute(1, 0, 2) # (seq_len, batch_size, input_dim)
        out = self.pe_layer(out) # (seq_len, batch_size, input_dim)
        out = self.encoder_layer(out) # (seq_len, batch_size, input_dim)
        out = out.transpose(0, 1) # (batch_size, seq_len, input_dim)
        stats = out.mean(dim=1) # (batch_size, input_dim)
        out = self.pred_layer(stats) # (batch_size, 2)
        return out
    
