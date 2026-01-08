# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from base import BaseModel

rnn_module = {
    'RNN': nn.RNN,
    'LSTM': nn.LSTM,
    'GRU': nn.GRU,
}

class GRNInfer_Branch(BaseModel):
    def __init__(self, feature='vector', rnn_mode='LSTM', rnn_layer=3, rnn_bidirection=True, model_resume_path=None):
        super(GRNInfer_Branch, self).__init__()
        assert feature in ['onehot', 'vector', 'bert'], 'feature must be one of onehot, vector, bert'
        assert rnn_mode in ['RNN', 'LSTM', 'GRU'], 'rnn_mode must be one of RNN, LSTM, GRU'
        assert rnn_layer >= 1, 'rnn_layer must be greater than 1'
        assert isinstance(rnn_bidirection, bool)
        if feature == 'onehot':
            self.input_dim_prot = 20
            self.input_dim_dna = 4
        elif feature == 'vector':
            self.input_dim_prot = 100
            self.input_dim_dna = 100
        elif feature == 'bert':
            self.input_dim_prot = 26
            self.input_dim_dna = 69
        else:
            raise ValueError('feature must be one of onehot, vector, bert')
        
        self.rnn_mode = rnn_mode
        self.rnn_layer = rnn_layer
        self.rnn_bidirection = rnn_bidirection

        self.prt_conv = nn.Conv1d(in_channels=self.input_dim_prot, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.prt_rnn = rnn_module[self.rnn_mode](input_size=64, hidden_size=64, num_layers=self.rnn_layer, batch_first=True, bidirectional=self.rnn_bidirection)
        self.dna_conv_1 = nn.Conv1d(in_channels=self.input_dim_dna, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.dna_conv_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same')
        self.prt_att_dna = nn.MultiheadAttention(embed_dim=64, num_heads=8, kdim=64, vdim=64, batch_first=True)

        self.dense_1_EXTRACTION_start = nn.Linear(64*self.rnn_layer*2 if self.rnn_bidirection else 64*self.rnn_layer, 32)
        self.dense_2_EXTRACTION_start = nn.Linear(32, 16)
        self.dense_1_EXTRACTION_end = nn.Linear(64*self.rnn_layer*2 if self.rnn_bidirection else 64*self.rnn_layer, 32)
        self.dense_2_EXTRACTION_end = nn.Linear(32, 16)

        self.dense_1_MERGE = nn.Linear(32, 16)
        self.dense_2_MERGE = nn.Linear(16, 8)
        self.dense_3_MERGE = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU(1e-2)
        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_weights)
        if model_resume_path is not None:
            graph_model_checkpoint = torch.load(model_resume_path)
            self.load_state_dict(graph_model_checkpoint['model_state_dict'])

    def forward(self, input):
        TF, TF_len, GENE = input['TF'], input['TF_len'], input['GENE']
        
        # TF Branch 
        temp_TF = TF.permute(0, 2, 1)
        temp_TF = self.prt_conv(temp_TF)
        temp_TF = temp_TF.permute(0, 2, 1)
        temp_TF_pack = rnn_utils.pack_padded_sequence(temp_TF, TF_len.to('cpu'), batch_first=True)
        if self.rnn_mode == 'LSTM': temp_TF_pack, (h_n, _) = self.prt_rnn(temp_TF_pack)
        else: temp_TF_pack, h_n = self.prt_rnn(temp_TF_pack)
        h_n = h_n.permute(1, 0, 2)
        temp_TF = h_n.contiguous().view(h_n.size(0), self.rnn_layer*2 if self.rnn_bidirection else self.rnn_layer, -1) # shape: (batch_size, rnn_layer*2, 64)

        # GENE Branch
        temp_GENE = GENE.permute(0, 2, 1)
        temp_GENE = self.dna_conv_1(temp_GENE)
        temp_GENE = self.relu(temp_GENE)
        temp_GENE = self.dna_conv_2(temp_GENE)
        temp_GENE = self.relu(temp_GENE)
        temp_GENE = temp_GENE.permute(0, 2, 1)
        temp_attn_output, temp_attn_output_weights = self.prt_att_dna(temp_TF, temp_GENE, temp_GENE)

        # Flatten
        temp_TF = temp_TF.contiguous().view(temp_TF.size(0), -1)
        temp_GENE = temp_attn_output.contiguous().view(temp_attn_output.size(0), -1)

        # Extraction Feature
        temp_TF = self.dense_1_EXTRACTION_start(temp_TF)
        temp_TF = self.relu(temp_TF)
        temp_TF = self.dense_2_EXTRACTION_start(temp_TF)
        temp_GENE = self.dense_1_EXTRACTION_end(temp_GENE)
        temp_GENE = self.relu(temp_GENE)
        temp_GENE = self.dense_2_EXTRACTION_end(temp_GENE)

        # Linear
        merge = torch.cat([temp_TF, temp_GENE], dim=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))
        return prob, temp_attn_output_weights

    def forward_feature(self, input):
        TF, TF_len, GENE = input['TF'], input['TF_len'], input['GENE']
        
        # TF Branch 
        temp_TF = TF.permute(0, 2, 1)
        temp_TF = self.prt_conv(temp_TF)
        temp_TF = temp_TF.permute(0, 2, 1)
        temp_TF_pack = rnn_utils.pack_padded_sequence(temp_TF, TF_len.to('cpu'), batch_first=True)
        if self.rnn_mode == 'LSTM': temp_TF_pack, (h_n, _) = self.prt_rnn(temp_TF_pack)
        else: temp_TF_pack, h_n = self.prt_rnn(temp_TF_pack)
        h_n = h_n.permute(1, 0, 2)
        temp_TF = h_n.contiguous().view(h_n.size(0), self.rnn_layer*2 if self.rnn_bidirection else self.rnn_layer, -1)

        # GENE Branch
        temp_GENE = GENE.permute(0, 2, 1)
        temp_GENE = self.dna_conv_1(temp_GENE)
        temp_GENE = self.relu(temp_GENE)
        temp_GENE = self.dna_conv_2(temp_GENE)
        temp_GENE = self.relu(temp_GENE)
        temp_GENE = temp_GENE.permute(0, 2, 1)
        temp_attn_output, temp_attn_output_weights = self.prt_att_dna(temp_TF, temp_GENE, temp_GENE)

        # Flatten
        temp_TF = temp_TF.contiguous().view(temp_TF.size(0), -1)
        temp_GENE = temp_attn_output.contiguous().view(temp_attn_output.size(0), -1)

        # Extraction Feature
        temp_TF = self.dense_1_EXTRACTION_start(temp_TF)
        temp_TF = self.relu(temp_TF)
        temp_TF = self.dense_2_EXTRACTION_start(temp_TF)
        temp_GENE = self.dense_1_EXTRACTION_end(temp_GENE)
        temp_GENE = self.relu(temp_GENE)
        temp_GENE = self.dense_2_EXTRACTION_end(temp_GENE)

        # Linear
        merge = torch.cat([temp_TF, temp_GENE], dim=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))

        latent_feature = merge
        return latent_feature, prob, temp_attn_output_weights

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.RNN or type(m) == nn.LSTM or type(m) == nn.GRU:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.01)
        if type(m) == nn.MultiheadAttention:
            for name, param in m.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0.01)

