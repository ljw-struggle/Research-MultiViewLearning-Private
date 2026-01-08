# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from torch_geometric.nn import GCNConv, GATConv, GINConv

aggregation_module = {
    'GCN': GCNConv,
    'GAT': lambda in_channels, out_channels: GATConv(in_channels, out_channels, heads=3, concat=False, dropout=0),
    'GIN': lambda in_channels, out_channels: GINConv(nn.Sequential(nn.Linear(in_channels, out_channels), nn.LeakyReLU(1e-2), nn.Linear(out_channels, out_channels)), train_eps=True),
}

class GRNInfer_Graph(BaseModel):
    def __init__(self, input_dim, embed_dim=64, aggregation_mode='GAT', architecture='multi_layer', model_resume_path=None):
        super(GRNInfer_Graph, self).__init__()
        assert aggregation_mode in aggregation_module.keys(), 'Aggregation mode not supported'
        assert architecture in ['single_layer', 'multi_layer', 'multi_level', 'multi_channel'], 'Architecture not supported'
        
        if architecture == 'single_layer':
            self.model = GRNInfer_Graph_Single_Layer(input_dim, embed_dim, aggregation_mode)
        if architecture == 'multi_layer':
            self.model = GRNInfer_Graph_Multi_Layer(input_dim, embed_dim, aggregation_mode)
        if architecture == 'multi_level':
            self.model = GRNInfer_Graph_Multi_Level(input_dim, embed_dim, aggregation_mode)
        if architecture == 'multi_channel':
            self.model = GRNInfer_Graph_Multi_Channel(input_dim, embed_dim, aggregation_mode)

        if model_resume_path is not None:
            graph_model_checkpoint = torch.load(model_resume_path)
            self.load_state_dict(graph_model_checkpoint['model_state_dict'])
    
    def forward(self, node, edge_index_encode, edge_index_decode):
        return self.model.forward(node, edge_index_encode, edge_index_decode)
    
    def forward_feature(self, node, edge_index_encode, edge_index_decode):
        return self.model.forward_feature(node, edge_index_encode, edge_index_decode)
    

class GRNInfer_Graph_Single_Layer(BaseModel):
    def __init__(self, input_dim, embed_dim=64, aggregation_mode='GAT'):
        super(GRNInfer_Graph_Single_Layer, self).__init__()
        self.embeding = nn.Linear(input_dim, embed_dim)
        self.conv_1 = aggregation_module[aggregation_mode](embed_dim, embed_dim)

        # self.norm = nn.BatchNorm1d(embed_dim)

        self.dense_1_EXTRACTION_start = nn.Linear(embed_dim, 32)
        self.dense_2_EXTRACTION_start = nn.Linear(32, 16)
        self.dense_1_EXTRACTION_end = nn.Linear(embed_dim, 32)
        self.dense_2_EXTRACTION_end = nn.Linear(32, 16)
        
        self.dense_1_MERGE = nn.Linear(32, 16)
        self.dense_2_MERGE = nn.Linear(16, 8)
        self.dense_3_MERGE = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU(1e-2)
        self.sigmoid = nn.Sigmoid()
        
        self.apply(self.init_weights)

    def forward(self, node, edge_index_encode, edge_index_decode):
        node_embed = self.embeding(node)
        node = self.conv_1(node_embed, edge_index_encode)
        # node = self.norm(node)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))
        
        return prob

    def forward_feature(self, node, edge_index_encode, edge_index_decode):
        node = self.embeding(node)
        node = self.conv_1(node, edge_index_encode)
        # node = self.norm(node)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))

        latent_feature = merge
        return latent_feature, prob
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.BatchNorm1d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class GRNInfer_Graph_Multi_Layer(BaseModel):
    def __init__(self, input_dim, embed_dim=64, aggregation_mode='GAT'):
        super(GRNInfer_Graph_Multi_Layer, self).__init__()
        self.embeding = nn.Linear(input_dim, embed_dim)
        self.conv_1 = aggregation_module[aggregation_mode](embed_dim, embed_dim)
        self.conv_2 = aggregation_module[aggregation_mode](embed_dim, embed_dim)

        # self.norm = nn.BatchNorm1d(embed_dim)

        self.dense_1_EXTRACTION_start = nn.Linear(embed_dim, 32)
        self.dense_2_EXTRACTION_start = nn.Linear(32, 16)
        self.dense_1_EXTRACTION_end = nn.Linear(embed_dim, 32)
        self.dense_2_EXTRACTION_end = nn.Linear(32, 16)
        
        self.dense_1_MERGE = nn.Linear(32, 16)
        self.dense_2_MERGE = nn.Linear(16, 8)
        self.dense_3_MERGE = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU(1e-2)
        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_weights)

    def forward(self, node, edge_index_encode, edge_index_decode):
        node_embed = self.embeding(node)
        node = self.conv_1(node_embed, edge_index_encode)
        node = self.relu(node)
        node = self.conv_2(node, edge_index_encode)
        # node = self.norm(node)
        feature_construct_loss = F.mse_loss(node_embed, node)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))
        return prob, feature_construct_loss

    def forward_feature(self, node, edge_index_encode, edge_index_decode):
        node = self.embeding(node)
        node = self.conv_1(node, edge_index_encode)
        node = self.relu(node)
        node = self.conv_2(node, edge_index_encode)
        # node = self.norm(node)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))

        latent_feature = merge
        return latent_feature, prob
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.BatchNorm1d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class GRNInfer_Graph_Multi_Level(BaseModel):
    def __init__(self, input_dim, embed_dim=64, aggregation_mode='GAT'):
        super(GRNInfer_Graph_Multi_Level, self).__init__()
        self.embeding = nn.Linear(input_dim, embed_dim)
        self.conv_1 = aggregation_module[aggregation_mode](embed_dim, embed_dim)
        self.conv_2 = aggregation_module[aggregation_mode](embed_dim, embed_dim)

        # self.norm = nn.BatchNorm1d(2 * embed_dim)

        self.dense_1_EXTRACTION_start = nn.Linear(2 * embed_dim, 32)
        self.dense_2_EXTRACTION_start = nn.Linear(32, 16)
        self.dense_1_EXTRACTION_end = nn.Linear(2 * embed_dim, 32)
        self.dense_2_EXTRACTION_end = nn.Linear(32, 16)
        
        self.dense_1_MERGE = nn.Linear(32, 16)
        self.dense_2_MERGE = nn.Linear(16, 8)
        self.dense_3_MERGE = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU(1e-2)
        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_weights)

    def forward(self, node, edge_index_encode, edge_index_decode):
        node_embed = self.embeding(node)
        node_level_1 = self.conv_1(node_embed, edge_index_encode)
        node_level_1_ = self.relu(node_level_1)
        node_level_2 = self.conv_2(node_level_1_, edge_index_encode)
        node = torch.concat([node_level_1, node_level_2], axis=1)
        # node = self.norm(node)
        feature_construct_loss = F.mse_loss(node_embed, node)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, hidden_channels + out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, hidden_channels + out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))
        return prob, feature_construct_loss

    def forward_feature(self, node, edge_index_encode, edge_index_decode):
        node = self.embeding(node)
        node_level_1 = self.conv_1(node, edge_index_encode)
        node_level_1 = self.relu(node_level_1)
        node_level_2 = self.conv_2(node_level_1, edge_index_encode)
        node = torch.concat([node_level_1, node_level_2], axis=1)
        # node = self.norm(node)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))

        latent_feature = merge
        return latent_feature, prob

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.BatchNorm1d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class GRNInfer_Graph_Multi_Channel(BaseModel):
    def __init__(self, input_dim, embed_dim=64, aggregation_mode='GCN'):
        super(GRNInfer_Graph_Multi_Channel, self).__init__()
        self.embeding = nn.Linear(input_dim, embed_dim)
        self.conv_1 = aggregation_module[aggregation_mode](embed_dim, embed_dim)
        self.conv_2 = aggregation_module[aggregation_mode](embed_dim, embed_dim)
        self.norm = nn.BatchNorm1d(2 * embed_dim)
        
        self.dense_1_EXTRACTION_start = nn.Linear(2 * embed_dim, 32)
        self.dense_2_EXTRACTION_start = nn.Linear(32, 16)
        self.dense_1_EXTRACTION_end = nn.Linear(2 * embed_dim, 32)
        self.dense_2_EXTRACTION_end = nn.Linear(32, 16)
        
        self.dense_1_MERGE = nn.Linear(32, 16)
        self.dense_2_MERGE = nn.Linear(16, 8)
        self.dense_3_MERGE = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU(1e-2)
        self.sigmoid = nn.Sigmoid()

        self.apply(self.init_weights)

    def forward(self, node, edge_index_encode, edge_index_decode):
        node_embed = self.embeding(node)
        node_channel_1 = self.conv_1(node, edge_index_encode)
        node_channel_2 = self.conv_2(node, edge_index_encode)
        node = torch.concat([node_channel_1, node_channel_2], axis=1)
        # node = self.norm(node)
        feature_construct_loss = F.mse_loss(torch.concat([node_channel_1, node_channel_2], axis=1), node_embed)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, 2 * out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, 2 * out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))
        return prob, feature_construct_loss

    def forward_feature(self, node, edge_index_encode, edge_index_decode):
        node = self.embeding(node)
        node_channel_1 = self.conv_1(node, edge_index_encode)
        node_channel_2 = self.conv_2(node, edge_index_encode)
        node = torch.concat([node_channel_1, node_channel_2], axis=1)
        # node = self.norm(node)

        node_start = node[edge_index_decode[0]] # shape = (decode_edge_num, out_channels)
        node_end = node[edge_index_decode[1]] # shape = (decode_edge_num, out_channels)
        node_start = self.dense_1_EXTRACTION_start(node_start)
        node_start = self.relu(node_start)
        node_start = self.dense_2_EXTRACTION_start(node_start)
        node_end = self.dense_1_EXTRACTION_end(node_end)
        node_end = self.relu(node_end)
        node_end = self.dense_2_EXTRACTION_end(node_end)

        merge = torch.concat([node_start, node_end], axis=1)
        merge = self.dense_1_MERGE(merge)
        merge = self.relu(merge)
        merge = self.dense_2_MERGE(merge)
        merge = self.relu(merge)
        logit = self.dense_3_MERGE(merge)
        prob = self.sigmoid(torch.flatten(logit))

        latent_feature = merge
        return latent_feature, prob

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.BatchNorm1d:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
