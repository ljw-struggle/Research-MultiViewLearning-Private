# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class GENELink(nn.Module):
    def __init__(self, input_dim, hidden_1_dim, hidden_2_dim, hidden_3_dim, output_dim, num_head_1, num_head_2, alpha=0.2, causal_inference=True, reduction='concate'):
        super(GENELink, self).__init__()
        self.input_dim = input_dim
        assert reduction in ['mean', 'concate']
        if reduction == 'mean':
            self.hidden_1_dim = hidden_1_dim
            self.hidden_2_dim = hidden_2_dim
        elif reduction == 'concate':
            self.hidden_1_dim = num_head_1 * hidden_1_dim
            self.hidden_2_dim = num_head_2 * hidden_2_dim
        self.hidden_3_dim = hidden_3_dim
        self.output_dim = output_dim
        self.num_head_1 = num_head_1
        self.num_head_2 = num_head_2
        self.alpha = alpha
        self.causal_inference = causal_inference
        self.reduction = reduction
        
        self.ConvLayer1 = [AttentionLayer(input_dim, hidden_1_dim, alpha) for _ in range(num_head_1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer_1_AttentionHead_{}'.format(i), attention)
        self.ConvLayer2 = [AttentionLayer(self.hidden_1_dim, hidden_2_dim, alpha) for _ in range(num_head_2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer_2_AttentionHead_{}'.format(i), attention)

        self.tf_linear_1 = nn.Linear(self.hidden_2_dim, hidden_3_dim)
        self.tf_linear_2 = nn.Linear(hidden_3_dim, output_dim)
        self.target_linear_1 = nn.Linear(self.hidden_2_dim, hidden_3_dim)
        self.target_linear_2 = nn.Linear(hidden_3_dim, output_dim)
        self.linear = nn.Linear(2*output_dim, 2)
        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()
        for attention in self.ConvLayer2:
            attention.reset_parameters()
        nn.init.xavier_uniform_(self.tf_linear_1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear_1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear_2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear_2.weight, gain=1.414)

    def encode(self, x, adj):
        assert self.reduction in ['mean', 'concate']
        if self.reduction =='concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1)
            x = F.elu(x)
            out = torch.cat([att(x, adj) for att in self.ConvLayer2], dim=1)
        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)
            out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]),dim=0)
        return out

    def decode(self, tf_embed, target_embed):
        self.type = 'MLP' if self.causal_inference else 'dot'
        assert self.type in ['dot', 'cosine', 'MLP']
        if self.type =='dot':
            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob, dim=1).view(-1)
        elif self.type =='cosine':
            prob = torch.cosine_similarity(tf_embed, target_embed, dim=1).view(-1)

        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)
        return prob

    def forward(self, x, adj, train_sample):
        embed = self.encode(x, adj)

        tf_embed = self.tf_linear_1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01) # training=self.training
        tf_embed = self.tf_linear_2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)

        target_embed = self.target_linear_1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01) # training=self.training
        target_embed = self.target_linear_2(target_embed)
        target_embed = F.leaky_relu(target_embed)

        self.tf_ouput = tf_embed
        self.target_output = target_embed

        train_tf = tf_embed[train_sample[:, 0]]
        train_target = target_embed[train_sample[:, 1]]
        
        pred = self.decode(train_tf, train_target)
        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output

    
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim, 1)))
        self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node, adj):
        v = torch.matmul(node, self.weight)
        e = self.prepare_attentional_mechanism_input(v)

        mask = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense()>0, e, mask)
        attention = F.softmax(attention, dim=1)
        # attention = F.softmax(e, dim=1)

        attention = F.dropout(attention, training=self.training)
        output_data = torch.matmul(attention, v)

        output_data = F.leaky_relu(output_data, negative_slope=self.alpha)
        output_data = F.normalize(output_data, p=2, dim=1)        
        output_data = output_data + self.bias
        return output_data
    
    def prepare_attentional_mechanism_input(self, node):
        Q = torch.matmul(node, self.a[:self.output_dim, :]) # shape: (N, 1)
        K = torch.matmul(node, self.a[self.output_dim:, :]) # shape: (N, 1)
        e = F.leaky_relu(Q + K.T, negative_slope=self.alpha) # shape: (N, N)
        return e
    


# # GENELink for different architecture
# class GENELink_Graph(nn.Module):
#     def __init__(self, input_dim, embed_dim=128, head_num=3, architecture='multi_layer', model_resume_path=None):
#         super(GENELink_Graph, self).__init__()
#         assert architecture in ['multi_layer', 'multi_level'], 'Architecture not supported'
        
#         if architecture == 'multi_layer':
#             self.model = GENELink_Graph_Multi_Layer(input_dim, embed_dim, embed_dim//2, embed_dim//4, embed_dim//8, head_num, head_num, 0.2)
#         if architecture == 'multi_level':
#             self.model = GENELink_Graph_Multi_Level(input_dim, embed_dim, embed_dim//2, embed_dim//4, embed_dim//8, head_num, head_num, 0.2)

#         if model_resume_path is not None:
#             graph_model_checkpoint = torch.load(model_resume_path)
#             self.load_state_dict(graph_model_checkpoint['model_state_dict'])
    
#     def forward(self, node, edge_index_encode, edge_index_decode):
#         return self.model.forward(node, edge_index_encode, edge_index_decode)
    
#     def forward_feature(self, node, edge_index_encode, edge_index_decode):
#         return self.model.forward_feature(node, edge_index_encode, edge_index_decode)
    

# class GENELink_Graph_Multi_Layer(nn.Module):
#     def __init__(self, input_dim, hidden_1_dim=128, hidden_2_dim=64, hidden_3_dim=32, output_dim=16, num_head_1=3, num_head_2=3, alpha=0.2):
#         super(GENELink_Graph_Multi_Layer, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_1_dim = hidden_1_dim
#         self.hidden_2_dim = hidden_2_dim
#         self.hidden_3_dim = hidden_3_dim
#         self.output_dim = output_dim

#         self.num_head_1 = num_head_1
#         self.num_head_2 = num_head_2
        
#         self.alpha = alpha

#         self.ConvLayer1 = [AttentionLayer(input_dim, self.hidden_1_dim, self.alpha) for _ in range(self.num_head_1)]
#         for i, attention in enumerate(self.ConvLayer1):
#             self.add_module('ConvLayer1_AttentionHead{}'.format(i), attention)

#         self.ConvLayer2 = [AttentionLayer(self.hidden_1_dim * self.num_head_1, self.hidden_2_dim, self.alpha) for _ in range(self.num_head_2)]
#         for i, attention in enumerate(self.ConvLayer2):
#             self.add_module('ConvLayer2_AttentionHead{}'.format(i), attention)

#         self.tf_linear1 = nn.Linear(self.hidden_2_dim * self.num_head_2, self.hidden_3_dim)
#         self.tf_linear2 = nn.Linear(self.hidden_3_dim, self.output_dim)

#         self.target_linear1 = nn.Linear(self.hidden_2_dim * self.num_head_2, self.hidden_3_dim)
#         self.target_linear2 = nn.Linear(self.hidden_3_dim, self.output_dim)

#         self.linear = nn.Linear(2 * self.output_dim, 2)
#         self.reset_parameters()

#     def reset_parameters(self):
#         for attention in self.ConvLayer1:
#             attention.reset_parameters()
#         for attention in self.ConvLayer2:
#             attention.reset_parameters()
#         nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
#         nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
#         nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
#         nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

#     def forward(self, node, adj, sample):
#         node = torch.cat([att(node, adj) for att in self.ConvLayer1], dim=1)
#         node = F.elu(node)
#         node = torch.cat([att(node, adj) for att in self.ConvLayer2], dim=1)

#         tf_embed = self.tf_linear1(node)
#         tf_embed = F.leaky_relu(tf_embed)
#         tf_embed = F.dropout(tf_embed, p=0.2, training=self.training)
#         tf_embed = self.tf_linear2(tf_embed)
#         tf_embed = F.leaky_relu(tf_embed)

#         target_embed = self.target_linear1(node)
#         target_embed = F.leaky_relu(target_embed)
#         target_embed = F.dropout(target_embed, p=0.2, training=self.training)
#         target_embed = self.target_linear2(target_embed)
#         target_embed = F.leaky_relu(target_embed)

#         tf_embed = tf_embed[sample[:, 0]]
#         target_embed = target_embed[sample[:, 1]]
#         concat_embed = torch.cat([tf_embed, target_embed], dim=1)

#         logits = self.linear(concat_embed)
#         prob = F.softmax(logits, dim=1)
#         return prob

#     def forward_feature(self, node, adj, sample):
#         node = torch.cat([att(node, adj) for att in self.ConvLayer1], dim=1)
#         node = F.elu(node)
#         node = torch.cat([att(node, adj) for att in self.ConvLayer2], dim=1)

#         tf_embed = self.tf_linear1(node)
#         tf_embed = F.leaky_relu(tf_embed)
#         tf_embed = self.tf_linear2(tf_embed)
#         tf_embed = F.leaky_relu(tf_embed)

#         target_embed = self.target_linear1(node)
#         target_embed = F.leaky_relu(target_embed)
#         target_embed = self.target_linear2(target_embed)
#         target_embed = F.leaky_relu(target_embed)

#         tf_embed = tf_embed[sample[:, 0]]
#         target_embed = target_embed[sample[:, 1]]

#         return tf_embed, target_embed
    

# class GENELink_Graph_Multi_Level(nn.Module):
#     def __init__(self, input_dim, hidden_1_dim=128, hidden_2_dim=64, hidden_3_dim=32, output_dim=16, num_head_1=3, num_head_2=3, alpha=0.2):
#         super(GENELink_Graph_Multi_Level, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_1_dim = hidden_1_dim
#         self.hidden_2_dim = hidden_2_dim
#         self.hidden_3_dim = hidden_3_dim
#         self.output_dim = output_dim

#         self.num_head_1 = num_head_1
#         self.num_head_2 = num_head_2
        
#         self.alpha = alpha

#         self.ConvLayer1 = [AttentionLayer(input_dim, self.hidden_1_dim, self.alpha) for _ in range(self.num_head_1)]
#         for i, attention in enumerate(self.ConvLayer1):
#             self.add_module('ConvLayer1_AttentionHead{}'.format(i), attention)

#         self.ConvLayer2 = [AttentionLayer(self.hidden_1_dim * self.num_head_1, self.hidden_2_dim, self.alpha) for _ in range(self.num_head_2)]
#         for i, attention in enumerate(self.ConvLayer2):
#             self.add_module('ConvLayer2_AttentionHead{}'.format(i), attention)

#         self.tf_linear1 = nn.Linear(self.hidden_1_dim * self.num_head_1 + self.hidden_2_dim * self.num_head_2, self.hidden_3_dim)
#         self.tf_linear2 = nn.Linear(self.hidden_3_dim, self.output_dim)

#         self.target_linear1 = nn.Linear(self.hidden_1_dim * self.num_head_1 + self.hidden_2_dim * self.num_head_2, self.hidden_3_dim)
#         self.target_linear2 = nn.Linear(self.hidden_3_dim, self.output_dim)

#         self.linear = nn.Linear(2 * self.output_dim, 2)
#         self.reset_parameters()

#     def reset_parameters(self):
#         for attention in self.ConvLayer1:
#             attention.reset_parameters()
#         for attention in self.ConvLayer2:
#             attention.reset_parameters()
#         nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
#         nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
#         nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
#         nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

#     def forward(self, node, adj, sample):
#         node_level_1 = torch.cat([att(node, adj) for att in self.ConvLayer1], dim=1)
#         node = F.elu(node_level_1)
#         node_level_2 = torch.cat([att(node, adj) for att in self.ConvLayer2], dim=1)
#         node = torch.cat([node_level_1, node_level_2], dim=1)

#         tf_embed = self.tf_linear1(node)
#         tf_embed = F.leaky_relu(tf_embed)
#         tf_embed = F.dropout(tf_embed, p=0.2, training=self.training)
#         tf_embed = self.tf_linear2(tf_embed)
#         tf_embed = F.leaky_relu(tf_embed)

#         target_embed = self.target_linear1(node)
#         target_embed = F.leaky_relu(target_embed)
#         target_embed = F.dropout(target_embed, p=0.2, training=self.training)
#         target_embed = self.target_linear2(target_embed)
#         target_embed = F.leaky_relu(target_embed)

#         tf_embed = tf_embed[sample[:, 0]]
#         target_embed = target_embed[sample[:, 1]]
#         concat_embed = torch.cat([tf_embed, target_embed], dim=1)

#         logits = self.linear(concat_embed)
#         prob = F.softmax(logits, dim=1)
#         return prob

#     def forward_feature(self, node, adj, sample):
#         node_level_1 = torch.cat([att(node, adj) for att in self.ConvLayer1], dim=1)
#         node = F.elu(node_level_1)
#         node_level_2 = torch.cat([att(node, adj) for att in self.ConvLayer2], dim=1)
#         node = torch.cat([node_level_1, node_level_2], dim=1)

#         tf_embed = self.tf_linear1(node)
#         tf_embed = F.leaky_relu(tf_embed)
#         tf_embed = self.tf_linear2(tf_embed)
#         tf_embed = F.leaky_relu(tf_embed)

#         target_embed = self.target_linear1(node)
#         target_embed = F.leaky_relu(target_embed)
#         target_embed = self.target_linear2(target_embed)
#         target_embed = F.leaky_relu(target_embed)

#         tf_embed = tf_embed[sample[:, 0]]
#         target_embed = target_embed[sample[:, 1]]

#         return tf_embed, target_embed
        

# class AttentionLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, alpha=0.2):
#         super(AttentionLayer, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.alpha = alpha

#         self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
#         self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
#         self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim, 1)))
#         self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight.data, gain=1.414)
#         nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
#         if self.bias is not None:
#             self.bias.data.fill_(0)
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)

#     def forward(self, node, adj):
#         v = torch.matmul(node, self.weight)
#         e = self.prepare_attentional_mechanism_input(v)

#         mask = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj.to_dense()>0, e, mask)
#         attention = F.softmax(attention, dim=1)
#         # attention = F.softmax(e, dim=1)

#         attention = F.dropout(attention, training=self.training)
#         output_data = torch.matmul(attention, v)

#         output_data = F.leaky_relu(output_data, negative_slope=self.alpha)
#         output_data = F.normalize(output_data, p=2, dim=1)        
#         output_data = output_data + self.bias
#         return output_data
    
#     def prepare_attentional_mechanism_input(self, node):
#         Q = torch.matmul(node, self.a[:self.output_dim, :]) # shape: (N, 1)
#         K = torch.matmul(node, self.a[self.output_dim:, :]) # shape: (N, 1)
#         e = F.leaky_relu(Q + K.T, negative_slope=self.alpha) # shape: (N, N)
#         return e
    