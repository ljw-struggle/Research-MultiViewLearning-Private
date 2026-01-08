# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from .model_expression import GRNInfer_Graph
from .model_sequence import GRNInfer_Branch


class GRNInfer_Mix(BaseModel):
    def __init__(self, graph_model_args, branch_model_args, branch_model_resume_path, fusion_method='concate', dropout_rate=0, loss_cls_coeff=1, loss_conf_coeff=1):
        # Reference: https://github.com/TencentAILabHealthcare/mmdynamics
        super(GRNInfer_Mix, self).__init__()
        self.fusion_method = fusion_method
        self.dropout_rate = dropout_rate
        self.loss_cls_coeff = loss_cls_coeff
        self.loss_conf_coeff = loss_conf_coeff
        self.graph_model = GRNInfer_Graph(**graph_model_args)
        self.branch_model = GRNInfer_Branch(**branch_model_args)

        assert self.fusion_method in ['only_expression', 'concate_sequence_scratch', 'concate_sequence_resume', 'mmdynamics']
        if self.fusion_method == 'only_expression':
            self.final_classifier_layer = nn.Linear(8, 2)
            self.branch_model_resume_path = None
        if self.fusion_method == 'concate_sequence_scratch':
            self.feature_encoder_graph = nn.Linear(8, 8)
            self.feature_encoder_branch = nn.Linear(8, 8)
            self.final_classifier_layer_1 = nn.Linear(16, 8)
            self.final_classifier_layer_2 = nn.Linear(8, 2)
            self.branch_model_resume_path = None
            self.freeze_graph_branch_model(freeze_branch=True, freeze_graph=False)
        if self.fusion_method == 'concate_sequence_resume':
            self.feature_encoder_graph = nn.Linear(8, 8)
            self.feature_encoder_branch = nn.Linear(8, 8)
            self.final_classifier_layer_1 = nn.Linear(16, 8)
            self.final_classifier_layer_2 = nn.Linear(8, 2)
            self.branch_model_resume_path = branch_model_resume_path
            self.freeze_graph_branch_model(freeze_branch=True, freeze_graph=False)
        if self.fusion_method == 'mmdynamics':
            self.feature_encoder_graph = nn.Linear(8, 8)
            self.tcp_confidence_layer_graph = nn.Linear(8, 1)
            self.tcp_classifier_layer_graph = nn.Linear(8, 2)
            self.feature_encoder_branch = nn.Linear(8, 8)
            self.tcp_confidence_layer_branch = nn.Linear(8, 1)
            self.tcp_classifier_layer_branch = nn.Linear(8, 2)
            self.final_classifier_layer_1 = nn.Linear(16, 8)
            self.final_classifier_layer_2 = nn.Linear(8, 2)
            self.branch_model_resume_path = branch_model_resume_path

        self.relu = nn.LeakyReLU(1e-2)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.apply(self.init_weights)
        self.init_graph_branch_model(None, self.branch_model_resume_path)


    def forward(self, graph_input, branch_input):
        # Get Different Modality Information.
        expression_latent_feature, prob_expression = self.graph_model.forward_feature(graph_input['NODE'], graph_input['ENCODE'], graph_input['DECODE'])
        sequence_latent_feature, prob_sequence, _ = self.branch_model.forward_feature(branch_input)

        if self.fusion_method == 'only_expression':
            prob_logits = self.final_classifier_layer(expression_latent_feature)
            return prob_logits, None, None, None, None
        if self.fusion_method == 'concate_sequence_scratch':
            expression_latent_feature = self.feature_encoder_graph(expression_latent_feature)
            expression_latent_feature = self.relu(expression_latent_feature)
            expression_latent_feature = self.dropout(expression_latent_feature)

            sequence_latent_feature = self.feature_encoder_branch(sequence_latent_feature)
            sequence_latent_feature = self.relu(sequence_latent_feature)
            sequence_latent_feature = self.dropout(sequence_latent_feature)

            multi_modality_feature = torch.cat((expression_latent_feature, sequence_latent_feature), 1)
            multi_modality_feature = self.final_classifier_layer_1(multi_modality_feature)
            multi_modality_feature = self.relu(multi_modality_feature)
            prob_logits = self.final_classifier_layer_2(multi_modality_feature)
            return prob_logits, None, None, None, None
        if self.fusion_method == 'concate_sequence_resume':
            expression_latent_feature = self.feature_encoder_graph(expression_latent_feature)
            expression_latent_feature = self.relu(expression_latent_feature)
            expression_latent_feature = self.dropout(expression_latent_feature)

            sequence_latent_feature = self.feature_encoder_branch(sequence_latent_feature)
            sequence_latent_feature = self.relu(sequence_latent_feature)
            sequence_latent_feature = self.dropout(sequence_latent_feature)

            multi_modality_feature = torch.cat((expression_latent_feature, sequence_latent_feature), 1)
            multi_modality_feature = self.final_classifier_layer_1(multi_modality_feature)
            multi_modality_feature = self.relu(multi_modality_feature)
            prob_logits = self.final_classifier_layer_2(multi_modality_feature)
            return prob_logits, None, None, None, None
        if self.fusion_method == 'mmdynamics': # Multi-Modality Dynamics.
            expression_latent_feature = self.feature_encoder_graph(expression_latent_feature)
            expression_latent_feature = self.relu(expression_latent_feature)
            expression_latent_feature = self.dropout(expression_latent_feature)
            classifier_expression = self.tcp_classifier_layer_graph(expression_latent_feature) # shape: [batch_size, 2]
            confidence_expression = self.tcp_confidence_layer_graph(expression_latent_feature) # shape: [batch_size, 1]
            expression_latent_feature = expression_latent_feature * confidence_expression # shape: [batch_size, 8]

            sequence_latent_feature = self.feature_encoder_branch(sequence_latent_feature)
            sequence_latent_feature = self.relu(sequence_latent_feature)
            sequence_latent_feature = self.dropout(sequence_latent_feature)
            classifier_sequence = self.tcp_classifier_layer_branch(sequence_latent_feature) # shape: [batch_size, 2]
            confidence_sequence = self.tcp_confidence_layer_branch(sequence_latent_feature) # shape: [batch_size, 1]
            sequence_latent_feature = sequence_latent_feature * confidence_expression # shape: [batch_size, 8]

            multi_modality_feature = torch.cat((expression_latent_feature, sequence_latent_feature), 1)
            multi_modality_feature = self.final_classifier_layer_1(multi_modality_feature)
            multi_modality_feature = self.relu(multi_modality_feature)
            prob_logits = self.final_classifier_layer_2(multi_modality_feature)
            return prob_logits, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def init_graph_branch_model(self, graph_model_resume_path, branch_model_resume_path):
        if graph_model_resume_path is not None:
            graph_model_checkpoint = torch.load(graph_model_resume_path)
            self.graph_model.load_state_dict(graph_model_checkpoint['model_state_dict'])
        if branch_model_resume_path is not None:
            branch_model_checkpoint = torch.load(branch_model_resume_path)
            self.branch_model.load_state_dict(branch_model_checkpoint['model_state_dict'])

    def freeze_graph_branch_model(self, freeze_graph=True, freeze_branch=True):
        if freeze_graph:
            for param in self.graph_model.parameters():
                param.requires_grad = False
        if freeze_branch:
            for param in self.branch_model.parameters():
                param.requires_grad = False
    
    def unfreeze_graph_branch_model(self, unfreeze_graph=True, unfreeze_branch=True):
        if unfreeze_graph:
            for param in self.graph_model.parameters():
                param.requires_grad = True
        if unfreeze_branch:
            for param in self.branch_model.parameters():
                param.requires_grad = True
    
    def mmdynamics_loss(self, classifier_expression, confidence_expression, classifier_sequence, confidence_sequence, label, weight):
        # Reference: https://github.com/TencentAILabHealthcare/mmdynamics
        # classifier_expression: [batch_size, 2]
        # confidence_expression: [batch_size, 1]
        # classifier_sequence: [batch_size, 2]
        # confidence_sequence: [batch_size, 1]
        # label: [batch_size]
        # weight: [batch_size]

        pred_expression = torch.softmax(classifier_expression, dim=1) # [batch_size, 2]
        confidence_expression = torch.sigmoid(confidence_expression) # [batch_size, 1]
        prob_target_expression = torch.gather(pred_expression, 1, label.unsqueeze(1)).view(-1) # [batch_size] (TCP)
        # confidence_loss_expression = F.mse_loss(confidence_expression.view(-1), prob_target_expression, reduction='none') # [batch_size]
        # confidence_loss_expression = torch.mean(confidence_loss_expression * weight) # [1]
        # classifier_loss_expression = F.nll_loss(F.log_softmax(classifier_expression, dim=1), label, reduction='none') # [batch_size]
        # classifier_loss_expression = torch.mean(classifier_loss_expression * weight) # [1]
        confidence_loss_expression = F.mse_loss(confidence_expression.view(-1), prob_target_expression, reduction='mean') # [1]
        classifier_loss_expression = F.nll_loss(F.log_softmax(classifier_expression, dim=1), label, reduction='mean') # [1]

        pred_sequence = torch.softmax(classifier_sequence, dim=1) # [batch_size, 2]
        confidence_sequence = torch.sigmoid(confidence_sequence) # [batch_size, 1]
        prob_target_sequence = torch.gather(pred_sequence, 1, label.unsqueeze(1)).view(-1) # [batch_size] (TCP)
        # confidence_loss_sequence = F.mse_loss(confidence_sequence.view(-1), prob_target_sequence, reduction='none') # [batch_size]
        # confidence_loss_sequence = torch.mean(confidence_loss_sequence * weight) # [1]
        # classifier_loss_sequence = F.nll_loss(F.log_softmax(classifier_sequence, dim=1), label, reduction='none') # [batch_size]
        # classifier_loss_sequence = torch.mean(classifier_loss_sequence * weight) # [1]
        confidence_loss_sequence = F.mse_loss(confidence_sequence.view(-1), prob_target_sequence, reduction='mean') # [1]
        classifier_loss_sequence = F.nll_loss(F.log_softmax(classifier_sequence, dim=1), label, reduction='mean') # [1]

        classifier_loss = classifier_loss_expression + classifier_loss_sequence
        confidence_loss = confidence_loss_expression + confidence_loss_sequence
        return self.loss_cls_coeff * classifier_loss + self.loss_conf_coeff * confidence_loss
