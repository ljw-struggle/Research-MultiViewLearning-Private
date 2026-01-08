import torch
import torch.nn as nn
import torch.nn.functional as F
import os, argparse, math, numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.spatial.distance import cdist, pdist


def load_feature_and_hyperedge(data_dir, use_mRNA=True, use_meth=True, use_miRNA=True, k_list=[10], is_prob=True, m_prob=1, edge_type='euclid'):
    def construct_hyperedge_with_KNN(X, modality, k_list=[10], is_prob=True, m_prob=1, edge_type='euclid'):
        hyperedge_mat_list = []
        for k in k_list:
            if edge_type == 'euclid':
                distance_mat = cdist(X, X, 'euclid')
                distance_mat = np.loadtxt(os.path.join(data_dir, "distance_mat_modality_{}.csv".format(modality)), delimiter=',')
                hyperedge_mat = np.zeros(distance_mat.shape)
                for center_idx in range(distance_mat.shape[0]):
                    distance_mat[center_idx, center_idx] = 0
                    distance_vec = distance_mat[center_idx]
                    distance_vec_avg = np.average(distance_vec)
                    nearest_idx = np.array(np.argsort(distance_vec)).squeeze() 
                    nearest_idx[k - 1] = center_idx if not np.any(nearest_idx[:k] == center_idx) else nearest_idx[k - 1] # add the center node to the nearest neighbors if it is not in the top k
                    for node_idx in nearest_idx[:k]:
                        hyperedge_mat[node_idx, center_idx] = np.exp(-distance_vec[node_idx] ** 2 / (m_prob * distance_vec_avg) ** 2) if is_prob else 1.0 # Gaussian kernel for computing the hyperedge weight
            hyperedge_mat_list.append(hyperedge_mat)
        return np.hstack(hyperedge_mat_list)
    
    def generate_G_from_H(H, variable_weight=False):
        # Calculate G from hypgerraph incidence matrix H, where G = DV2 * H * W * invDE * HT * DV2
        H = np.array(H) # shape: N X M, N is the number of nodes, M is the number of hyperedges
        W = np.ones(H.shape[1]) # the weight of the hyperedge
        DV = np.sum(H * W, axis=1) # the degree of the node
        DE = np.sum(H, axis=0) # the degree of the hyperedge
        # invDE = np.mat(np.diag(np.power(DE, -1))) # shape: M X M
        # invDV2 = np.mat(np.diag(np.power(DV, -0.5))) # shape: N X N
        # W = np.mat(np.diag(W)) # shape: M X M
        # H = np.mat(H) # shape: N X M # numpy.matrix is deprecated in numpy 1.20; for more details, please refer to https://numpy.org/doc/stable/reference/arrays.legacy.html#deprecation-of-numpy-matrix
        invDE = np.asmatrix(np.diag(np.power(DE, -1))) # shape: M X M
        invDV2 = np.asmatrix(np.diag(np.power(DV, -0.5))) # shape: N X N
        W = np.asmatrix(np.diag(W)) # shape: M X M
        H = np.asmatrix(H) # shape: N X M
        if variable_weight:
            return invDV2 * H, W, invDE * H.T * invDV2
        else:
            return invDV2 * H * W * invDE * H.T * invDV2 # shape: N X N

    # Load the multi-omics data and concatenate the modality-specific features.
    data_train_list = []
    data_test_list = []
    data_list = []
    for i in range(1, 4): # num_view = 3
        data_train = np.loadtxt(os.path.join(data_dir, str(i) + "_tr.csv"), delimiter=',')
        data_test = np.loadtxt(os.path.join(data_dir, str(i) + "_te.csv"), delimiter=',')
        data_train_min = np.min(data_train, axis=0, keepdims=True) 
        data_train_max = np.max(data_train, axis=0, keepdims=True)
        data_train = (data_train - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_test = (data_test - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_train_list.append(data_train.astype(float))
        data_test_list.append(data_test.astype(float))
        data_list.append(np.concatenate([data_train, data_test], axis=0)) # shape: (num_train+num_test, num_feature)
    label_train = np.loadtxt(os.path.join(data_dir, "labels_tr.csv"), delimiter=',').astype(int)
    label_test = np.loadtxt(os.path.join(data_dir, "labels_te.csv"), delimiter=',').astype(int)
    label = np.concatenate([label_train, label_test], axis=0) # shape: (num_train+num_test, )
    data_train_indices = np.arange(label_train.shape[0])
    data_test_indices = np.arange(label_train.shape[0], label_train.shape[0] + label_test.shape[0])
    # data_list: list of numpy arrays, each array is a matrix with shape (num_train+num_test, num_feature)
    # label: numpy array with shape (num_train+num_test, )
    # data_train_indices: numpy array with shape (num_train, )
    # data_test_indices: numpy array with shape (num_test, )
    print('data_list mRNA shape:', data_list[0].shape)
    print('data_list meth shape:', data_list[1].shape)
    print('data_list miRNA shape:', data_list[2].shape)
    print('label shape:', label.shape)
    print('data_train_indices shape:', data_train_indices.shape)
    print('data_test_indices shape:', data_test_indices.shape)

    # Construct the multi-omics hypergraph incidence matrix and concatenate the modality-specific hyperedges.
    hyperedge_mRNA = construct_hyperedge_with_KNN(data_list[0], modality=0, k_list=k_list, is_prob=is_prob, m_prob=m_prob, edge_type=edge_type) if use_mRNA else None
    hyperedge_meth = construct_hyperedge_with_KNN(data_list[1], modality=1, k_list=k_list, is_prob=is_prob, m_prob=m_prob, edge_type=edge_type) if use_meth else None
    hyperedge_miRNA = construct_hyperedge_with_KNN(data_list[2], modality=2, k_list=k_list, is_prob=is_prob, m_prob=m_prob, edge_type=edge_type) if use_miRNA else None
    hyperedge_multi_omics = None
    for hyperedge in [hyperedge_mRNA, hyperedge_meth, hyperedge_miRNA]:
        if hyperedge is not None:
            hyperedge_multi_omics = hyperedge if hyperedge_multi_omics is None else np.hstack((hyperedge_multi_omics, hyperedge))
    # hyperedge_multi_omics: numpy array with shape (num_train+num_test, num_hyperedge * num_view)
    print('hyperedge_multi_omics shape:', hyperedge_multi_omics.shape)
            
    # Convert the multi-omics hyperedge to pre calculated G (G = DV2 * H * W * invDE * HT * DV2)
    pre_calc_G = generate_G_from_H(hyperedge_multi_omics, variable_weight=False)
    print('pre_calculate hypegraph G shape:', pre_calc_G.shape)
    print('label shape:', label.shape)
    return data_list, label, data_train_indices, data_test_indices, hyperedge_multi_omics, pre_calc_G


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight) # node feature transformation
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(G, x) # hypergraph convolution
        return x


class HGEMMD(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.num_class = num_class
        self.dropout = dropout
        self.FeatureInfoEncoder = nn.ModuleList([nn.Linear(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([nn.Linear(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([nn.Linear(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([nn.Linear(hidden_dim[0], num_class) for _ in range(self.views)])
        self.HGNN = HGNN_conv(self.views*hidden_dim[0], hidden_dim[0])
        self.MMClasifier = []
        assert len(hidden_dim) >= 1, "The length of hidden dim need to be greater than or equal to 1."
        if len(hidden_dim) == 1:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], num_class))
        else:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], hidden_dim[1]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
            for layer in range(1, len(hidden_dim) -1):
                self.MMClasifier.append(nn.Linear(hidden_dim[layer], hidden_dim[layer+1]))
                self.MMClasifier.append(nn.ReLU())
                self.MMClasifier.append(nn.Dropout(p=dropout))
            self.MMClasifier.append(nn.Linear(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data_list, G):
        FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding = dict(), dict(), dict(), dict()
        for view in range(self.views):
            featureinfo = torch.sigmoid(self.FeatureInfoEncoder[view](data_list[view]))
            feature = self.FeatureEncoder[view](data_list[view] * featureinfo)
            feature = F.dropout(F.relu(feature), self.dropout, training=self.training)
            tcp_logit = self.TCPClassifierLayer[view](feature)
            tcp_confidence = self.TCPConfidenceLayer[view](feature)
            feature = feature * tcp_confidence
            FeatureInfo[view] = featureinfo; ModalityEmbedding[view] = feature
            TCPLogit[view] = tcp_logit; TCPConfidence[view] = tcp_confidence
        MMfeature_mmdynamics = torch.cat([i for i in ModalityEmbedding.values()], dim=1) # shape: N X (num_view*hidden_dim[0])
        MMfeature_hypergraph = self.HGNN(MMfeature_mmdynamics, G) # shape: N X hidden_dim[0]
        MMlogit = self.MMClasifier(torch.cat([MMfeature_mmdynamics, MMfeature_hypergraph], dim=1)) # shape: N X num_class
        return MMlogit, FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding, MMfeature_mmdynamics, MMfeature_hypergraph
    
    def forward_criterion(self, data_list, labeled_indices, unlabeled_indices, label, pre_calc_G, lambda_1=0.1, lambda_2=0.1):
        MMlogit, FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding, MMfeature, MMfeature_hypergraph = self.forward(data_list, pre_calc_G)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Loss for Multimodal Classifier
        MMloss = criterion(MMlogit[labeled_indices], label[labeled_indices])
        
        # Intra-Sample Learning: Multimodal Dynamics Loss
        intra_sample_loss = 0
        for view in range(self.views):
            view_pred = F.softmax(TCPLogit[view], dim=1)
            view_conf = torch.gather(input=view_pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = F.mse_loss(TCPConfidence[view].view(-1)[labeled_indices], view_conf[labeled_indices]) + criterion(TCPLogit[view][labeled_indices], label[labeled_indices])
            sparsity_loss = torch.mean(FeatureInfo[view][labeled_indices])
            intra_sample_loss = intra_sample_loss + confidence_loss + sparsity_loss
            
        # Inter-Sample Learning: Relational Consistency Loss (Semi-supervised Learning)
        # Reference: Hypergraph-enhanced Dual Semi-supervised Graph Classification
        # This loss needs to be calculated on the whole dataset, including labeled and unlabeled data. 
        # So, It could be regarded as a semi-supervised learning loss based on the whole dataset. (It does not require the label information.)
        tau = 0.1
        anchor_embeddings_original = MMfeature[labeled_indices] # shape: (num_labeled, num_view*hidden_dim[0])
        unlabeled_embeddings_original = MMfeature[unlabeled_indices] # shape: (num_unlabeled, num_view*hidden_dim[0])
        anchor_embeddings_hypergraph = MMfeature_hypergraph[labeled_indices] # shape: (num_labeled, hidden_dim[0])
        unlabeled_embeddings_hypergraph = MMfeature_hypergraph[unlabeled_indices] # shape: (num_unlabeled, hidden_dim[0])
        cos_sim_original = F.cosine_similarity(unlabeled_embeddings_original.unsqueeze(1), anchor_embeddings_original.unsqueeze(0), dim=-1) # shape: (num_unlabeled, num_labeled)
        P_u = F.softmax(cos_sim_original / tau, dim=-1)  # Similarity distribution for original view, shape: (num_unlabeled, num_labeled)
        cos_sim_hypergraph = F.cosine_similarity(unlabeled_embeddings_hypergraph.unsqueeze(1), anchor_embeddings_hypergraph.unsqueeze(0), dim=-1) # shape: (num_unlabeled, num_labeled)
        Q_u = F.softmax(cos_sim_hypergraph / tau, dim=-1)  # Similarity distribution for hypergraph view, shape: (num_unlabeled, num_labeled)
        # inter_sample_loss = torch.mean(torch.sum(P_u * torch.log(P_u / (Q_u + 1e-8)), dim=-1) + torch.sum(Q_u * torch.log(Q_u / (P_u + 1e-8)), dim=-1))
        inter_sample_loss = torch.mean(torch.mean(P_u * torch.log(P_u / (Q_u + 1e-8)), dim=-1) + torch.mean(Q_u * torch.log(Q_u / (P_u + 1e-8)), dim=-1))
        
        # Combine intra-sample and inter-sample losses
        # MMloss = MMloss + lambda_1 * intra_sample_loss + lambda_2 * inter_sample_loss
        MMloss = MMloss + lambda_1*intra_sample_loss + lambda_2*inter_sample_loss
        return MMloss

            
class HGEMMD_loss_curve_visual(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.num_class = num_class
        self.dropout = dropout
        self.FeatureInfoEncoder = nn.ModuleList([nn.Linear(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([nn.Linear(in_dim[view], hidden_dim[0]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([nn.Linear(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([nn.Linear(hidden_dim[0], num_class) for _ in range(self.views)])
        self.HGNN = HGNN_conv(self.views*hidden_dim[0], hidden_dim[0])
        self.MMClasifier = []
        assert len(hidden_dim) >= 1, "The length of hidden dim need to be greater than or equal to 1."
        if len(hidden_dim) == 1:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], num_class))
        else:
            self.MMClasifier.append(nn.Linear((self.views+1)*hidden_dim[0], hidden_dim[1]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
            for layer in range(1, len(hidden_dim) -1):
                self.MMClasifier.append(nn.Linear(hidden_dim[layer], hidden_dim[layer+1]))
                self.MMClasifier.append(nn.ReLU())
                self.MMClasifier.append(nn.Dropout(p=dropout))
            self.MMClasifier.append(nn.Linear(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data_list, G):
        FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding = dict(), dict(), dict(), dict()
        for view in range(self.views):
            featureinfo = torch.sigmoid(self.FeatureInfoEncoder[view](data_list[view]))
            feature = self.FeatureEncoder[view](data_list[view] * featureinfo)
            feature = F.dropout(F.relu(feature), self.dropout, training=self.training)
            tcp_logit = self.TCPClassifierLayer[view](feature)
            tcp_confidence = self.TCPConfidenceLayer[view](feature)
            feature = feature * tcp_confidence
            FeatureInfo[view] = featureinfo; ModalityEmbedding[view] = feature
            TCPLogit[view] = tcp_logit; TCPConfidence[view] = tcp_confidence
        MMfeature_mmdynamics = torch.cat([i for i in ModalityEmbedding.values()], dim=1) # shape: N X (num_view*hidden_dim[0])
        MMfeature_hypergraph = self.HGNN(MMfeature_mmdynamics, G) # shape: N X hidden_dim[0]
        MMlogit = self.MMClasifier(torch.cat([MMfeature_mmdynamics, MMfeature_hypergraph], dim=1)) # shape: N X num_class
        return MMlogit, FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding, MMfeature_mmdynamics, MMfeature_hypergraph
    
    def forward_criterion(self, data_list, labeled_indices, unlabeled_indices, label, pre_calc_G, lambda_1=0.1, lambda_2=0.1):
        MMlogit, FeatureInfo, TCPLogit, TCPConfidence, ModalityEmbedding, MMfeature, MMfeature_hypergraph = self.forward(data_list, pre_calc_G)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # Loss for Multimodal Classifier
        MMloss_train = criterion(MMlogit[labeled_indices], label[labeled_indices])
        MMloss_test = criterion(MMlogit[unlabeled_indices], label[unlabeled_indices])
        
        # Intra-Sample Learning: Multimodal Dynamics Loss
        intra_sample_loss_train = 0
        intra_sample_loss_test = 0
        intra_sample_loss_confidence_train = 0
        intra_sample_loss_confidence_test = 0
        intra_sample_loss_sparsity_train = 0
        intra_sample_loss_sparsity_test = 0
        for view in range(self.views):
            view_pred = F.softmax(TCPLogit[view], dim=1)
            view_conf = torch.gather(input=view_pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss_train = F.mse_loss(TCPConfidence[view].view(-1)[labeled_indices], view_conf[labeled_indices]) + criterion(TCPLogit[view][labeled_indices], label[labeled_indices])
            confidence_loss_test = F.mse_loss(TCPConfidence[view].view(-1)[unlabeled_indices], view_conf[unlabeled_indices]) + criterion(TCPLogit[view][unlabeled_indices], label[unlabeled_indices])
            sparsity_loss_train = torch.mean(FeatureInfo[view][labeled_indices])
            sparsity_loss_test = torch.mean(FeatureInfo[view][unlabeled_indices])
            intra_sample_loss_train = intra_sample_loss_train + confidence_loss_train + sparsity_loss_train
            intra_sample_loss_test = intra_sample_loss_test + confidence_loss_test + sparsity_loss_test
            intra_sample_loss_confidence_train = intra_sample_loss_confidence_train + confidence_loss_train
            intra_sample_loss_confidence_test = intra_sample_loss_confidence_test + confidence_loss_test
            intra_sample_loss_sparsity_train = intra_sample_loss_sparsity_train + sparsity_loss_train
            intra_sample_loss_sparsity_test = intra_sample_loss_sparsity_test + sparsity_loss_test
            
        # Inter-Sample Learning: Relational Consistency Loss 
        tau = 0.1
        anchor_embeddings_original = MMfeature[labeled_indices] # shape: (num_labeled, num_view*hidden_dim[0])
        unlabeled_embeddings_original = MMfeature[unlabeled_indices] # shape: (num_unlabeled, num_view*hidden_dim[0])
        anchor_embeddings_hypergraph = MMfeature_hypergraph[labeled_indices] # shape: (num_labeled, hidden_dim[0])
        unlabeled_embeddings_hypergraph = MMfeature_hypergraph[unlabeled_indices] # shape: (num_unlabeled, hidden_dim[0])
        cos_sim_original = F.cosine_similarity(unlabeled_embeddings_original.unsqueeze(1), anchor_embeddings_original.unsqueeze(0), dim=-1) # shape: (num_unlabeled, num_labeled)
        P_u = F.softmax(cos_sim_original / tau, dim=-1)  # Similarity distribution for original view, shape: (num_unlabeled, num_labeled)
        cos_sim_hypergraph = F.cosine_similarity(unlabeled_embeddings_hypergraph.unsqueeze(1), anchor_embeddings_hypergraph.unsqueeze(0), dim=-1) # shape: (num_unlabeled, num_labeled)
        Q_u = F.softmax(cos_sim_hypergraph / tau, dim=-1)  # Similarity distribution for hypergraph view, shape: (num_unlabeled, num_labeled)
        # inter_sample_loss = torch.mean(torch.sum(P_u * torch.log(P_u / (Q_u + 1e-8)), dim=-1) + torch.sum(Q_u * torch.log(Q_u / (P_u + 1e-8)), dim=-1))
        inter_sample_loss = torch.mean(torch.mean(P_u * torch.log(P_u / (Q_u + 1e-8)), dim=-1) + torch.mean(Q_u * torch.log(Q_u / (P_u + 1e-8)), dim=-1))
        
        # Combine intra-sample and inter-sample losses
        # MMloss = MMloss + lambda_1 * intra_sample_loss + lambda_2 * inter_sample_loss
        FinalLoss = MMloss_train + lambda_1*intra_sample_loss_train + lambda_2*inter_sample_loss
        return FinalLoss, MMloss_train, MMloss_test, intra_sample_loss_confidence_train, intra_sample_loss_confidence_test, intra_sample_loss_sparsity_train, intra_sample_loss_sparsity_test, inter_sample_loss