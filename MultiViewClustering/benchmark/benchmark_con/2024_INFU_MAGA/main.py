import math, faiss, random, argparse, scipy.sparse as sp, numpy as np # faiss: (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from _utils import evaluate, load_data

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim, 500), nn.PReLU(), nn.Linear(500, 500), nn.PReLU(), nn.Linear(500, 2000), nn.PReLU(), nn.Linear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)
        
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.PReLU(), nn.Linear(2000, 500), nn.PReLU(), nn.Linear(500, 500), nn.PReLU(), nn.Linear(500, input_dim))

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, neighbor_num):
        super(Network, self).__init__()
        self.view = view; self.class_num = class_num; self.neighbor_num = neighbor_num; self.alpha = 1.0
        self.encoders = []; self.decoders = []; self.embedding_projs = []; self.graph_embedding_projs = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
            self.embedding_projs.append(nn.Sequential(nn.Linear(feature_dim, high_feature_dim)))
            self.graph_embedding_projs.append(nn.Sequential(nn.Linear(high_feature_dim, high_feature_dim), nn.PReLU()))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.embedding_projs = nn.ModuleList(self.embedding_projs)
        self.graph_embedding_projs = nn.ModuleList(self.graph_embedding_projs)
        # fusion module: TransformerEncoderLayer, TransformerEncoder, embed_proj, fusion_proj, GCNencoder, cluster_proj
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=high_feature_dim*view, nhead=1, dim_feedforward=256)
        # self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1) # not used
        # self.final_embedding_proj = nn.Sequential(nn.Linear(feature_dim, high_feature_dim)) # not used
        self.final_fusion_proj = nn.Sequential(nn.Linear(high_feature_dim*view, high_feature_dim))
        # self.final_graph_embedding_proj = nn.Sequential(nn.Linear(high_feature_dim, high_feature_dim), nn.PReLU()) # not used
        self.final_clustering_proj = nn.Sequential(nn.Linear(high_feature_dim, high_feature_dim), nn.PReLU(), nn.Linear(high_feature_dim, class_num), nn.Softmax(dim=1))
        
    def forward(self, xs):
        zs = []; hs = []; rs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            r = self.decoders[v](z)
            f = self.embedding_projs[v](z)
            zs.append(z); hs.append(f); rs.append(r)
        return zs, hs, rs

    def forward_fusion(self, xs):
        hs = []; qs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            h = F.normalize(self.graph_embedding_projs[v](self.spectral_graph_convolution(self.embedding_projs[v](z), self.neighbor_num, degree=16, alpha=0.9)), dim=1) # shape: (batch_size, out_feature)
            q = self.final_clustering_proj(h) # shape: (batch_size, class_num)
            hs.append(h); qs.append(q)
        h_concat = torch.cat(hs, dim=1).unsqueeze(dim=1) # shape: (batch_size, 1, high_feature_dim*view)
        h_fusion = F.normalize(self.final_fusion_proj(self.TransformerEncoderLayer(h_concat).squeeze(dim=1)), dim=1) # shape: (batch_size, high_feature_dim)
        q_fusion = self.final_clustering_proj(h_fusion) # shape: (batch_size, class_num)    
        return hs, h_fusion, qs, q_fusion
    
    def spectral_graph_convolution(self, features, neighbor_num, degree=16, alpha=0.9):
        # Get the embedding which is a weighted combination of the original embedding and the embedding after multiple graph convolutions.
        embedding_graph = alpha * features # shape: (batch_size, feature_dim)
        adjacency_matrix = self.build_adjacency_matrix(features.detach().cpu().numpy(), neighbor_num).to(features.device) # shape: (batch_size, batch_size)
        for _ in range(degree):
            features = torch.spmm(adjacency_matrix, features) # shape: (batch_size, feature_dim), sparse matrix multiplication
            embedding_graph = embedding_graph + (1-alpha) * features / degree # shape: (batch_size, feature_dim)
        return embedding_graph
    
    @staticmethod # input numpy array, output torch sparse tensor
    def build_adjacency_matrix(features, k): # feature: (batch_size, feature_dim), k: number of nearest neighbors
        # 1. Search the nearest neighbor by brute force.
        index = faiss.IndexFlatL2(features.shape[1]) # use L2 distance to search the nearest neighbor by brute force
        index.add(features) # add the instance to the index, shape: (batch_size, feature_dim)
        _, ind = index.search(features, k + 1) # search the (k+1)-nearest neighbor, shape: (batch_size, k+1)
        # 2. Calculate the distance between the instance and the k-nearest neighbor (exclude the instance itself).
        dist = np.array([np.linalg.norm(features[i]-features[ind[i][1:]], axis=1) for i in range(features.shape[0])]) # shape: (batch_size, k)
        # 3. Calculate the adjacency matrix using the Gaussian kernel.
        affinity_matrix = np.exp(-dist ** 2 / 2) # affinity matrix using the Gaussian kernel, shape: (batch_size, k)
        adjacency_matrix = np.zeros((features.shape[0], features.shape[0])) # shape: (batch_size, batch_size)
        for i in range(features.shape[0]):
            adjacency_matrix[i, ind[i][1:]] = affinity_matrix[i]; adjacency_matrix[ind[i][1:], i] = affinity_matrix[i]
        # 4. Normalize the adjacency matrix.
        # AugNormAdj: A' = (D + I)^-1/2 * (A + I) * (D + I)^-1/2; NormAdj: A' = (D)^-1/2 * (A) * (D)^-1/2
        # AugNormAdj: adjacency_matrix = adjacency_matrix + np.eye(adjacency_matrix.shape[0]) # Augment the adjacency matrix with self-loops.
        adjacency_matrix = sp.coo_matrix(adjacency_matrix) # Convert a numpy array to a scipy sparse matrix.
        row_sum = np.array(adjacency_matrix.sum(1)) # shape: (batch_size,)
        d_inv_sqrt = np.power(row_sum, -0.5) # shape: (batch_size,)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # set the inf value to 0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # diagonal matrix, shape: (batch_size, batch_size)
        normalized_adjacency_matrix = d_mat_inv_sqrt.dot(adjacency_matrix).dot(d_mat_inv_sqrt).tocoo().astype(np.float32) # normalized adjacency matrix, sparse matrix, shape: (batch_size, batch_size)
        # 5. Convert the normalized adjacency matrix to a torch sparse tensor.
        indices = torch.from_numpy(np.vstack((normalized_adjacency_matrix.row, normalized_adjacency_matrix.col)).astype(np.int64)) # shape: (2, num_nonzero)
        values = torch.from_numpy(normalized_adjacency_matrix.data) # shape: (num_nonzero,)
        shape = torch.Size(normalized_adjacency_matrix.shape) # shape: (batch_size, batch_size)
        normalized_adjacency_matrix_sparse = torch.sparse.FloatTensor(indices, values, shape).float() # shape: (batch_size, batch_size)
        return normalized_adjacency_matrix_sparse
    
class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, lambda_1, lambda_2, eta, device):  
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.device = device
        self.lambda_1 = lambda_1; self.lambda_2 = lambda_2; self.eta = eta
        # self.softmax = nn.Softmax(dim=1) # not used
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # input logits and labels, output the loss

    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask

    def forword_feature(self, h_i, h_j):
        feature_num = h_i.shape[0] # shape: (feature_dim, batch_size); Attention: not implement as the MFLVC. This is a bug!!!
        N = 2 * feature_num
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T)/self.temperature_f
        sim_i_j = torch.diag(sim, diagonal=feature_num)
        sim_j_i = torch.diag(sim, diagonal=-feature_num)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.get_correlated_mask(N)
        negative_samples = sim[mask].reshape(N,-1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss_contrast = self.criterion(logits, labels)
        loss_contrast /= N 
        return self.lambda_1 * loss_contrast

    def forward_pui_label(self, ologits, plogits): 
        # Partition Uncertainty Index Loss: ologits: original logits, plogits: perturbed logits; shape: (batch_size, class_num)
        # loss_ce: cross entropy loss to measure the cluster alignment, loss_ne: negative entropy loss to measure the cluster balance.
        assert ologits.shape == plogits.shape, ('Inputs are required to have same shape')
        # ologits = F.softmax(ologits); plogits = F.softmax(plogits) # not used
        similarity = torch.mm(F.normalize(ologits.t(), p=2, dim=1), F.normalize(plogits, p=2, dim=0)) # shape: (class_num, class_num), similarity is the cosine similarity matrix
        loss_ce = self.criterion(similarity, torch.arange(similarity.size(0)).to(self.device)) # shape: (1,), loss_ce is the loss for the similarity matrix
        o = ologits.sum(0).view(-1) # shape: (class_num,), o is the probability of the cluster assignments of the view i
        o /= o.sum() # shape: (class_num,), o is the normalized probability of the cluster assignments of the view i
        loss_ne = math.log(o.size(0)) + (o * o.log()).sum() # shape: (1,), loss_ne is the negative entropy of the cluster assignments of the view i
        return self.lambda_2 * loss_ce + self.eta * loss_ne # shape: (1,), loss is the total loss

    def forword_debiased_instance(self, h, h_i, y_pred, temperature=0.5, neg_size=128):
        y_sam = torch.LongTensor(y_pred) # shape: (batch_size,), y_sam is the label of the instance
        class_sam = []
        for m in range(np.max(y_pred) + 1): # shape: (class_num,), m is the label of the instance
            class_del = torch.ones(int(self.batch_size), dtype=bool) # shape: (batch_size,), class_del is the mask for the negative samples
            class_del[np.where(y_sam.cpu() == m)] = 0 # shape: (batch_size,), class_del is the mask for the negative samples
            class_neg = torch.arange(self.batch_size).masked_select(class_del) # shape: (neg_size,), class_neg is the index of the negative samples
            neg_sam_id = random.sample(range(0, class_neg.shape[0]), int(neg_size)) # shape: (neg_size,), neg_sam_id is the index of the negative samples
            class_sam.append(class_neg[neg_sam_id]) # shape: (neg_size,), class_sam is the index of the negative samples
        out = h # shape: (batch_size, feature_dim), out is the feature of the instance
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature) # shape: (batch_size, batch_size), neg is the exponential of the similarity between the negative samples
        neg_samp = torch.zeros(neg.shape[0], int(neg_size)) # shape: (batch_size, neg_size), neg_samp is the negative samples
        for n in range(np.max(y_pred) + 1):
            neg_samp[np.where(y_sam.cpu() == n)] = neg.cpu().index_select(1, class_sam[n])[np.where(y_sam.cpu() == n)] # shape: (batch_size, neg_size), neg_samp is the negative samples
        neg_samp = neg_samp.to(self.device) # shape: (batch_size, neg_size), neg_samp is the negative samples
        Ng = neg_samp.sum(dim=-1) # shape: (batch_size,), Ng is the sum of the negative samples
        out = h # shape: (batch_size, feature_dim), out is the feature of the instance
        pos = torch.exp(torch.mm(out, out.t().contiguous())) # shape: (batch_size, batch_size), pos is the exponential of the similarity between the positive samples
        pos = torch.diag(torch.exp(torch.mm(out, h_i.t().contiguous()))) # shape: (batch_size,), pos is the exponential of the similarity between the positive samples and the feature of the instance i
        return self.lambda_1 * (- torch.log(pos / (Ng))).mean() # shape: (1,), loss is the loss for the debiased instance

def valid(model, device, dataset, view, data_size, class_num):
    model.eval()
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        _, h_fusion, _, q_fusion = model.forward_fusion(x_list)
    pred_q = np.argmax(np.array(q_fusion.cpu().detach().numpy()), axis=1)
    nmi_q, ari_q, acc_q, pur_q = evaluate(y, pred_q)
    pred_h = KMeans(n_clusters=class_num, n_init=10) .fit_predict(h_fusion.cpu().detach().numpy())
    nmi_h, ari_h, acc_h, pur_h = evaluate(y, pred_h)
    return acc_q, nmi_q, pur_q, ari_q, acc_h, nmi_h, ari_h,  pur_h

def get_top_k_nearest_label(feature, label, k):
    index = faiss.IndexFlatL2(feature.shape[1])  # use L2 distance to search the nearest neighbor by brute force
    index.add(feature) # add the feature to the index, shape: (batch_size, feature_dim)
    distances, indices = index.search(feature, k) # search the k-nearest neighbor, shape: (batch_size, k)
    top_k_nearest_label = label[indices] # get the label of the k-nearest neighbor, shape: (batch_size, k)
    top_k_nearest_label = torch.from_numpy(top_k_nearest_label) # convert the label to tensor, shape: (batch_size, k)
    return top_k_nearest_label 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAGA')
    parser.add_argument('--dataset', default='Fashion') # 'Fashion', 'LabelMe'
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--learning_rate", default=0.0001)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument('--neighbor_num', default=5, type=int)
    parser.add_argument('--feature_dim', default = 512, type=int)
    parser.add_argument('--gcn_dim', default = 128, type=int)
    parser.add_argument('--tau', default= 0.1 , type=float)
    parser.add_argument('--lambda1', default = 1.0 , type=float)
    parser.add_argument('--lambda2', default = 1.0 , type=float)
    parser.add_argument('--eta', default = 1.0, type=float)
    parser.add_argument('--neg_size', default = 128, type=int)
    parser.add_argument('--mse_epochs', default = 200, type=int)
    parser.add_argument('--con_epochs', default = 200, type=int)
    parser.add_argument('--seed', default = 10, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## 1. Set seed for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model, optimizer, criterion.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = Network(view, dims, args.feature_dim, args.gcn_dim, class_num, args.neighbor_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
    criterion  = Loss(args.batch_size, class_num, args.temperature_f, args.lambda1, args.lambda2, args.eta, device)
    
    ## 3. Train the model.
    for epoch in range(args.mse_epochs):
        tot_loss_list = []
        for batch_idx, (xs, labels, _) in enumerate(data_loader):
            xns = [xs[v] + torch.randn_like(xs[v]) * 0.1 for v in range(view)]; xns = [xn.to(device) for xn in xns]; xs = [x.to(device) for x in xs]
            optimizer.zero_grad()
            zs, hs, rs = model(xns)
            loss = sum([F.mse_loss(xs[v], rs[v]) for v in range(view)])
            loss.backward()
            optimizer.step()
            tot_loss_list.append(loss.item())
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(tot_loss_list)))
    y_pred = KMeans(n_clusters=class_num, n_init=10) .fit_predict(torch.cat(hs, dim=1).data.cpu().numpy())
    nmi, ari, acc, pur = evaluate(labels.flatten().data.cpu().numpy(), y_pred.flatten())
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    for epoch in range(args.con_epochs):
        total_loss_list = []
        for batch_idx, (xs, labels, _) in enumerate(data_loader):
            xns=[xs[v] + torch.randn_like(xs[v]) * 0.1 for v in range(view)]; xns = [xn.to(device) for xn in xns]; xs = [x.to(device) for x in xs]
            optimizer.zero_grad()
            zs, hs, rs = model(xns)
            hs, h_fusion, qs, q_fusion = model.forward_fusion(xs)
            y_pred_km = KMeans(n_clusters=class_num, n_init=10) .fit_predict(h_fusion.data.cpu().numpy())
            loss_list = []
            for v in range(view):
                loss_list.append(F.mse_loss(xs[v], rs[v])) # reconstruction loss
                loss_list.append(criterion.forword_feature(h_fusion.T, hs[v].T)) # feature contrastive loss
                loss_list.append(criterion.forword_debiased_instance(h_fusion, hs[v], y_pred_km, neg_size=args.neg_size)) # debiased instance loss
                top_1_nearest_label = get_top_k_nearest_label(h_fusion.detach().cpu().numpy(), qs[v].detach().cpu().numpy(), k=1)
                top_1_nearest_label = top_1_nearest_label.squeeze(dim=1) # shape: (batch_size,), maybe have some bugs here. top 1 nearest label is its own label!!!
                q_nearest_sample = top_1_nearest_label.to(device) # convert the label to tensor, shape: (batch_size,)
                loss_list.append(criterion.forward_pui_label(q_fusion, q_nearest_sample)) # partition uncertainty index loss
            loss = sum(loss_list) # total loss
            loss.backward()
            optimizer.step()
            total_loss_list.append(loss.item())
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(total_loss_list)))  
        if epoch % 10 == 0:
            acc, nmi, pur, ari, acc_k, nmi_k, ari_k,  pur_k= valid(model, device, dataset, view, data_size, class_num)
            print('Clustering results on semantic labels: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
            print('Clustering results on kmeans clustering: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc_k, nmi_k, ari_k, pur_k))
            