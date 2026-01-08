import os, sys, math, random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim))

    def forward(self, x):
        return self.decoder(x)
    
class MFLVC(nn.Module): # MFLVC from MFLVC
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(MFLVC, self).__init__()
        # self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim) for v in range(view)])
        # self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim) for v in range(view)])
        # self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        # self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        # self.view = view
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        self.head = nn.Sequential(nn.Linear(feature_dim * view, class_num), nn.Softmax(dim=1)) # not used in the implementation
        self.view = view

    def forward(self, x_list):
        h_list = []; z_list = []; q_list = []; r_list = []
        for v in range(self.view):
            z = self.encoders[v](x_list[v])
            h = F.normalize(self.feature_contrastive_module(z), dim=1) # L2 normalization
            q = self.label_contrastive_module(z)
            r = self.decoders[v](z)
            z_list.append(z); h_list.append(h); q_list.append(q); r_list.append(r)
        return z_list, h_list, q_list, r_list
    
class MVCLLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(MVCLLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2) # input two vectors, output the cosine similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # input logits and labels, output the loss

    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask

    def forward_feature(self, h_i, h_j):
        # Compute the feature contrastive loss: 1/2*(L_fc^(i,j) + L_fc^(j,i))
        # h_i and h_j are the feature vectors of the positive samples, shape: (batch_size, feature_dim) unit vectors
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0) # shape: (N, feature_dim), h is the concatenation of h_i and h_j
        sim = torch.matmul(h, h.T) / self.temperature_f # shape: (N, N), sim is the cosine similarity matrix
        sim_i_j = torch.diag(sim, diagonal=self.batch_size) # shape: (N//2), sim_i_j is the cosine similarity between h_i and h_j
        sim_j_i = torch.diag(sim, diagonal=-self.batch_size) # shape: (N//2), sim_j_i is the cosine similarity between h_j and h_i
        sim_of_positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive samples
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative samples
        sim_of_negative_samples = sim[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative samples
        labels = torch.zeros(N).to(self.device).long() # shape: (N,), labels is the labels for the positive samples
        logits = torch.cat((sim_of_positive_samples, sim_of_negative_samples), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative samples
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for logits and labels
        loss /= N # shape: (1,), loss is the average loss for logits and labels, N is the number of samples
        return loss

    def forward_label(self, q_i, q_j):
        # Compute the label contrastive loss: 1/2*(L_lc^(i,j) + L_lc^(j,i))
        # q_i and q_j are the cluster assignments of the positive samples, shape: (batch_size, class_num)
        p_i = q_i.sum(0).view(-1) # shape: (class_num,), p_i is the probability of the cluster assignments of the view i
        p_i /= p_i.sum() # shape: (class_num,), p_i is the normalized probability of the cluster assignments of the view i
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum() # shape: (1,), ne_i is the negative information entropy of the cluster assignments of the view i
        p_j = q_j.sum(0).view(-1) # shape: (class_num,), p_j is the probability of the cluster assignments of the view j
        p_j /= p_j.sum() # shape: (class_num,), p_j is the normalized probability of the cluster assignments of the view j
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum() # shape: (1,), ne_j is the negative information entropy of the cluster assignments of the view j
        entropy = ne_i + ne_j # shape: (1,), entropy is the negative information entropy of the cluster assignments of the view i and view j

        N = 2 * self.class_num
        q = torch.cat((q_i.t(), q_j.t()), dim=0) # shape: (N, batch_size), q is the concatenation of q_i and q_j
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l # shape: (2*class_num, 2*class_num), sim is the cosine similarity matrix
        sim_i_j = torch.diag(sim, diagonal=self.class_num) # shape: (class_num,), sim_i_j is the cosine similarity between q_i and q_j
        sim_j_i = torch.diag(sim, diagonal=-self.class_num) # shape: (class_num,), sim_j_i is the cosine similarity between q_j and q_i
        sim_of_positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive samples
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative samples
        sim_of_negative_samples = sim[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative samples
        labels = torch.zeros(N).to(self.device).long() # shape: (N,), labels is the labels for the positive samples
        logits = torch.cat((sim_of_positive_samples, sim_of_negative_samples), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative samples
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for the positive and negative samples
        loss /= N # shape: (1,), loss is the average loss for the positive and negative samples
        return loss + entropy

def validation(model, dataset, view, data_size, class_num, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, _ = next(iter(test_loader))
    x_list = [x_list[v].to(device) for v in range(view)]
    y = y.numpy()
    with torch.no_grad():
        zs, hs, qs, rs = model.forward(x_list) # qs.shape: (batch_size, class_num) * view, sum(qs).shape: (batch_size, class_num)
        final_pred = torch.argmax(sum(qs)/view, dim=1) # shape: (batch_size,)
    final_pred = final_pred.detach().cpu().numpy() # shape: (batch_size,)
    nmi, ari, acc, pur = evaluate(y, final_pred)
    print("Clustering on latent q (k-means):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur
    
def match_pseudo_label_to_pred(y_true, y_pred):
    # match the pseudo-labels to the predicted labels using the Hungarian algorithm
    # For example: y_true = [0, 0, 0, 1, 1, 1], y_pred = [1, 1, 2, 0, 0, 0], the matched y_true = [1, 1, 1, 0, 0, 0]
    # For example: y_true = [0, 0, 0, 0, 1, 1], y_pred = [1, 1, 1, 1, 1, 1], the matched y_true = [1, 1, 1, 1, x, x] this is not allowed. (num_pred_clusters < num_true_clusters)
    # For example: y_true = [0, 0, 0, 1, 1, 1], y_pred = [1, 1, 2, 0, 0, 1], the matched y_true = [1, 1, 1, 0, 0, 0] this is allowed. (num_pred_clusters > num_true_clusters)
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(y_true.max()+1)] for i in range(y_pred.max()+1)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    row_ind, col_ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, row_ind is the row indices (predicted clusters), col_ind is the column indices (true clusters)
    # Create mapping: true_label -> predicted_label using dictionary
    mapping = {int(col_ind[j]): int(row_ind[j]) for j in range(len(row_ind))}
    matched_y_true = np.array([mapping[y_true[i]] for i in range(y_true.shape[0])])
    matched_y_true = torch.from_numpy(matched_y_true).long() # shape: (num_samples,)
    return matched_y_true

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim; self.num_heads = num_heads; self.dropout_rate = dropout_rate
        self.head_dim = self.hidden_dim // self.num_heads
        self.linear_q = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.linear_k = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.linear_v = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.linear_bias = nn.Linear(6, self.num_heads) # not used in the implementation
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Linear(self.num_heads * self.head_dim, 1)

    def forward(self, q, k, v): 
        batch_size = q.size(0); 
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, self.head_dim) # shape: (batch_size, target_seq_len, num_heads, head_dim)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, self.head_dim) # shape: (batch_size, source_seq_len, num_heads, head_dim)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, self.head_dim) # shape: (batch_size, source_seq_len, num_heads, head_dim)
        q = q.transpose(1, 2)                  # [batch_size, num_heads, target_seq_len, head_dim]
        k = k.transpose(1, 2).transpose(2, 3)  # [batch_size, num_heads, head_dim, source_seq_len]
        v = v.transpose(1, 2)                  # [batch_size, num_heads, source_seq_len, head_dim]
        attention_scores = torch.matmul(q, k) / math.sqrt(self.head_dim) # [batch_size, num_heads, target_seq_len, source_seq_len]
        attention_weights = F.softmax(attention_scores, dim=3) # [batch_size, num_heads, target_seq_len, source_seq_len]
        attention_weights = self.dropout(attention_weights) # [batch_size, num_heads, target_seq_len, source_seq_len]
        output = attention_weights.matmul(v)  # [batch_size, num_heads, target_seq_len, head_dim]
        output = output.transpose(1, 2).contiguous()  # [batch_size, target_seq_len, num_heads, head_dim]
        output = output.view(batch_size, self.num_heads * self.head_dim) # shape: (batch_size, num_heads * head_dim)
        output = self.output_layer(output) # shape: (batch_size, 1)
        return output # shape: (batch_size, 1)

class FeedForwardNetwork(nn.Module):
    def __init__(self, view, hidden_dim):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = nn.Linear(view, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, view)
        self.gelu = nn.GELU()

    def forward(self, x): # x.shape: (batch_size, view)
        x = self.gelu(self.linear_1(x)) # shape: (batch_size, hidden_dim)
        x = self.linear_2(x).unsqueeze(1) # shape: (batch_size, 1, view)
        return x

def benchmark_2023_MM_DealMVC(dataset_name="BBCSport",
                              batch_size=256,
                              temperature_f=0.5,
                              temperature_l=1.0,
                              learning_rate=0.0003,
                              weight_decay=0.0,
                              mse_epochs=300,
                              con_epochs=100,
                              tune_epochs=50,
                              feature_dim=512,
                              high_feature_dim=512,
                              seed=15,
                              threshold=0.8,
                              num_heads=8,
                              hidden_dim=256,
                              dropout_rate=0.5,
                              ffn_hidden_dim=32,
                              verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MFLVC(view, dims, feature_dim, high_feature_dim, class_num).to(device)
    mha_net = MultiHeadAttention(hidden_dim, num_heads, dropout_rate).to(device)
    ffn_net = FeedForwardNetwork(view, ffn_hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_mha_net = torch.optim.Adam(mha_net.parameters(), lr=learning_rate, weight_decay=weight_decay) # not used in the implementation
    optimizer_ffn_net = torch.optim.Adam(ffn_net.parameters(), lr=learning_rate, weight_decay=weight_decay) # not used in the implementation
    criterion = MVCLLoss(batch_size, class_num, temperature_f, temperature_l, device)
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss() # input logits and labels, output the loss, but the predicted probability has been normalized by softmax, so there is two softmax layers in the model. NLL loss is better than CrossEntropyLoss.

    ## 3. Train and evaluate the model.
    ## 3.1. Reconstruction Pre-training
    model.train()
    for epoch in range(mse_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            _, _, _, rs = model(x_list)
            loss = sum([criterion_mse(x_list[v], rs[v]) for v in range(view)])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("Epoch: {} Total Loss: {:.4f}".format(epoch + 1, np.mean(losses))) if verbose else None
    ## 3.2. Multi-view Contrastive Learning. (DealMVC has a error implementation here, maybe hahaha)
    for epoch in range(con_epochs):
        # proposed local and global contrative calibration loss
        tot_loss = 0.
        # init p target distribution and adaptive weight
        adaptive_weight = torch.FloatTensor(np.ones(view)/view).to(device).unsqueeze(1) # shape: (view, 1)
        p_sample = torch.FloatTensor(np.ones(view)/view).to(device) # shape: (view,)
        for batch_idx, (xs, _, _) in enumerate(dataloader):
            xs = [x.to(device) for x in xs] # shape: (batch_size, input_dim) * view
            optimizer.zero_grad()
            zs, hs, qs, rs = model(xs)
            loss_list = []
            # local contrastive calibration
            sim_qs_probs_list = [] # shape: (batch_size, batch_size) * view * (view-1)/2
            for v in range(view):
                for w in range(v+1, view):
                    # similarity of features of the samples in any two views
                    sim_hs = torch.exp(torch.mm(hs[v], hs[w].t())) # shape: (batch_size, batch_size), exponential of the cosine similarity matrix (has been normalized by L2 norm)
                    sim_hs_probs = F.normalize(sim_hs, p=1, dim=1) # shape: (batch_size, batch_size), L1 normalization across the last dimension
                    # similarity of labels of the samples in any two views
                    sim_qs = torch.mm(qs[v], qs[w].t()).fill_diagonal_(1) # shape: (batch_size, batch_size), not cosine similarity matrix just matrix multiplication, set the diagonal elements to 1
                    sim_qs_probs = F.normalize(sim_qs * (sim_qs >= threshold), p=1, dim=1) # shape: (batch_size, batch_size), L1 normalization across the last dimension, mask the negative samples, keep the positive samples
                    sim_qs_probs_list.append(sim_qs_probs) # shape: (batch_size, batch_size) * view * (view-1)/2
                    loss_contrast_local = - (torch.log(sim_hs_probs + 1e-7) * sim_qs_probs).sum(1) # shape: (batch_size,), negative log likelihood loss
                    loss_list.append(loss_contrast_local.mean()) # shape: (1,)
                loss_list.append(criterion_mse(xs[v], rs[v])) # shape: (1,) Reconstruction loss
                
            # global contrastive calibration
            hs_concat = torch.cat([torch.mean(hs[v], 1).unsqueeze(1) for v in range(view)], dim=1).t() # shape: (view, batch_size)
            weight_mha_net = mha_net(hs_concat, hs_concat, hs_concat) # shape: (view, 1)
            weight_ffn_net = ffn_net(p_sample) # shape: (view, 1), target distribution
            weight_adaptive = F.softmax(weight_mha_net * weight_ffn_net, dim=0) # shape: (view, 1), adaptive weights for each view
            adaptive_weight = weight_adaptive * adaptive_weight # shape: (view, 1), adaptive weights for each view
            fusion_feature = torch.stack([adaptive_weight[v].item() * hs[v] for v in range(view)], dim=0).sum(dim=0) # shape: (view, batch_size, feature_dim) -> (batch_size, feature_dim)
            fusion_label = model.label_contrastive_module(fusion_feature) # shape: (batch_size, class_num)
            sim_fusion_fea = torch.exp(torch.mm(fusion_feature, fusion_feature.t())) # shape: (batch_size, batch_size)
            sim_fusion_fea_probs = sim_fusion_fea / sim_fusion_fea.sum(1, keepdim=True) # shape: (batch_size, batch_size), L1 normalization across the last dimension   
            # sim_fusion_fea_probs = F.normalize(sim_fusion_fea, p=1, dim=1) # shape: (batch_size, batch_size), L1 normalization across the last dimension
            sim_fusion_label = torch.mm(fusion_label, fusion_label.t()).fill_diagonal_(1) # shape: (batch_size, batch_size), not cosine similarity matrix just matrix multiplication
            sim_fusion_label_probs = F.normalize(sim_fusion_label * (sim_fusion_label >= threshold), p=1, dim=1) # shape: (batch_size, batch_size), L1 normalization across the last dimension
            loss_contrast_global = - (torch.log(sim_fusion_fea_probs + 1e-7) * sim_fusion_label_probs).sum(1) # shape: (batch_size,), negative log likelihood loss
            loss_list.append(loss_contrast_global.mean()) # shape: (1,)

            # local and global contrastive calibration loss
            for v in range(view - 1 if view == 2 else view):
                loss_list.append(F.mse_loss(sim_fusion_label_probs, sim_qs_probs_list[v])) # shape: (1,), mean squared error loss
            
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print("Contrastive Learning Epoch: {} Total Loss: {:.4f}".format(epoch + 1, tot_loss / len(dataloader))) if verbose else None
    ## 3.3. Multi-view Pseudo-label Refinement
    ## 3.3.1. Get the pseudo-labels from the high-level features
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]
    model.eval()
    with torch.no_grad():
        _, hs, _, _ = model.forward(x_list) 
    hs = [MinMaxScaler().fit_transform(hs[v].detach().cpu().numpy()) for v in range(view)] # Normalize the features by MinMaxScaler
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    pseudo_label_list = [kmeans.fit_predict(hs[v]).reshape(data_size) for v in range(view)]
    ## 3.3.2. Refine by matching the pseudo-labels to the predicted clusters
    # model.train()
    for epoch in range(tune_epochs):
        optimizer.zero_grad()
        _, _, qs, _ = model(x_list) # qs is the cluster assignments of each modality, shape: (batch_size, class_num) * view
        loss_list = []
        for v in range(view):
            q = torch.argmax(qs[v], dim=1).detach().cpu().numpy()
            p_matched = match_pseudo_label_to_pred(pseudo_label_list[v], q).to(device)
            loss_list.append(criterion_ce(qs[v], p_matched))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        print("Pseudo-label Refinement Epoch: {} Total Loss: {:.4f}".format(epoch + 1, loss.item())) if verbose else None
    nmi, ari, acc, pur = validation(model, dataset, view, data_size, class_num, verbose=verbose)
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DealMVC")
    parser.add_argument("--dataset", type=str, default="BBCSport")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--temperature_f", type=float, default=0.5)
    parser.add_argument("--temperature_l", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--mse_epochs", type=int, default=300)
    parser.add_argument("--con_epochs", type=int, default=100)
    parser.add_argument("--tune_epochs", type=int, default=50)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--high_feature_dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=15)
    # DealMVC Specific Hyperparameters
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--ffn_hidden_dim", type=int, default=32)
    args = parser.parse_args()
    
    nmi, ari, acc, pur = benchmark_2023_MM_DealMVC(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        temperature_f=args.temperature_f,
        temperature_l=args.temperature_l,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mse_epochs=args.mse_epochs,
        con_epochs=args.con_epochs,
        tune_epochs=args.tune_epochs,
        feature_dim=args.feature_dim,
        high_feature_dim=args.high_feature_dim,
        seed=args.seed,
        threshold=args.threshold,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate,
        ffn_hidden_dim=args.ffn_hidden_dim,
        verbose=False
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))