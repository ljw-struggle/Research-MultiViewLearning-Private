import os, math, argparse, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from itertools import combinations
from _utils import evaluate, load_data

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, input_dim))

    def forward(self, x):
        return self.decoder(x)

class DEC(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha = 1.0):
        super(DEC, self).__init__()
        self.n_clusters = n_clusters; self.hidden_dim = hidden_dim; self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize q to sum to 1 across clusters
        return q # shape: [batch_size, n_clusters]
    
    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, dim=0) # shape: (batch_size, num_clusters)
        # p = weight / torch.sum(weight, dim=1, keepdim=True) # Normalize p to sum to 1 across clusters, shape: [batch_size, num_clusters]
        p = (weight.t() / torch.sum(weight, 1)).t() # shape: (batch_size, num_clusters) # reproduction of the code in the CVCL paper
        return p # shape: (batch_size, num_clusters)

class MCMC(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(MCMC, self).__init__()
        self.encoders = []; self.decoders = []; self.decs = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
            self.decs.append(DEC(class_num, high_feature_dim, 1.0))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.decs = nn.ModuleList(self.decs)
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.label_contrastive_module = nn.Sequential(nn.Linear(high_feature_dim, class_num), nn.Softmax(dim=1))
        self.view = view

    def forward(self, x_list):
        h_list = []; z_list = []; q_list = []; r_list = []
        for v in range(self.view):
            z = self.encoders[v](x_list[v])
            h = F.normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(h)
            r = self.decoders[v](z)
            h_list.append(h); q_list.append(q); r_list.append(r); z_list.append(z)
        return h_list, q_list, r_list, z_list
    
    def forward_cluster(self, x_list):
        h_list = []; z_list = []; q_list = []; r_list = []
        for v in range(self.view):
            z = self.encoders[v](x_list[v])
            h = self.feature_contrastive_module(z)
            q = self.label_contrastive_module(h)
            r = self.decoders[v](z)
            h_list.append(h); q_list.append(q); r_list.append(r); z_list.append(z)
        return h_list, q_list, r_list, z_list

class MCMCLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l,  device):
        super(MCMCLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2) # input two vectors, output the cosine similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # input logits and labels, output the loss
        self.predictor = nn.Sequential(
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )

    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask

    def get_samples_mask(self, N, sim, K=1): # MCMC has a error implementation here. (maybe hahaha)
        positive_mask = torch.zeros(N//2, N//2).fill_diagonal_(1)
        positive_mask_upper = torch.concat([torch.zeros(N//2, N//2), positive_mask], dim=1)
        positive_mask_lower = torch.concat([positive_mask, torch.zeros(N//2, N//2)], dim=1)
        positive_mask = torch.concat([positive_mask_upper, positive_mask_lower], dim=0)
        positive_mask = positive_mask.bool()
        negative_mask = torch.ones(N//2, N//2).fill_diagonal_(0)
        negative_mask = torch.concat([negative_mask, negative_mask], dim=1)
        negative_mask = torch.concat([negative_mask, negative_mask], dim=0)
        negative_mask = negative_mask.bool()
        for i in range(N//2):
            view_i_sim = sim[i, N//2:]
            view_j_sim = sim[N//2 + i, :N//2]
            concate_sim = torch.cat((view_i_sim, view_j_sim), dim=0)
            _, top_indices = concate_sim.topk(2 + 2*K)
            count = 0
            for idx in top_indices:
                if idx >= N//2 and idx != N//2 + i:
                    negative_mask[self.batch_size + i, idx - N//2] = False
                    positive_mask[self.batch_size + i, idx - N//2] = True
                    count += 1
                if idx < N//2 and idx != i:
                    negative_mask[i, self.batch_size + idx] = False
                    positive_mask[i, self.batch_size + idx] = True
                    count += 1
                if count == 2*K:
                    break
        return positive_mask, negative_mask

    def forward_neighbor_feature(self, h_i, h_j): # MCMC has a error implementation here. (maybe hahaha)
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        positive_mask, negative_mask = self.get_samples_mask(N, sim, K=1)
        sim_of_positive_samples = sim[positive_mask].reshape(N, -1) # shape: (N, 2)
        sim_of_negative_samples = sim[negative_mask].reshape(N, -1)
        labels = torch.zeros(N).to(sim_of_positive_samples.device).long()
        logits = torch.cat((sim_of_positive_samples, sim_of_negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
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

        q_i = q_i.t() # shape: (class_num, batch_size), q_i is the transpose of q_i
        q_j = q_j.t() # shape: (class_num, batch_size), q_j is the transpose of q_j
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0) # shape: (2*class_num, batch_size), q is the concatenation of q_i and q_j
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l # shape: (2*class_num, 2*class_num), sim is the cosine similarity matrix
        sim_i_j = torch.diag(sim, self.class_num) # shape: (class_num,), sim_i_j is the cosine similarity between q_i and q_j
        sim_j_i = torch.diag(sim, -self.class_num) # shape: (class_num,), sim_j_i is the cosine similarity between q_j and q_i
        sim_of_positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive clusters
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative clusters
        sim_of_negative_clusters = sim[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative clusters
        labels = torch.zeros(N).to(sim_of_positive_clusters.device).long() # shape: (N,), labels is the labels for the positive clusters
        logits = torch.cat((sim_of_positive_clusters, sim_of_negative_clusters), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative clusters
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for the positive and negative clusters
        loss /= N # shape: (1,), loss is the average loss for the positive and negative clusters
        return loss + entropy

    def compute_cluster_loss(self, q_centers, k_centers): # MCMC has a error implementation here. (maybe hahaha)
        # q_centers.shape: (class_num, feature_dim), k_centers.shape: (class_num, feature_dim)
        d_q = q_centers.mm(q_centers.T) / 1.0 # shape: (class_num, class_num)
        d_k = (q_centers * k_centers).sum(dim=1) / 1.0
        d_q[torch.arange(self.class_num), torch.arange(self.class_num)] = d_k
        mask = ~torch.diag(torch.ones(self.class_num)).bool() # shape: (class_num, class_num) with diagonal elements set to False
        neg = d_q[mask].reshape(-1, self.class_num - 1)
        loss = - d_k + torch.logsumexp(torch.cat([d_k.reshape(self.class_num, 1), neg], dim=1), dim=1) # InfoNCE loss
        loss = loss.sum() / (self.class_num) # shape: (1,), loss is the average loss for the positive and negative clusters
        return loss
    
    # def compute_cluster_loss(self, q_centers, k_centers):
    #     self.num_cluster = self.class_num
    #     d_q = q_centers.mm(q_centers.T) / 1.0   #1.0
    #     d_k = (q_centers * k_centers).sum(dim=1) / 1.0
    #     d_q = d_q.float()
    #     d_q[torch.arange(self.num_cluster), torch.arange(self.num_cluster)] = d_k
    #     mask = torch.zeros((self.num_cluster, self.num_cluster), dtype=torch.bool, device=d_q.device)
    #     d_q.masked_fill_(mask, -10)
    #     pos = d_q.diag(0)
    #     mask = torch.ones((self.num_cluster, self.num_cluster))
    #     mask = mask.fill_diagonal_(0).bool()
    #     neg = d_q[mask].reshape(-1, self.num_cluster - 1)
    #     loss = - pos + torch.logsumexp(torch.cat([pos.reshape(self.num_cluster, 1), neg], dim=1), dim=1) 
    #     loss = loss.sum() / (self.num_cluster)
    #     return 1.0 * loss

    def compute_centers(self, h, pseudo_label): 
        # h.shape: (batch_size, feature_dim), pseudo_label.shape: (batch_size, class_num)
        if len(pseudo_label.size()) > 1:
            weight = pseudo_label.T # shape: (class_num, batch_size)
        else:
            weight = torch.zeros(self.class_num, h.size(0)).to(self.device)  # shape: (class_num, batch_size)
            weight[pseudo_label, torch.arange(h.size(0))] = 1 # shape: (class_num, batch_size)
        weight = F.normalize(weight, p=2, dim=1)  # L2 normalization, shape: (class_num, batch_size)
        centers = torch.mm(weight, h) # shape: (class_num, feature_dim), centers is the centers of the clusters
        centers = F.normalize(centers, p=2, dim=1) # shape: (class_num, feature_dim), centers is the normalized centers of the clusters
        return centers # shape: (class_num, feature_dim)
    
def validation(model, dataset, view, data_size, class_num, eval_multi_view=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        hs, qs, rs, zs = model.forward(x_list) 
        _, qs, _, _ = model.forward_cluster(x_list)
        multi_pred = [torch.argmax(qs[v], dim=1) for v in range(view)] # shape: (batch_size,) * view
        final_pred = torch.argmax(sum(qs)/view, dim=1) # shape: (batch_size,)
    low_level_vector = [z.detach().cpu().numpy() for z in zs] # shape: (batch_size, feature_dim) * view
    high_level_vector = [h.detach().cpu().numpy() for h in hs] # shape: (batch_size, feature_dim) * view
    multi_pred = [p.detach().cpu().numpy() for p in multi_pred] # shape: (batch_size,) * view
    final_pred = final_pred.detach().cpu().numpy() # shape: (batch_size,)
    if eval_multi_view:
        print("Clustering results on low-level features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(low_level_vector[v])
            nmi, ari, acc, pur = evaluate(y, y_pred)
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v + 1, acc, nmi, ari, pur))
        print("Clustering results on high-level features of each view:")
        for v in range(view):
            kmeans = KMeans(n_clusters=class_num, n_init=100)
            y_pred = kmeans.fit_predict(high_level_vector[v])
            nmi, ari, acc, pur = evaluate(y, y_pred)
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v + 1, acc, nmi, ari, pur))
        print("Clustering results on cluster assignments of each view:")
        for v in range(view):
            nmi, ari, acc, pur = evaluate(y, multi_pred[v])
            print('For view {}, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(v+1, acc, nmi, ari, pur))
    print("Clustering results on overall cluster assignments: " + str(class_num))
    nmi, ari, acc, pur = evaluate(y, final_pred)
    print('Overall, ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MCMC')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument("--temperature_f", default=0.5, type=float)
    parser.add_argument("--temperature_l", default=1.0, type=float)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=50, type=int)
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--high_feature_dim", default=128, type=int)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs('./result', exist_ok=True)
    dataset_name = args.dataset; batch_size = args.batch_size; temperature_f = args.temperature_f; temperature_l = args.temperature_l; learning_rate = args.learning_rate; weight_decay = args.weight_decay; 
    mse_epochs = args.mse_epochs; con_epochs = args.con_epochs; feature_dim = args.feature_dim; high_feature_dim = args.high_feature_dim; seed = args.seed
    alpha = 1; beta = 0; gamma = 0.0001; zeta = 0.0001
    batch_size_dict = {"BDGP": 64, "CCV": 256, "Fashion": 100}
    mse_epochs_dict = {"BDGP": 20, "CCV": 200, "Fashion": 50}
    con_epochs_dict = {"BDGP": 100, "CCV": 100, "Fashion": 100}
    seed_dict = {"BDGP": 10, "CCV": 3, "Fashion": 10}
    batch_size = batch_size_dict[args.dataset]; mse_epochs = mse_epochs_dict[args.dataset]; con_epochs = con_epochs_dict[args.dataset]; seed = seed_dict[args.dataset]
    
    ## 1. Set seed for reproducibility.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = MCMC(view, dims, feature_dim, high_feature_dim, class_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MCMCLoss(batch_size, class_num, temperature_f, temperature_l, device)
    criterion_mse = torch.nn.MSELoss()
    criterion_kld = torch.nn.KLDivLoss(reduction="batchmean", log_target=False)
    
    ## 3. Train and evaluate the model.
    ## 3.1. Reconstruction Pre-training
    model.train()
    for epoch in range(mse_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            _, _, rs, _ = model(x_list)
            loss = sum([criterion_mse(x_list[v], rs[v]) for v in range(view)])
            # print(loss.item())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Reconstruction Pre-training Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(losses)))
    validation(model, dataset, view, data_size, class_num, eval_multi_view=False)
    ## 3.2. Multi-view Contrastive Learning
    model.train()
    for epoch in range(con_epochs):
        losses = []
        for x_list, y, idx in dataloader:
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            hs, qs, rs, zs = model(x_list)
            loss_list = []; cluster_center_list = []; cluster_result_list = []
            # for v in range(view): # MCMC has a error implementation here. (maybe hahaha)
            #     cluster_center = criterion.compute_centers(hs[v], qs[v].argmax(1))
            #     cluster_center_list.append(cluster_center)
            #     if v == 0:
            #         cluster_result = model.decs[v](hs[v])
            #         cluster_result_list.append(cluster_result)
            #         model.decs[v].cluster_centers.data.copy_(cluster_center.detach())
            #     else:
            #         model.decs[v].cluster_centers.data.copy_(cluster_center.detach())
            #         cluster_result = model.decs[v](hs[v])
            #         cluster_result_list.append(cluster_result)
            # for v, w in combinations(range(view), 2):
            #     loss_list.append(alpha * criterion.forward_neighbor_feature(hs[v], hs[w]))   
            #     loss_list.append(beta * criterion.forward_label(qs[v], qs[w]))     
            #     loss_list.append(gamma * criterion.compute_cluster_loss(cluster_center_list[v], cluster_center_list[w])) 
            # for v in range(view): # MCMC has a error implementation here. (maybe hahaha)
            #     # target_p = criterion.target_distribution(qs[v]).detach()
            #     # loss_list.append(zeta * criterion_kld(qs[v].log(), target_p))
            #     # loss_list.append(zeta * criterion_kld(model.decs[v](hs[v]).log(), target_p))
            #     loss_list.append(zeta * criterion_kld(qs[v].log(), cluster_result_list[v]))
            #     loss_list.append(criterion_mse(x_list[v], rs[v]))
            for v in range(view): # MCMC has a error implementation here. (maybe hahaha)
                loss_list.append(criterion_mse(x_list[v], rs[v]))
                cluster_center_v = criterion.compute_centers(hs[v], qs[v].argmax(1))            
                cluster_results_v = model.decs[v](hs[v])
                model.decs[v].cluster_centers.data.copy_(cluster_center_v.detach())
                for w in range(v+1, view):
                    loss_list.append(alpha*criterion.forward_neighbor_feature(hs[v], hs[w]))   
                    loss_list.append(beta*criterion.forward_label(qs[v], qs[w]))     
                    cluster_center_w = criterion.compute_centers(hs[w], qs[w].argmax(1))                    
                    model.decs[w].cluster_centers.data.copy_(cluster_center_w.detach())
                    loss_list.append(gamma*criterion.compute_cluster_loss(cluster_center_v, cluster_center_w)) 
                loss_list.append(zeta * criterion_kld(qs[v].log(), cluster_results_v))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print('Contrastive Learning Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(np.mean(losses)))
    validation(model, dataset, view, data_size, class_num, eval_multi_view=False)
    torch.save(model.state_dict(), './result/' + dataset_name + '.pth')
    