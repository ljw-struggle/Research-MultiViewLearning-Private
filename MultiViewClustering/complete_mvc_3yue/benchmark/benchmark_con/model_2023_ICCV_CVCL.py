import os, sys, random, argparse, numpy as np, scipy.io as sio
import torch
import torch.nn as nn
from itertools import combinations
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim_list):
        super(Encoder, self).__init__()
        dims = [input_dim] + hidden_dim_list + [feature_dim]
        self.encoder = nn.Sequential()
        for i in range(len(dims) - 1):
            self.encoder.add_module('Linear_%d' % i, nn.Linear(dims[i], dims[i+1]))
            self.encoder.add_module('ReLU_%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim_list):
        super(Decoder, self).__init__()
        dims = [feature_dim] + list(reversed(hidden_dim_list)) + [input_dim]
        self.decoder = nn.Sequential()
        for i in range(len(dims) - 1):
            self.decoder.add_module('Linear_%d' % i, nn.Linear(dims[i], dims[i+1]))
            self.decoder.add_module('ReLU_%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)

class CVCL(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
        super(CVCL, self).__init__()
        self.encoders = []
        self.decoders = []
        for idx in range(num_views):
            self.encoders.append(Encoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(Decoder(input_sizes[idx], dim_high_feature, dims))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.label_learning_module = nn.Sequential(nn.Linear(dim_high_feature, dim_low_feature), nn.Linear(dim_low_feature, num_clusters), nn.Softmax(dim=1))
        self.view = num_views
        
    def forward(self, x_list):
        q_list = []; r_list = []; z_list = []
        for idx in range(self.view):
            z = self.encoders[idx](x_list[idx])
            r = self.decoders[idx](z)
            q = self.label_learning_module(z)
            q_list.append(q); r_list.append(r); z_list.append(z)
        return q_list, r_list, z_list

class CVCLLoss(nn.Module):
    def __init__(self, num_samples, num_clusters, temperature_l, normalized, device):
        super(CVCLLoss, self).__init__()
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.temperature_l = temperature_l
        self.normalized = normalized
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
    
    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask

    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i); q_j = self.target_distribution(q_j)
        p_i = q_i.sum(0).view(-1) # shape: (class_num,), p_i is the probability of the cluster assignments of the view i
        p_i /= p_i.sum() # shape: (class_num,), p_i is the normalized probability of the cluster assignments of the view i
        ne_i = (p_i * torch.log(p_i)).sum() # shape: (1,), ne_i is the negative information entropy of the cluster assignments of the view i
        p_j = q_j.sum(0).view(-1) # shape: (class_num,), p_j is the probability of the cluster assignments of the view j
        p_j /= p_j.sum() # shape: (class_num,), p_j is the normalized probability of the cluster assignments of the view j
        ne_j = (p_j * torch.log(p_j)).sum() # shape: (1,), ne_j is the negative information entropy of the cluster assignments of the view j
        entropy = ne_i + ne_j # shape: (1,), entropy is the negative information entropy of the cluster assignments of the view i and view j
        return entropy

    def forward_label(self, q_i, q_j):
        # Compute the label contrastive loss: 1/2*(L_lc^(i,j) + L_lc^(j,i))
        # q_i and q_j are the cluster assignments of the positive samples, shape: (batch_size, class_num)
        q_i = self.target_distribution(q_i); q_j = self.target_distribution(q_j)
        N = 2 * self.num_clusters
        q = torch.cat((q_i.t(), q_j.t()), dim=0) # shape: (N, batch_size), q is the concatenation of q_i and q_j
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l if self.normalized else torch.matmul(q, q.T) / self.temperature_l
        sim_i_j = torch.diag(sim, self.num_clusters) # shape: (class_num,), sim_i_j is the cosine similarity between q_i and q_j
        sim_j_i = torch.diag(sim, -self.num_clusters) # shape: (class_num,), sim_j_i is the cosine similarity between q_j and q_i
        sim_of_positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive clusters
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative clusters
        sim_of_negative_clusters = sim[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative clusters
        labels = torch.zeros(N).to(self.device).long() # shape: (N,), labels is the labels for the positive clusters
        logits = torch.cat((sim_of_positive_clusters, sim_of_negative_clusters), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative clusters
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for the positive and negative clusters
        loss /= N # shape: (1,), loss is the average loss for the positive and negative clusters
        return loss

    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, dim=0) # shape: (batch_size, num_clusters)
        # p = weight / torch.sum(weight, dim=1, keepdim=True) # Normalize p to sum to 1 across clusters, shape: [batch_size, num_clusters]
        p = (weight.t() / torch.sum(weight, 1)).t() # shape: (batch_size, num_clusters) # reproduction of the code in the CVCL paper
        return p # shape: (batch_size, num_clusters)

def validation(model, dataset, view, data_size, class_num, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False)
    x_list, y, _ = next(iter(test_loader))
    x_list = [x_list[v].to(device) for v in range(view)]
    y = y.numpy()
    with torch.no_grad():
        qs, _, _ = model.forward(x_list)
        final_pred = torch.argmax(sum(qs)/view, dim=1) # shape: (batch_size,)
    final_pred = final_pred.detach().cpu().numpy() # shape: (batch_size,)
    nmi, ari, acc, pur = evaluate(y, final_pred)
    print("Clustering on latent q (k-means):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur
    
def benchmark_2023_ICCV_CVCL(dataset_name="MSRCv1",
                             learning_rate=0.0005,
                             weight_decay=0.0,
                             batch_size=35,
                             seed=10,
                             mse_epochs=200,
                             con_epochs=400,
                             normalized=False,
                             temperature_l=1.0,
                             dim_high_feature=2000,
                             dim_low_feature=1024,
                             hidden_dims=[256, 512],
                             alpha=0.01,
                             beta=0.005,
                             verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, feature_dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = CVCL(view, feature_dims, hidden_dims, dim_high_feature, dim_low_feature, class_num).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion_cvcl = CVCLLoss(batch_size, class_num, temperature_l, normalized, device)
    criterion_mse = torch.nn.MSELoss()
    
    ## 3. Train the model.
    ## 3.1. Reconstruction Pre-training (MSE Loss)
    model.train()
    for epoch in range(mse_epochs):
        losses = []
        for batch_idx, (x_list, _, _) in enumerate(dataloader):
            x_list = [x_list[v].to(device) for v in range(view)]
            optimizer.zero_grad()
            _, rs, _ = model(x_list)
            loss = sum([criterion_mse(x_list[v], rs[v]) for v in range(view)])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("Pre-training Epoch: {} Total Loss: {:.4f}".format(epoch + 1, np.mean(losses))) if verbose else None
    
    ## 3.2. Multi-view Contrastive Learning
    for epoch in range(con_epochs):
        model.train()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        losses = []
        for batch_idx, (x_list, _, _) in enumerate(dataloader):
            x_list = [x_list[v].to(device) for v in range(view)]
            qs, rs, _ = model(x_list)
            loss_list = []
            for v, w in combinations(range(view), 2):
                loss_list.append(alpha * criterion_cvcl.forward_label(qs[v], qs[w]))
                loss_list.append(beta * criterion_cvcl.forward_prob(qs[v], qs[w]))
            for v in range(view):
                loss_list.append(criterion_mse(x_list[v], rs[v]))
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("Contrastive Learning Epoch: {} Total Loss: {:.4f}".format(epoch + 1, np.mean(losses))) if verbose else None
    
    ## 4. Evaluate the model.
    nmi, ari, acc, pur = validation(model, dataset, view, data_size, class_num, verbose=verbose)
    return nmi, ari, acc, pur

def get_config(args):
    if args.dataset == "MSRCv1": # db checked 92.86
        dataset_name = "MSRCv1"; learning_rate = 0.0005; weight_decay = 0.; batch_size = 35; seed = 10; mse_epochs = 200; con_epochs = 400; normalized = False; temperature_l = 1.0;
        dim_high_feature = 2000; dim_low_feature = 1024; dims = [256, 512]; alpha = 0.01; beta = 0.005;
    elif args.dataset == "MNIST-USPS": # db checked 99.7
        dataset_name = "MNIST-USPS"; learning_rate = 0.0001; weight_decay = 0.; batch_size = 50; seed = 10; mse_epochs = 200; con_epochs = 200; normalized = False; temperature_l = 1.0;
        dim_high_feature = 1500; dim_low_feature = 1024; dims = [256, 512, 1024]; alpha = 0.05; beta = 0.05;
    elif args.dataset == "COIL20": # db checked 84.65
        dataset_name = "COIL20"; learning_rate = 0.0005; weight_decay = 0.; batch_size = 180; seed = 50; mse_epochs = 200; con_epochs = 400; normalized = False; temperature_l = 1.0;
        dim_high_feature = 768; dim_low_feature = 200; dims = [256, 512, 1024, 2048]; alpha = 0.01; beta = 0.01;
    elif args.dataset == "scene": # db checked 44.59
        dataset_name = "scene"; learning_rate = 0.0005; weight_decay = 0.; batch_size = 69; seed = 10; mse_epochs = 200; con_epochs = 100; normalized = False; temperature_l = 1.0;
        dim_high_feature = 1500; dim_low_feature = 256; dims = [256, 512, 1024, 2048]; alpha = 0.01; beta = 0.05;
    elif args.dataset == "hand": # db checked 96.85
        dataset_name = "hand"; learning_rate = 0.0001; weight_decay = 0.; batch_size = 200; seed = 50; mse_epochs = 200; con_epochs = 200; normalized = True; temperature_l = 1.0;
        dim_high_feature = 1024; dim_low_feature = 1024; dims = [256, 512, 1024]; alpha = 0.005; beta = 0.001;
    elif args.dataset == "Fashion": # db checked 99.31
        dataset_name = "Fashion"; learning_rate = 0.0005; weight_decay = 0.; batch_size = 100; seed = 20; mse_epochs = 200; con_epochs = 100; normalized = True; temperature_l = 0.5;
        dim_high_feature = 2000; dim_low_feature = 500; dims = [256, 512]; alpha = 0.005; beta = 0.005;
    elif args.dataset == "BDGP": # db checked 99.2
        dataset_name = "BDGP"; learning_rate = 0.0001; weight_decay = 0.; batch_size = 250; seed = 10; mse_epochs = 200; con_epochs = 100; normalized = True; temperature_l = 1.0;
        dim_high_feature = 2000; dim_low_feature = 1024; dims = [256, 512]; alpha = 0.01; beta = 0.01;
    return dataset_name, learning_rate, weight_decay, batch_size, seed, mse_epochs, con_epochs, normalized, temperature_l, dim_high_feature, dim_low_feature, dims, alpha, beta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVCLNet")
    parser.add_argument("--dataset", default="MSRCv1", type=str, choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP'])
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--temperature_l", default=1.0, type=float)
    parser.add_argument("--normalized", default=False, type=bool)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=100, type=int)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()
    
    dataset_name, learning_rate, weight_decay, batch_size, seed, mse_epochs, con_epochs, normalized, temperature_l, dim_high_feature, dim_low_feature, hidden_dims, alpha, beta = get_config(args)
    nmi, ari, acc, pur = benchmark_2023_ICCV_CVCL(
        dataset_name=dataset_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=seed,
        mse_epochs=mse_epochs,
        con_epochs=con_epochs,
        normalized=normalized,
        temperature_l=temperature_l,
        dim_high_feature=dim_high_feature,
        dim_low_feature=dim_low_feature,
        hidden_dims=hidden_dims,
        alpha=alpha,
        beta=beta,
        verbose=False
    )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))