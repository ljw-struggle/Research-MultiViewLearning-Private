import os, sys, random, argparse, numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
try:
    from ..dataset import load_data
    from ..metric import evaluate
except ImportError: # Add parent directory to path when running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from benchmark.dataset import load_data
    from benchmark.metric import evaluate

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, hidden_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters; self.hidden_dim = hidden_dim; self.alpha = alpha
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_centers.data)

    def forward(self, x):
        q = 1.0 / (1.0 + torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2) / self.alpha) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        q = q ** ((self.alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / torch.sum(q, dim=1, keepdim=True)  # Normalize q to sum to 1 across clusters
        return q

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, dim=0) # shape: [batch_size, n_clusters]
        p = weight / torch.sum(weight, dim=1, keepdim=True)  # Normalize p to sum to 1 across clusters
        return p

class DEMVC(nn.Module): # Deep Embedded Multi-view Clustering (DEMVC)
    def __init__(self, view=2, input_dims=[1750, 79], feature_dim=10, class_num=10, alpha=1.0):
        super(DEMVC, self).__init__()
        self.view = view
        self.autoencoders = nn.ModuleList([AutoEncoder(input_dims[v], feature_dim) for v in range(view)])
        self.clustering_layers = nn.ModuleList(ClusteringLayer(class_num, feature_dim, alpha) for v in range(view))
    
    def forward(self, x_list):
        q_list = []; encoded_list = []; decoded_list = []
        for v in range(self.view):
            encoded, decoded = self.autoencoders[v](x_list[v])
            q = self.clustering_layers[v](encoded)
            q_list.append(q); encoded_list.append(encoded); decoded_list.append(decoded)
        return q_list, encoded_list, decoded_list

    @staticmethod
    def cluster_assignment_distribution(embedding, cluster_centers, alpha=1.0): # embedding: shape: [batch_size, feature_dim], cluster_centers: shape: [n_clusters, feature_dim]
        # embedding.unsqueeze(1).shape: [batch_size, 1, feature_dim]
        # cluster_centers.shape: [n_clusters, feature_dim]
        # embedding.unsqueeze(1) - cluster_centers: shape: [batch_size, n_clusters, feature_dim]
        q = 1.0 / (1.0 + torch.sum((embedding.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha) # shape: [batch_size, n_clusters]
        q = q ** ((alpha + 1.0) / 2.0) # shape: [batch_size, n_clusters]
        q = q / torch.sum(q, dim=1, keepdim=True) # shape: [batch_size, n_clusters]
        return q # shape: [batch_size, n_clusters]

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / torch.sum(q, dim=0) # shape: [batch_size, n_clusters]
        p = weight / torch.sum(weight, dim=1, keepdim=True)  # Normalize p to sum to 1 across clusters
        return p # shape: [batch_size, n_clusters]

def validation(model, data, label, view, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        q_list, encoded_list, decoded_list = model(data)
    q_views_list = [q.detach().cpu().numpy() for q in q_list] # shape: view * [data_size, class_num]
    q_global = sum(q_views_list) / view # shape: (data_size, class_num)
    y_pred_global = np.argmax(q_global, axis=1) # shape: (data_size,)
    nmi, ari, acc, pur = evaluate(label.cpu().detach().numpy(), y_pred_global)
    print("Clustering on latent q (k-means):") if verbose else None
    print("ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(acc, nmi, ari, pur)) if verbose else None
    return nmi, ari, acc, pur

def benchmark_2021_INSC_DEMVC_NC(dataset_name='BDGP', batch_size=256, pretrain_learning_rate=0.001, pretrain_epochs=500, dec_learning_rate=0.001, maxiter=30000, update_interval=1000, lambda_mse=1.0, lambda_clustering=0.1, verbose=False, random_state=42):
    # DEMVC_NC: every view perform deep embedded clustering independently.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    data = [data[v].to(device) for v in range(view)] # shape: [view, data_size, dims[v]]
    label = label.to(device) # shape: [data_size]

    ## 3. Initialize model and optimizers.
    model = DEMVC(view=view, input_dims=dims, feature_dim=10, class_num=class_num).to(device)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=pretrain_learning_rate)
    optimizer_train = torch.optim.Adam(model.parameters(), lr=dec_learning_rate)
    criterion_mse = nn.MSELoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean', log_target=False) # batchmean: average the loss over the batch dimension.
    
    ## 4. Pretrain the AutoEncoder.
    model.train()
    for epoch in range(pretrain_epochs):
        perm = torch.randperm(data_size)  # Shuffle the data
        total_loss_list = []
        for i in range(0, data_size, batch_size):
            optimizer_pretrain.zero_grad()
            data_batch = [data[v][perm[i:i+batch_size]] for v in range(view)]
            _, _, decoded_list = model(data_batch) # shape: [batch_size, sum(dims)] if view == -1, shape: [batch_size, dims[view]]
            loss = sum([criterion_mse(data_batch[v], decoded_list[v]) for v in range(view)])
            loss.backward()
            optimizer_pretrain.step()
            total_loss_list.append(loss.item())
        print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(total_loss_list):.4f}') if verbose else None
        
    ## 5. Initialize cluster centers with KMeans.
    model.eval()
    q_list, encoded_list, decoded_list = model(data)
    features_list = [encoded.detach().cpu().numpy() for encoded in encoded_list]
    for v in range(view):
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        kmeans.fit_predict(features_list[v])
        model.clustering_layers[v].cluster_centers.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device))
    
    ## 6. Train the DEMVC model.
    index_array = np.arange(data_size)
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            model.eval()
            with torch.no_grad():
                q_list, encoded_list, decoded_list = model(data)
            q_views_list = [q.detach().cpu().numpy() for q in q_list] # shape: view * [data_size, class_num]
            q_global = sum(q_views_list) / view # shape: (data_size, class_num)
            y_pred_views_list = [np.argmax(q_view, axis=1) for q_view in q_views_list] # shape: view * [data_size,]
            y_pred_global = np.argmax(q_global, axis=1) # shape: (data_size,)
            alignment_ratio = np.sum(np.all(np.stack(y_pred_views_list) == y_pred_global, axis=0)) / data_size # shape: scalar
            P = [model.target_distribution(q_list[v]).detach() for v in range(view)] # shape: view * [data_size, class_num]
            nmi, ari, acc, pur = evaluate(label.cpu().detach().numpy(), y_pred_global)
            print("Iteration: {} ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(ite, acc, nmi, ari, pur)) if verbose else None
        model.train()
        optimizer_train.zero_grad()
        idx = index_array[(ite * batch_size) % data_size:min((ite * batch_size) % data_size + batch_size, data_size)] # ensure the index is within the range of the data
        data_batch = [data[v][idx] for v in range(view)]
        q_batch, encoded_batch, decoded_batch = model(data_batch) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        loss = lambda_clustering * sum([criterion_kl(torch.log(q_batch[v]), P[v][idx]) for v in range(view)]) + lambda_mse * sum([criterion_mse(decoded_batch[v], data_batch[v]) for v in range(view)])
        loss.backward()
        optimizer_train.step()
    
    ## 7. Evaluate the model.
    nmi, ari, acc, pur = validation(model, data, label, view, verbose=verbose)
    return nmi, ari, acc, pur
    
def benchmark_2021_INSC_DEMVC(dataset_name='BDGP', batch_size=256, view_first=0, pretrain_learning_rate=0.001, pretrain_epochs=500, dec_learning_rate=0.001, maxiter=30000, update_interval=1000, lambda_mse=1.0, lambda_clustering=0.1, verbose=False, random_state=42):
    # DEMVC: every view perform deep embedded clustering iteratively.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## 1. Set seed for reproducibility.
    random.seed(random_state); 
    np.random.seed(random_state); 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    data, label, idx = next(iter(dataloader))
    data = [data[v].to(device) for v in range(view)] # shape: [view, data_size, dims[v]]
    label = label.to(device) # shape: [data_size]

    ## 3. Initialize model and optimizers.
    model = DEMVC(view=view, input_dims=dims, feature_dim=10, class_num=class_num).to(device)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=pretrain_learning_rate)
    optimizer_train = torch.optim.Adam(model.parameters(), lr=dec_learning_rate)
    criterion_mse = nn.MSELoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean', log_target=False) # batchmean: average the loss over the batch dimension.
    
    ## 4. Pretrain the AutoEncoder.
    model.train()
    for epoch in range(pretrain_epochs):
        perm = torch.randperm(data_size)  # Shuffle the data
        total_loss_list = []
        for i in range(0, data_size, batch_size):
            optimizer_pretrain.zero_grad()
            data_batch = [data[v][perm[i:i+batch_size]] for v in range(view)]
            _, _, decoded_list = model(data_batch) # shape: [batch_size, sum(dims)] if view == -1, shape: [batch_size, dims[view]]
            loss = sum([criterion_mse(data_batch[v], decoded_list[v]) for v in range(view)])
            loss.backward()
            optimizer_pretrain.step()
            total_loss_list.append(loss.item())
        print(f'[Pretrain] Epoch {epoch + 1}, Loss: {np.mean(total_loss_list):.4f}') if verbose else None
        
    ## 5. Initialize cluster centers with KMeans.
    model.eval()
    q_list, encoded_list, decoded_list = model(data) # shape: view * [data_size, class_num], view * [data_size, feature_dim], view * [data_size, dims[view]]
    features_list = [encoded.detach().cpu().numpy() for encoded in encoded_list]
    centers_list = []
    for v in range(view):
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        kmeans.fit_predict(features_list[v])
        centers_list.append(kmeans.cluster_centers_)
    for v in range(view):
        model.clustering_layers[v].cluster_centers.data.copy_(torch.tensor(centers_list[v], dtype=torch.float32).to(device))
        # model.clustering_layers[v].cluster_centers.data.copy_(torch.tensor(centers_list[view_first], dtype=torch.float32).to(device))
    
    ## 6. Train the DEMVC model.
    index_array = np.arange(data_size)
    view_first = 0
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            model.eval()
            with torch.no_grad():
                q_list, encoded_list, decoded_list = model(data) # shape: view * [data_size, class_num], view * [data_size, feature_dim], view * [data_size, dims[view]]
            q_views_list = [q.detach().cpu().numpy() for q in q_list] # shape: view * [data_size, class_num]
            q_global = sum(q_views_list) / view # shape: (data_size, class_num)
            y_pred_views_list = [np.argmax(q_view, axis=1) for q_view in q_views_list] # shape: view * [data_size,]
            y_pred_global = np.argmax(q_global, axis=1) # shape: (data_size,)
            alignment_ratio = np.sum(np.all(np.stack(y_pred_views_list) == y_pred_global, axis=0)) / data_size # shape: scalar
            P = [model.target_distribution(q_list[view_first]).detach() for v in range(view)] # shape: view * [data_size, class_num]
            nmi, ari, acc, pur = evaluate(label.cpu().detach().numpy(), y_pred_global)
            print("Iteration: {} ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}".format(ite, acc, nmi, ari, pur)) if verbose else None
            view_first = (view_first + 1) % view
        model.train()
        optimizer_train.zero_grad()
        idx = index_array[(ite * batch_size) % data_size:min((ite * batch_size) % data_size + batch_size, data_size)] # ensure the index is within the range of the data
        data_batch = [data[v][idx] for v in range(view)]
        q_batch, encoded_batch, decoded_batch = model(data_batch) # similarity matrix q using t-distribution, shape: [batch_size, n_clusters]
        loss = lambda_clustering * sum([criterion_kl(torch.log(q_batch[v]), P[v][idx]) for v in range(view)]) + lambda_mse * sum([criterion_mse(decoded_batch[v], data_batch[v]) for v in range(view)])
        loss.backward()
        optimizer_train.step()
    
    ## 7. Evaluate the model.
    nmi, ari, acc, pur = validation(model, data, label, view, verbose=verbose)
    return nmi, ari, acc, pur

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEMVC")
    parser.add_argument("--dataset", default="BDGP", type=str)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--pretrain_learning_rate", default=0.001, type=float)
    parser.add_argument("--pretrain_epochs", default=500, type=int)
    parser.add_argument("--dec_learning_rate", default=0.001, type=float)
    parser.add_argument("--maxiter", default=30000, type=int)
    parser.add_argument("--update_interval", default=1000, type=int)
    parser.add_argument("--lambda_mse", default=1.0, type=float)
    parser.add_argument("--lambda_clustering", default=0.1, type=float)
    parser.add_argument("--mode", default="DEMVC", type=str, choices=["DEMVC", "DEMVC_NC"])
    parser.add_argument("--view_first", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    
    if args.mode == "DEMVC_NC":
        nmi, ari, acc, pur = benchmark_2021_INSC_DEMVC_NC(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            pretrain_learning_rate=args.pretrain_learning_rate,
            pretrain_epochs=args.pretrain_epochs,
            dec_learning_rate=args.dec_learning_rate,
            maxiter=args.maxiter,
            update_interval=args.update_interval,
            lambda_mse=args.lambda_mse,
            lambda_clustering=args.lambda_clustering,
            verbose=False,
            random_state=args.seed
        )
    else:
        nmi, ari, acc, pur = benchmark_2021_INSC_DEMVC(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            view_first=args.view_first,
            pretrain_learning_rate=args.pretrain_learning_rate,
            pretrain_epochs=args.pretrain_epochs,
            dec_learning_rate=args.dec_learning_rate,
            maxiter=args.maxiter,
            update_interval=args.update_interval,
            lambda_mse=args.lambda_mse,
            lambda_clustering=args.lambda_clustering,
            verbose=False,
            random_state=args.seed
        )
    print("NMI = {:.4f}; ARI = {:.4f}; ACC = {:.4f}; PUR = {:.4f}".format(nmi, ari, acc, pur))
    