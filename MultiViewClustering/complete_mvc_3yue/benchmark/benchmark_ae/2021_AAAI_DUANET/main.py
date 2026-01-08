import argparse, umap, itertools, random
import numpy as np, scipy.io as sio, matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import scale # standardize the data, equivalent to (x - mean(x)) / std(x)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics.cluster import contingency_matrix, rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X; self.Y = Y

    def __getitem__(self, index):
        x = [x[index] for x in self.X]; y = self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.Y)

class ReconstructionNet(nn.Module):
    def __init__(self, h_dim, feature_dim):
        super(ReconstructionNet, self).__init__()
        self.linears = nn.Sequential(nn.Linear(h_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, feature_dim))

    def forward(self, h):
        return self.linears(h)

class UncertaintyNet(nn.Module):
    def __init__(self, h_dim, feature_dim):
        super(UncertaintyNet, self).__init__()
        self.linears = nn.Sequential(nn.Linear(h_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, h):
        return self.linears(h)

def cluster_acc(labels_true, labels_pred):
    labels_true = np.array(labels_true); labels_pred = np.array(labels_pred)
    D = max(labels_pred.max(), labels_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(labels_pred)):
        w[labels_pred[i], labels_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)  # align clusters using the Hungarian algorithm
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / len(labels_pred)

# def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(D)] for i in range(D)], dtype=np.int64)
#     # w = np.zeros((D, D), dtype=np.int64)
#     # for i in range(y_pred.size):
#     #     w[y_pred[i], y_true[i]] += 1
#     ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm
#     return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] 

def b3_precision_recall_fscore(labels_true, labels_pred): # Calculate B^3 variant of precision, recall and F-score
    # Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation metrics based on formal constraints." Information retrieval 12.4 (2009): 461-486.
    true_clusters = {k: set(np.where(labels_true == k)[0]) for k in np.unique(labels_true)} # Build mapping: cluster_id -> set(sample indices)
    pred_clusters = {k: set(np.where(labels_pred == k)[0]) for k in np.unique(labels_pred)} # Build mapping: cluster_id -> set(sample indices)
    true_clusters = {k: frozenset(v) for k, v in true_clusters.items()}; pred_clusters = {k: frozenset(v) for k, v in pred_clusters.items()} # Freeze sets for hashability
    n_samples = len(labels_true); intersection_cache = {}; precision_total = 0.0; recall_total = 0.0
    for i in range(n_samples):
        true_cluster = true_clusters[labels_true[i]]; pred_cluster = pred_clusters[labels_pred[i]]; key = (pred_cluster, true_cluster)
        intersection_cache[key] = pred_cluster & true_cluster if key not in intersection_cache else intersection_cache[key] # Pre-compute intersections to avoid redundancy
        precision_total += len(intersection_cache[key]) / len(pred_cluster); recall_total += len(intersection_cache[key]) / len(true_cluster)
    precision = precision_total / n_samples; recall = recall_total / n_samples; f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def clustering(H, Y, n_clusters, count=10):
    acc_array = np.zeros(count); nmi_array = np.zeros(count); ri_array = np.zeros(count); f1_array = np.zeros(count)
    for i in range(count): # Randomly cluster the features multiple times and evaluate the clustering performance
        Y_pred = KMeans(n_clusters=n_clusters).fit_predict(H)
        acc_array[i] = cluster_acc(Y, Y_pred) * 100; nmi_array[i] = normalized_mutual_info_score(Y, Y_pred) * 100
        ri_array[i] = rand_score(Y, Y_pred) * 100; 
        f1_array[i] = b3_precision_recall_fscore(Y, Y_pred) * 100
    return acc_array.mean(), acc_array.std(), nmi_array.mean(), nmi_array.std(), ri_array.mean(), ri_array.std(), f1_array.mean(), f1_array.std()

def classification(H, Y, k=3, count=1, test_size=0.2):
    acc_array = np.zeros(count); precision_array = np.zeros(count); recall_array = np.zeros(count); f1_array = np.zeros(count)
    for i in range(count): # Randomly split the data and train the classifier multiple times
        train_data, test_data, train_label, test_label = train_test_split(H, Y, test_size=test_size, stratify=Y)
        classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1).fit(train_data, train_label)
        test_pred = classifier.predict(test_data)
        acc_array[i] = accuracy_score(test_label, test_pred)*100; precision_array[i] = precision_score(test_label, test_pred, average='macro')*100
        recall_array[i] = recall_score(test_label, test_pred, average='macro')*100; f1_array[i] = f1_score(test_label, test_pred, average='macro')*100
    return acc_array.mean(), acc_array.std(), precision_array.mean(), precision_array.std(), recall_array.mean(), recall_array.std(), f1_array.mean(), f1_array.std()

def tsne(H, Y):
    embedding = TSNE(n_components=2, random_state=0).fit_transform(H)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=Y.squeeze(), s=50, cmap=plt.cm.get_cmap('Paired', len(np.unique(Y))))
    plt.colorbar(ticks=range(len(np.unique(Y)) + 1)); plt.axis('off'); plt.xticks([]); plt.yticks([]); plt.show()

def umapp(H, Y):
    embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(H)
    plt.figure(figsize=(10, 10), dpi=80)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=Y.squeeze(), s=50, cmap=plt.cm.get_cmap('Paired', len(np.unique(Y))))
    plt.colorbar(ticks=range(len(np.unique(Y)) + 1)); plt.axis('off'); plt.xticks([]); plt.yticks([]); plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=float, default=20, help='size of the latent space [default: 20]')
    parser.add_argument('--batch_size', type=int, default=2000, help='input batch size for training [default: 2000]')
    parser.add_argument('--epoch_num', type=int, default=[200, 100], help='number of epochs to train [default: 500]')
    parser.add_argument('--learning_rate', type=float, default=[5e-3, 1e-3], help='learning rate [default: 1e-3]')
    parser.add_argument('--log_interval', type=int, default=10, help='how many batches to wait before logging training status [default: 10]')
    parser.add_argument('--times', type=int, default=30, help='number of times to run the experiment [default: 30]')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # random.seed(42); np.random.seed(42); torch.manual_seed(42) # Set random seed for reproducibility
    classification_acc = np.zeros([args.times]); classification_precision = np.zeros([args.times]); classification_recall = np.zeros([args.times]); classification_f1 = np.zeros([args.times])
    clustering_acc = np.zeros([args.times]); clustering_nmi = np.zeros([args.times]); clustering_ri = np.zeros([args.times]); clustering_f1 = np.zeros([args.times])
    for t in range(args.times):
        data = sio.loadmat('./data/cub.mat')
        data_X = data['X'][0] # shape: (views, features, samples)
        data_views = len(data_X) # number of views
        data_samples = data_X[0].shape[1] # number of samples
        data_features = [data_X[v].shape[0] for v in range(data_views)] # number of features in each view
        data_X = [data_X[v].T for v in range(data_views)] # shape: (samples, features), transpose to have samples as rows
        data_X = [scale(x) for x in data_X] # standardize each view
        data_X = [torch.from_numpy(x).to(device).float() for x in data_X] # convert to tensor
        data_Y = data['gt'].squeeze() - 1 # shape: (samples, )
        data_categories = np.unique(data_Y).shape[0] # number of categories
        dataset = MyDataset(data_X, data_Y) # create dataset
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        H = torch.normal(mean=torch.zeros([data_samples, args.latent_dim]), std=0.01).to(device).detach() # initialize latent space, detach to avoid gradients
        H.requires_grad_(True) # set requires_grad to True for optimization
        RNet = [ReconstructionNet(args.latent_dim, data_features[v]).to(device) for v in range(data_views)] # create fusion nets for each view
        UNet = [UncertaintyNet(args.latent_dim, data_features[v]).to(device) for v in range(data_views)] # create uncertainty nets for each view
        
        # 1\ Pretraining: optimize the latent space H to reconstruct each view
        optimizer_pre = torch.optim.Adam(itertools.chain(nn.ModuleList(RNet).parameters(), [H]), lr=args.learning_rate[0])
        for epoch_pre in range(args.epoch_num[0]):
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                optimizer_pre.zero_grad()
                h = H[idx]  # get the latent space for the current batch
                x_re = [RNet[v](h) for v in range(data_views)]  # reconstruct each view
                loss_pre = sum([F.mse_loss(x_re[v], x[v], reduction='mean') for v in range(data_views)]) # sum the losses from all views
                loss_pre.backward()
                optimizer_pre.step()
                print('Pretraining Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch_pre, (batch_idx+1) * len(y), len(dataloader.dataset), 100. * (batch_idx + 1) / len(dataloader), loss_pre)) if batch_idx % args.log_interval == 0 else None
        
        # 2\ Fine-tuning: optimize the latent space H to reconstruct each view and predict uncertainty
        optimizer = torch.optim.Adam(itertools.chain(nn.ModuleList(RNet).parameters(), nn.ModuleList(UNet).parameters(), [H]), lr=args.learning_rate[1])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        for epoch in range(args.epoch_num[1]):
            for batch_idx, (x, y, idx) in enumerate(dataloader):
                optimizer.zero_grad()
                h = H[idx]  # get the latent space for the current batch
                x_re = [RNet[v](h) for v in range(data_views)]  # reconstruct each view
                log_sigma_2 = [UNet[v](h) for v in range(data_views)]
                loss = sum([0.5 * torch.mean((x_re[v] - x[v])**2 * torch.exp(-log_sigma_2[v]) + log_sigma_2[v]) for v in range(data_views)]) # sum the losses from all views
                loss.backward()
                optimizer.step()
                scheduler.step()
                print('Training Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, (batch_idx + 1) * len(y), len(dataloader.dataset), 100. * (batch_idx + 1) / len(dataloader), loss)) if batch_idx % args.log_interval == 0 else None
        
        # 3\ Evaluation: evaluate the latent space H using clustering and classification
        H = H.detach().cpu().numpy() # detach H to numpy for evaluation
        classification_acc[t], _, classification_precision[t], _, classification_recall[t], _, classification_f1[t], _ = classification(H, data_Y, count=1, test_size=0.2) # Evaluate using classification
        clustering_acc[t], _, clustering_nmi[t], _, clustering_ri[t], _, clustering_f1[t], _ = clustering(H, data_Y, data_categories, count=10) # Evaluate using clustering
    print(f'Classification ACC = {classification_acc.mean():.2f}±{classification_acc.std():.2f}\n' + f'Classification p = {classification_precision.mean():.2f}±{classification_precision.std():.2f}\n' + f'Classification r = {classification_recall.mean():.2f}±{classification_recall.std():.2f}\n' + f'Classification f1 = {classification_f1.mean():.2f}±{classification_f1.std():.2f}\n')
    print(f'Clustering ACC = {clustering_acc.mean():.2f}±{clustering_acc.std():.2f}\n' + f'Clustering NMI = {clustering_nmi.mean():.2f}±{clustering_nmi.std():.2f}\n' + f'Clustering RI = {clustering_ri.mean():.2f}±{clustering_ri.std():.2f}\n' + f'Clustering f1 = {clustering_f1.mean():.2f}±{clustering_f1.std():.2f}\n')
    