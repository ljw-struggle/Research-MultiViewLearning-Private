import torch
import argparse
import numpy as np
from torchvision import datasets, transforms
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import Subset
import torch.nn as nn
from collections import OrderedDict
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat

class batch_MeanShift(object):
    def __init__(self, args):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs
    
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_compute_distance)(X, self.clusters[i]) for i in range(self.clusters.shape[0]))
        #for i in range(self.n_clusters)
        dis_mat = np.hstack(dis_mat)
        return dis_mat
    
    def init_cluster(self, X, indices=None):
        """ Generate initial clusters using sklearn.MeanShift """
        model = MeanShift()
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters
    
    def update_cluster(self, X, cluster_idx):
        """ Update clusters in MeanShift on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] + eta * X[i])
            self.clusters[cluster_idx] = updated_cluster
    
    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        return np.argmin(dis_mat, axis=1)

class batch_KMeans(object):
    def __init__(self, args):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs
    
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(delayed(_parallel_compute_distance)(X, self.clusters[i]) for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)
        return dis_mat
    
    def init_cluster(self, X, indices=None):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters, n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters
    
    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] + eta * X[i])
            self.clusters[cluster_idx] = updated_cluster
    
    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        return np.argmin(dis_mat, axis=1)

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.input_dim = args.input_dim
        self.output_dim = self.input_dim
        self.hidden_dims = args.hidden_dims
        self.hidden_dims.append(args.latent_dim)
        self.dims_list = (args.hidden_dims + args.hidden_dims[:-1][::-1])  # mirrored structure
        self.n_layers = len(self.dims_list)
        self.n_clusters = args.n_clusters
        # Validation check
        assert self.n_layers % 2 > 0
        assert self.dims_list[self.n_layers // 2] == args.latent_dim
        # Encoder Network
        layers = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_dims):
            if idx == 0:
                layers.update({'linear0': nn.Linear(self.input_dim, hidden_dim), 'activation0': nn.ReLU()})
            else:
                layers.update({'linear{}'.format(idx): nn.Linear(self.hidden_dims[idx-1], hidden_dim), 'activation{}'.format(idx): nn.ReLU(), 'bn{}'.format(idx): nn.BatchNorm1d(self.hidden_dims[idx])})
        self.encoder = nn.Sequential(layers)
        # Decoder Network
        layers = OrderedDict()
        tmp_hidden_dims = self.hidden_dims[::-1]
        for idx, hidden_dim in enumerate(tmp_hidden_dims):
            if idx == len(tmp_hidden_dims) - 1:
                layers.update({'linear{}'.format(idx): nn.Linear(hidden_dim, self.output_dim)})
            else:
                layers.update({'linear{}'.format(idx): nn.Linear(hidden_dim, tmp_hidden_dims[idx+1]),'activation{}'.format(idx): nn.ReLU(),'bn{}'.format(idx): nn.BatchNorm1d(tmp_hidden_dims[idx+1])})
        self.decoder = nn.Sequential(layers)
    
    def __repr__(self):
        repr_str = '[Structure]: {}-'.format(self.input_dim)
        for idx, dim in enumerate(self.dims_list):
                repr_str += '{}-'.format(dim)
        repr_str += str(self.output_dim) + '\n'
        repr_str += '[n_layers]: {}'.format(self.n_layers) + '\n'
        repr_str += '[n_clusters]: {}'.format(self.n_clusters) + '\n'
        repr_str += '[input_dims]: {}'.format(self.input_dim)
        return repr_str
    
    def __str__(self):
        return self.__repr__()
    
    def forward(self, X, latent=False):
        output = self.encoder(X)
        if latent:
            return output
        return self.decoder(output)

class DCN(nn.Module):
    def __init__(self, args):
        super(DCN, self).__init__()
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term 
        self.lamda = args.lamda  # coefficient of the reconstruction term
        self.device = torch.device(args.device)
        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))
        if not self.lamda > 0:
            msg = 'lamda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))
        if len(self.args.hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')
        if args.clustering == 'kmeans':
            self.clustering = batch_KMeans(args)
        elif args.clustering == 'meanshift':
            self.clustering = batch_MeanShift(args)
        else:
            raise RuntimeError('Error: no clustering chosen')
        self.autoencoder = AutoEncoder(args).to(self.device)
        self.criterion  = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.wd)
    
    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, cluster_id):
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)
        # Reconstruction error
        rec_loss = self.lamda * self.criterion(X, rec_X)
        # Regularization term on clustering
        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.clustering.clusters).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)
        return (rec_loss + dist_loss, rec_loss.detach().cpu().numpy(), dist_loss.detach().cpu().numpy())
    
    def pretrain(self, train_loader, epoch=100, verbose=True):
        if not self.args.pretrain:
            return
        print('========== Start pretraining ==========')
        rec_loss_list  =[]
        self.train()
        for e in range(epoch):
            for batch_idx, (data, _) in enumerate(train_loader):
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion(data, rec_X)
                if verbose and (batch_idx+1) % self.args.log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    print(msg.format(e, batch_idx+1, loss.detach().cpu().numpy()))
                    rec_loss_list.append(loss.detach().cpu().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()
        print('========== End pretraining ==========\n')
        # Initialize clusters in self.clustering after pre-training
        batch_X = []
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True)
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.clustering.init_cluster(batch_X)
        return rec_loss_list
    
    def fit(self, epoch, train_loader, verbose=True):
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(self.device)
            # Get the latent features
            with torch.no_grad():
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()
            # [Step-1] Update the assignment results
            cluster_id = self.clustering.update_assign(latent_X)
            # [Step-2] Update clusters in batch Clustering
            elem_count = np.bincount(cluster_id, minlength=self.args.n_clusters)
            for k in range(self.args.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.clustering.update_cluster(latent_X[cluster_id == k], k)
            # [Step-3] Update the network parameters         
            loss, rec_loss, dist_loss = self._loss(data, cluster_id)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if verbose and (batch_idx+1) % self.args.log_interval == 0:
                msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-Loss: {:.3f} | Dist-Loss: {:.3f}'
                print(msg.format(epoch, batch_idx+1, loss.detach().cpu().numpy(), rec_loss, dist_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Clustering Network')
    parser.add_argument('--dir', default='../Dataset/mnist', help='dataset directory')
    parser.add_argument('--input-dim', type=int, default=28*28, help='input dimension')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--pre-epoch', type=int, default=50, help='number of pre-train epochs')
    parser.add_argument('--pretrain', type=bool, default=True, help='whether use pre-training')
    parser.add_argument('--lamda', type=float, default=0.005, help='coefficient of the reconstruction loss')
    parser.add_argument('--beta', type=float, default=1, help='coefficient of the regularization term on clustering')
    parser.add_argument('--hidden-dims', default=[500, 500, 2000], help='learning rate (default: 1e-4)')
    parser.add_argument('--latent-dim', type=int, default=10, help='latent space dimension')
    parser.add_argument('--n-clusters', type=int, default=10, help='number of clusters in the latent space')
    parser.add_argument('--clustering', type=str, default='kmeans', help='choose a clustering method (default: kmeans) meanshift, tba')
    parser.add_argument('--n-jobs', type=int, default=1, help='number of jobs to run in parallel')
    parser.add_argument('--device', type=str, default='cpu', help='device for computation (default: cpu)')
    parser.add_argument('--log-interval', type=int, default=400, help='how many batches to wait before logging the training status')
    parser.add_argument('--test-run', action='store_true', help='short test run on a few instances of the dataset')
    args = parser.parse_args()
    transformer = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_set = datasets.MNIST(args.dir, train=True, download=True, transform=transformer)
    test_set  = datasets.MNIST(args.dir, train=False, transform=transformer)
    train_limit = list(range(0, len(train_set))) if not args.test_run else list(range(0, 500))    
    test_limit  = list(range(0, len(test_set)))  if not args.test_run else list(range(0, 500))    
    train_loader = torch.utils.data.DataLoader(Subset(train_set, train_limit), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Subset(test_set, test_limit), batch_size=args.batch_size, shuffle=False)
    model = DCN(args)    
    rec_loss_list = model.pretrain(train_loader, epoch=args.pre_epoch)
    nmi_list = []
    ari_list = []
    for e in range(args.epoch):
        model.train()
        model.fit(e, train_loader)
        model.eval()
        y_test = []
        y_pred = []
        for data, target in test_loader:
            batch_size = data.size()[0]
            data = data.view(batch_size, -1).to(model.device)
            latent_X = model.autoencoder(data, latent=True)
            latent_X = latent_X.detach().cpu().numpy()
            y_test.append(target.view(-1, 1).numpy())
            y_pred.append(model.clustering.update_assign(latent_X).reshape(-1, 1))
        y_test = np.vstack(y_test).reshape(-1)
        y_pred = np.vstack(y_pred).reshape(-1)
        NMI = normalized_mutual_info_score(y_test, y_pred)
        ARI = adjusted_rand_score(y_test, y_pred)
        nmi_list.append(NMI)
        ari_list.append(ARI)
        print('Epoch: {:02d} | NMI: {:.3f} | ARI: {:.3f}'.format(e+1, NMI, ARI))
