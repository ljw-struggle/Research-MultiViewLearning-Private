import argparse
from torch.utils.data import DataLoader
from _utils import evaluate, load_data
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
EPS = 1e-10


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


class DDC(nn.Module):
    def __init__(self, input_dim, n_clusters):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()
        hidden_layers = [nn.Linear(input_dim[0], 100), nn.ReLU(), nn.BatchNorm1d(num_features=100)]
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(100, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden


class WeightedMean(nn.Module):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self, n_views, input_sizes):
        super().__init__()
        self.n_views = n_views
        self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)
        self.output_size = self.get_weighted_sum_output_size(input_sizes)

    def get_weighted_sum_output_size(self, input_sizes):
        flat_sizes = [np.prod(s) for s in input_sizes]
        return [flat_sizes[0]]

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)


def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = F.softmax(weights, dim=0)
    out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
    return out


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, feature_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, kernel_size=5, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=5, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, kernel_size=3, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layer(x)


class BaseMVC(nn.Module):
    def __init__(self, input_size, feature_dim, class_num):
        super(BaseMVC, self).__init__()
        self.encoder = Encoder(input_size, feature_dim)
        self.cluster_module = DDC([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, x):
        z = self.encoder(x)
        output, hidden = self.cluster_module(z)
        return output, hidden


class SiMVC(nn.Module):
    def __init__(self, view, input_size, feature_dim):
        super(SiMVC, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        return zs, fused


class SiMVCLarge(nn.Module):
    def __init__(self, view, feature_dim):
        super(SiMVCLarge, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(ConvNet())
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        return zs, fused


class MVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num):
        super(MVC, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)
        self.cluster_module = DDC(self.fusion_module.output_size, class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        output, hidden = self.cluster_module(fused)
        return output, hidden


class MVCLarge(nn.Module):
    def __init__(self, view, feature_dim, class_num):
        super(MVCLarge, self).__init__()
        self.encoders = []
        self.view = view
        for v in range(view):
            self.encoders.append(ConvNet())
        self.encoders = nn.ModuleList(self.encoders)
        input_sizes = []
        for _ in range(view):
            input_sizes.append([feature_dim])
        self.fusion_module = WeightedMean(view, input_sizes=input_sizes)
        self.cluster_module = DDC(self.fusion_module.output_size, class_num)

    def forward(self, xs):
        zs = []
        for v in range(self.view):
            zs.append(self.encoders[v](xs[v]))
        fused = self.fusion_module(zs)
        output, hidden = self.cluster_module(fused)
        return output, hidden


class DSMVC(nn.Module):
    def __init__(self, view_old, view_new, input_size, feature_dim, class_num):
        super(DSMVC, self).__init__()
        self.view = view_new
        self.old_model = SiMVC(view_old, input_size, feature_dim)
        self.new_model = SiMVC(view_new, input_size, feature_dim)
        self.single = Encoder(input_size[view_new-1], feature_dim)
        self.gate = WeightedMean(3, [[feature_dim], [feature_dim], [feature_dim]])
        self.cluster_module = DDC([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs_old, fused_old = self.old_model(xs)
        zs_new, fused_new = self.new_model(xs)
        single = self.single(xs[self.view-1])
        fused = self.gate([fused_old, fused_new, single])
        output, hidden = self.cluster_module(fused)
        return zs_old, zs_new, output, hidden


class DSMVCLarge(nn.Module):
    def __init__(self, view_old, view_new, input_size, feature_dim, class_num):
        super(DSMVCLarge, self).__init__()
        self.view = view_new
        self.old_model = SiMVCLarge(view_old, feature_dim)
        self.new_model = SiMVCLarge(view_new, feature_dim)
        self.single = ConvNet()
        self.gate = WeightedMean(3, [[feature_dim], [feature_dim], [feature_dim]])
        self.cluster_module = DDC([feature_dim], class_num)
        self.apply(he_init_weights)

    def forward(self, xs):
        zs_old, fused_old = self.old_model(xs)
        zs_new, fused_new = self.new_model(xs)
        single = self.single(xs[self.view-1])
        fused = self.gate([fused_old, fused_new, single])
        output, hidden = self.cluster_module(fused)
        return zs_old, zs_new, output, hidden


import torch.nn as nn
import torch
import torch.nn.functional as F


class Safe(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_views = 2
        self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)


def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = F.softmax(weights, dim=0)
    out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
    return out


import torch
import torch.nn as nn
EPSILON = 1E-9


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device

    def forward_cluster(self, hidden, output, print_sign=False):
        hidden_kernel = vector_kernel(hidden, rel_sigma=0.15)
        l1 = self.DDC1(output, hidden_kernel, self.class_num)
        l2 = self.DDC2(output)
        l3 = self.DDC3(self.class_num, output, hidden_kernel)
        if print_sign:
            print(l1.item())
            print(l2.item())
            print(l3.item())
        return l1+l2+l3, l1.item() + l2.item() + l3.item()

    "Adopted from https://github.com/DanielTrosten/mvc"

    def triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _atleast_epsilon(self, X, eps=EPSILON):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: th.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: th.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.

        :param A: Cluster assignment matrix
        :type A:  th.Tensor
        :param K: Kernel matrix
        :type K: th.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: th.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=EPSILON ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self.triu(nom / torch.sqrt(dnom_squared))
        return d

    # ======================================================================================================================
    # Loss terms
    # ======================================================================================================================

    def DDC1(self, output, hidden_kernel, n_clusters):
        """
        L_1 loss from DDC
        """
        # required_tensors = ["hidden_kernel"]
        return self.d_cs(output, hidden_kernel, n_clusters)

    def DDC2(self, output):
        """
        L_2 loss from DDC
        """
        n = output.size(0)
        return 2 / (n * (n - 1)) * self.triu(output @ torch.t(output))

    def DDC2Flipped(self, output, n_clusters):
        """
        Flipped version of the L_2 loss from DDC. Used by EAMC
        """

        return 2 / (n_clusters * (n_clusters - 1)) * self.triu(torch.t(output) @ output)

    def DDC3(self, n_clusters, output, hidden_kernel):
        """
        L_3 loss from DDC
        """

        eye = torch.eye(n_clusters, device=self.device)

        m = torch.exp(-cdist(output, eye))
        return self.d_cs(m, hidden_kernel, n_clusters)


import torch as th
from torch.nn.functional import relu
EPSILON = 1E-9

"Inspired by the implementation in https://github.com/DanielTrosten/mvc"


def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=EPSILON):
    """
    Compute a Gaussian kernel matrix from a distance matrix.

    :param dist: Disatance matrix
    :type dist: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :param min_sigma: Minimum value for sigma. For numerical stability.
    :type min_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
    dist = relu(dist)
    sigma2 = rel_sigma * th.median(dist)
    # Disable gradient for sigma
    sigma2 = sigma2.detach()
    sigma2 = th.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
    k = th.exp(- dist / (2 * sigma2))
    return k


def vector_kernel(x, rel_sigma=0.15):
    """
    Compute a kernel matrix from the rows of a matrix.

    :param x: Input matrix
    :type x: th.Tensor
    :param rel_sigma: Multiplication factor for the sigma hyperparameter
    :type rel_sigma: float
    :return: Kernel matrix
    :rtype: th.Tensor
    """
    return kernel_from_distance_matrix(cdist(x, x), rel_sigma)


def cdist(X, Y):
    """
    Pairwise distance between rows of X and rows of Y.

    :param X: First input matrix
    :type X: th.Tensor
    :param Y: Second input matrix
    :type Y: th.Tensor
    :return: Matrix containing pairwise distances between rows of X and rows of Y
    :rtype: th.Tensor
    """
    xyT = X @ th.t(Y)
    x2 = th.sum(X**2, dim=1, keepdim=True)
    y2 = th.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + th.t(y2)
    return d

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='caltech_5m')
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--epochs", default=120)
parser.add_argument("--view", type=int, default=2)
args = parser.parse_args()
if args.dataset == "mnist_mv":
    args.feature_dim = 288


def train_safe_epoch(epoch, view, model, data_loader, criterion, optimizer, records, device):
    tot_loss = 0.
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, output, hidden = model(xs)
        loss, _ = criterion.forward_cluster(hidden, output)
        loss.backward()
        optimizer.step(epoch - 1 + batch_idx / len(data_loader))
        tot_loss += loss.item()
    if epoch == 120:
        records['safe'].append(tot_loss/len(data_loader))


class Optimizer:
    def __init__(self, params):
        """
        Wrapper class for optimizers

        :param cfg: Optimizer config
        :type cfg: config.defaults.Optimizer
        :param params: Parameters to associate with the optimizer
        :type params:
        """
        self.clip_norm = 5.0
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.5
        self.params = params
        self._opt = torch.optim.Adam(params, lr=1e-3)
        if self.scheduler_step_size is not None:
            assert self.scheduler_gamma is not None
            self._sch = torch.optim.lr_scheduler.StepLR(self._opt, step_size=self.scheduler_step_size,
                                                     gamma=self.scheduler_gamma)
        else:
            self._sch = None

    def zero_grad(self):
        return self._opt.zero_grad()

    def step(self, epoch):
        if self._sch is not None:
            # Only step the scheduler at integer epochs, and don't step on the first epoch.
            if epoch.is_integer() and epoch > 0:
                self._sch.step()

        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_norm)

        out = self._opt.step()
        return out


def valid(model, device, dataset, total_view):
    loader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
    )
    model.eval()
    pred_vector = []
    labels_vector = []

    for _, (xs, y, _) in enumerate(loader):
        for v in range(total_view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, output, _ = model(xs)
            pred_vector.extend(output.detach().cpu().numpy())
        labels_vector.extend(y.numpy())

    labels = np.array(labels_vector).reshape(len(labels_vector))
    pred_vec = np.argmax(np.array(pred_vector), axis=1)
    nmi, ari, acc, pur = evaluate(labels, pred_vec)
    return [acc, nmi, ari, pur]


def main():
    # prepare data and initial hyper-parameters
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_train_epoch = args.epochs
    view_num = args.view
    T = 20
    records = {"safe": [[], [], [], []]}
    loss_record = {"safe": []}

    for t in range(T):
        print('Iter:{}'.format(t))

        # initial safe MVC model
        safe_model = DSMVC(view_num-1, view_num, dims, args.feature_dim, class_num)
        safe_model = safe_model.to(device)
        criterion_safe = Loss(args.batch_size, class_num, device)

        for epoch in range(safe_train_epoch):
            if (epoch//20) % 2 == 0:
                for p in safe_model.gate.parameters():
                    p.requires_grad = False
                for p in safe_model.old_model.parameters():
                    p.requires_grad = True
                for p in safe_model.new_model.parameters():
                    p.requires_grad = True
                for p in safe_model.single.parameters():
                    p.requires_grad = True
                for p in safe_model.cluster_module.parameters():
                    p.requires_grad = True
                optimizer_theta = Optimizer(filter(lambda p: p.requires_grad, safe_model.parameters()))
                train_safe_epoch(epoch+1, view, safe_model, data_loader, criterion_safe, optimizer_theta, loss_record, device)
            else:
                for p in safe_model.gate.parameters():
                    p.requires_grad = True
                for p in safe_model.old_model.parameters():
                    p.requires_grad = False
                for p in safe_model.new_model.parameters():
                    p.requires_grad = False
                for p in safe_model.single.parameters():
                    p.requires_grad = False
                for p in safe_model.cluster_module.parameters():
                    p.requires_grad = False
                optimizer_lambda = Optimizer(filter(lambda p: p.requires_grad, safe_model.parameters()))
                train_safe_epoch(epoch+1, view, safe_model, data_loader, criterion_safe, optimizer_lambda, loss_record,
                                 device)

        res = valid(safe_model, device, dataset, view)
        for i in range(4):
            records["safe"][i].append(res[i])

        state = safe_model.state_dict()
        torch.save(state, './models/' + args.dataset + '/' + str(t) + '.pth')

    ind_ = np.argmax(np.array(records["safe"][0]))
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(records["safe"][0][ind_],
                                                        records["safe"][1][ind_],
                                                        records["safe"][3][ind_]))


if __name__ == '__main__':
    main()
