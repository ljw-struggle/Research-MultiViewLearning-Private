# -*- coding: utf-8 -*-
import os, argparse
import urllib, tarfile, pickle

import numpy as np
import sklearn.metrics as metrics
import sklearn.datasets as datasets

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def load_data(data_name, labeled, unlabeled):
    os.makedirs('../data') if not os.path.exists('../data') else None
    assert data_name in ['mnist', 'cifar'], 'dataset name {} is unknown.'.format(data_name)
    if data_name == 'mnist':
        mnist = datasets.fetch_openml('mnist_784', data_home='../data')
        x = np.reshape(np.array(mnist.data, dtype=np.float32), (mnist.data.shape[0], 1, 28, 28)).astype(np.float32) / 255.
        y = mnist.target.astype(np.int32)
        x_train, y_train = x[:60000], y[:60000] # x_train.shape = (60000, 1, 28, 28), y_train.shape = (60000,)
        x_test, y_test = x[60000:], y[60000:] # x_test.shape = (10000, 1, 28, 28), y_test.shape = (10000,)
        y_train_binary = np.ones(len(y_train), dtype=np.int32); y_train_binary[y_train % 2 == 1] = -1
        y_test_binary = np.ones(len(y_test), dtype=np.int32); y_test_binary[y_test % 2 == 1] = -1
    if data_name == 'cifar':
        os.makedirs('../data/mldata') if not os.path.exists('../data/mldata') else None
        urllib.request.urlretrieve('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', '../data/mldata/cifar-10-python.tar.gz') if not os.path.exists('../data/mldata/cifar-10-python.tar.gz') else None
        tarfile.open('../data/mldata/cifar-10-python.tar.gz').extractall('../data/mldata', None, numeric_owner=False) if not os.path.isdir('../data/mldata/cifar-10-batches-py') else None
        urllib.request.urlcleanup() if os.path.exists('../data/mldata/cifar-10-python.tar.gz') else None
        x_train = [pickle.load(open(os.path.join('../data/mldata/cifar-10-batches-py', 'data_batch_%d' % i), 'rb'), encoding='latin1')['data'] for i in range(1, 6)]
        y_train = [pickle.load(open(os.path.join('../data/mldata/cifar-10-batches-py', 'data_batch_%d' % i), 'rb'), encoding='latin1')['labels'] for i in range(1, 6)]
        x_train, y_train = np.concatenate(x_train, axis=0), np.concatenate(y_train, axis=0) # x_train.shape = (50000, 3072), y_train.shape = (50000,)
        data_dict = pickle.load(open(os.path.join('../data/mldata/cifar-10-batches-py', 'test_batch'), 'rb'), encoding='latin1')
        x_test, y_test = data_dict['data'], data_dict['labels'] # x_test.shape = (10000, 3072), y_test.shape = (10000,)
        x_train = np.reshape(x_train, (x_train.shape[0], 3, 32, 32)).astype(np.float32) / 255. # x_train.shape = (50000, 3, 32, 32)
        x_test = np.reshape(x_test, (x_test.shape[0], 3, 32, 32)).astype(np.float32) / 255. # x_test.shape = (10000, 3, 32, 32)
        y_train_binary = np.ones(len(y_train), dtype=np.int32); y_train_binary[(y_train == 2) | (y_train == 3) | (y_train == 4) | (y_train == 5) | (y_train == 6) | (y_train == 7)] = -1
        y_test_binary = np.ones(len(y_test), dtype=np.int32); y_test_binary[(y_test == 2) | (y_test == 3) | (y_test == 4) | (y_test == 5) | (y_test == 6) | (y_test == 7)] = -1

    assert (labeled + unlabeled) == len(x_train), 'The number of labeled and unlabeled data should be equal to the number of all data.' # make PU dataset
    np.random.seed(0); index = np.random.permutation(len(x_train)); x_train, y_train_binary = x_train[index], y_train_binary[index] # shuffle data
    xlp, xup, xun = x_train[y_train_binary == 1][:labeled], x_train[y_train_binary == 1][labeled:], x_train[y_train_binary == -1]; prior = len(xup) / (len(xup) + len(xun))
    x_train = np.concatenate((xlp, xup, xun), axis=0); y_train_binary = np.concatenate((np.ones(labeled), -np.ones(unlabeled)), axis=0)
    np.random.seed(0); index = np.random.permutation(len(x_train)); x_train, y_train_binary = x_train[index], y_train_binary[index] # shuffle data
    return (x_train, y_train_binary, prior), (x_test, y_test_binary)

class TLP(nn.Module):
    def __init__(self, dim=784):
        super(TLP, self).__init__()
        self.l1 = nn.Linear(dim, 100); self.l2 = nn.Linear(100, 1); self.relu = nn.ReLU()
    
    def forward(self, x):
        h = self.relu(self.l1(x.view(x.shape[0], -1))); h = self.l2(h)
        return h

class MLP(nn.Module):
    def __init__(self, dim=784):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(dim, 300); self.b1 = nn.BatchNorm1d(300); self.l2 = nn.Linear(300, 300); self.b2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300); self.b3 = nn.BatchNorm1d(300); self.l4 = nn.Linear(300, 300); self.b4 = nn.BatchNorm1d(300)
        self.l5 = nn.Linear(300, 1); self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.b1(self.l1(x.view(x.shape[0], -1)))); h = self.relu(self.b2(self.l2(h)))
        h = self.relu(self.b3(self.l3(h))); h = self.relu(self.b4(self.l4(h))); h = self.l5(h)
        return h

class CNN(nn.Module):
    def __init__(self, dim=10*8*8):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,   96,  3, padding='same'); self.b1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96,  96,  3, padding='same'); self.b2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96,  96,  3, padding=1, stride=2); self.b3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96,  192, 3, padding='same'); self.b4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, padding='same'); self.b5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2); self.b6 = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(192, 192, 3, padding='same'); self.b7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, 1, padding='same'); self.b8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 10,  1, padding='same'); self.b9 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(dim, 1000); self.fc2 = nn.Linear(1000, 1000); self.fc3 = nn.Linear(1000, 1); self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.b1(self.conv1(x))); h = self.relu(self.b2(self.conv2(h))); h = self.relu(self.b3(self.conv3(h)))
        h = self.relu(self.b4(self.conv4(h))); h = self.relu(self.b5(self.conv5(h))); h = self.relu(self.b6(self.conv6(h)))
        h = self.relu(self.b7(self.conv7(h))); h = self.relu(self.b8(self.conv8(h))); h = self.relu(self.b9(self.conv9(h)))
        h = self.relu(self.fc1(h.view(h.shape[0], -1))); h = self.relu(self.fc2(h)); h = self.fc3(h)
        return h

class PULoss(torch.nn.Module):
    def __init__(self, prior, beta=0, gamma=1, nnpu=True):
        super(PULoss, self).__init__()
        self.prior = prior; self.beta = beta; self.gamma = gamma; self.nnpu = nnpu
        self.criterion = lambda pred, label: 1 / (1 + torch.exp(pred * label))

    def forward(self, pred, label):
        positive_risk = self.prior * (1 / max([1., (label == 1).sum()])) * torch.sum(self.criterion(pred, 1) * (label == 1))
        negative_risk = - self.prior * (1 / max([1., (label == 1).sum()])) * torch.sum(self.criterion(pred, -1) * (label == 1)) + \
                                       (1 / max([1., (label == -1).sum()])) * torch.sum(self.criterion(pred, -1) * (label == -1))
        loss = positive_risk + negative_risk
        backward_loss = positive_risk + negative_risk
        if self.nnpu:
            loss = positive_risk - self.beta if negative_risk < - self.beta else (positive_risk + negative_risk)
            backward_loss = - self.gamma * negative_risk if negative_risk < - self.beta else (positive_risk + negative_risk)
        return loss, backward_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='non-negative / unbiased PU learning Chainer implementation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--preset', '-pre', default=None, type=str, choices=['exp-mnist-tlp', 'exp-mnist-mlp', 'exp-cifar-cnn'], help='The preset of configuration')
    parser.add_argument('--seed', '-s', default=0, type=int, help='The random seed')
    parser.add_argument('--model', '-m', default='tlp', type=str, choices=['linear', 'tlp', 'mlp'], help='The model')
    parser.add_argument('--data', '-d', default='mnist', type=str, choices=['mnist', 'cifar'], help='The data')
    parser.add_argument('--positive', '-l', default=100, type=int, help='The number of positive data')
    parser.add_argument('--unlabeled', '-u', default=59900, type=int, help='The number of unlabeled data')
    parser.add_argument('--beta', '-b', default=0., type=float, help='The beta parameter of nnPU')
    parser.add_argument('--gamma', '-g', default=1., type=float, help='The gamma parameter of nnPU')
    parser.add_argument('--epoch', '-e', default=100, type=int, help='The number of epochs')
    parser.add_argument('--batch_size', '-batch', default=30000, type=int, help='The batch size')
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='The learning rate')
    parser.add_argument('--output_dir', '-o', default='../result/exp-mnist-tlp', type=str, help='The output directory of the result')
    args = parser.parse_args()
    if args.preset == 'exp-mnist-tlp':
        args.model = 'tlp'; args.data = 'mnist'; args.positive = 100 ; args.unlabeled = 59900; args.batch_size = 30000; args.learning_rate = 1e-3; args.output_dir = '../result/exp-mnist-tlp'
    if args.preset == 'exp-mnist-mlp':
        args.model = 'mlp'; args.data = 'mnist'; args.positive = 1000; args.unlabeled = 59000; args.batch_size = 30000; args.Learning_rate = 1e-3; args.output_dir = '../result/exp-mnist-mlp'
    if args.preset == 'exp-cifar-cnn':
        args.model = 'cnn'; args.data = 'cifar'; args.positive = 1000; args.unlabeled = 49000; args.batch_size = 500  ; args.learning_rate = 1e-5; args.output_dir = '../result/exp-cifar-cnn'
    os.makedirs(args.output_dir) if not os.path.exists(args.output_dir) else None
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    (x_train, y_train, prior), (x_test, y_test) = load_data(args.data, args.positive, args.unlabeled)
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = {'tlp': TLP, 'mlp': MLP, 'cnn': CNN}
    models = {'uPU': model_dict[args.model]().to(device),
              'nnPU': model_dict[args.model]().to(device)}
    losses = {'uPU': PULoss(prior, beta=args.beta, gamma=args.gamma, nnpu=False), 
              'nnPU': PULoss(prior, beta=args.beta, gamma=args.gamma, nnpu=True)}
    writers = {'uPU': SummaryWriter(os.path.join(args.output_dir, 'uPU')), 
               'nnPU': SummaryWriter(os.path.join(args.output_dir, 'nnPU'))}
    optimizers = {'uPU': torch.optim.Adam(models['uPU'].parameters(), lr=args.learning_rate, weight_decay=5e-3), 
                  'nnPU': torch.optim.Adam(models['nnPU'].parameters(), lr=args.learning_rate, weight_decay=5e-3)}

    for epoch in range(args.epoch):
        for _, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            for key, model in models.items():
                model.train()
                optimizers[key].zero_grad()
                loss, backward_loss = losses[key](model(x_batch).squeeze(), y_batch.float())
                backward_loss.backward()
                optimizers[key].step()
        
        summary = {'nnPU': np.zeros(4), 'uPU': np.zeros(4)}; summary_error = {'nnPU': 0, 'uPU': 0}
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(valid_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                for key, model in models.items():
                    model.eval()
                    output = torch.sign(model(x_batch))
                    confusion_matrix = metrics.confusion_matrix(y_batch.cpu().numpy(), output.cpu().numpy(), labels=[-1, 1])
                    summary[key] += confusion_matrix[1, 1], confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0] # t_p, t_n, f_p, f_n
        for key, values in summary.items():
            t_p, t_n, f_p, f_n = values
            error_p = float(f_n) / (t_p + f_n)
            error_u = float(f_p) / (t_n + f_p)
            error = 2 * prior * error_p + error_u - prior
            summary_error[key] = error
            writers[key].add_scalar('train/error'.format(key), error, epoch)
        print('Train epoch: {}, nnPU error: {:.4f}, uPU error: {:.4f}'.format(epoch, summary_error['nnPU'], summary_error['uPU']))

        summary = {'nnPU': np.zeros(4), 'uPU': np.zeros(4)}; summary_error = {'nnPU': 0, 'uPU': 0}
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(test_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                for key, model in models.items():
                    model.eval()
                    output = torch.sign(model(x_batch))
                    confusion_matrix = metrics.confusion_matrix(y_batch.cpu().numpy(), output.cpu().numpy(), labels=[-1, 1])
                    summary[key] += confusion_matrix[1, 1], confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0] # t_p, t_n, f_p, f_n
        for key, values in summary.items():
            t_p, t_n, f_p, f_n = values
            error = float(f_p + f_n) / (t_p + t_n + f_p + f_n)
            summary_error[key] = error
            writers[key].add_scalar('test/error'.format(key), error, epoch)
        print('Test epoch: {}, nnPU error: {:.4f}, uPU error: {:.4f}'.format(epoch, summary_error['nnPU'], summary_error['uPU']))
    
    torch.save(model, os.path.join(args.output_dir, 'model.pth'))
