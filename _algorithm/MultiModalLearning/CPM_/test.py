import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import warnings
import numpy as np
from numpy.random import shuffle
import util.classfiy as classfiy
from util.util import read_data, xavier_init
from util.get_sn import get_sn
warnings.filterwarnings("ignore")
device = torch.device('cuda:2')

class CPMNets(nn.Module): # The architecture of the CPM
    def __init__(self, view_num, trainLen, testLen, layer_size, v, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNets, self).__init__()
        # initialize parameter
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        #initialize forward methods 
        self.net = self._make_view(v).cuda()

    def forward(self,h):
        h_views = self.net(h.cuda())
        return h_views
    '''
    def initialize_weight(self, dims_net):
        all_weight = dict()
        all_weight['w0'] = Variable(xavier_init(self.lsd_dim, dims_net[0]),requires_grad = True)
        all_weight['b0'] = Variable(torch.zeros([dims_net[0]]),requires_grad = True)
        for num in range(1, len(dims_net)):
            all_weight['w' + str(num)] = Variable(xavier_init(dims_net[num - 1], dims_net[num]),requires_grad = True)
            all_weight['b' + str(num)] = Variable(torch.zeros([dims_net[num]]),requires_grad = True)
        return all_weight
    '''
    def _make_view(self, v):
        dims_net = self.layer_size[v]
        net1 = nn.Sequential()
        w = torch.nn.Linear(self.lsd_dim, dims_net[0])
        nn.init.xavier_normal_(w.weight)
        nn.init.constant_(w.bias, 0.0)
        net1.add_module('lin'+str(0), w)
        for num in range(1, len(dims_net)):
            w = torch.nn.Linear(dims_net[num - 1], dims_net[num])
            nn.init.xavier_normal_(w.weight)
            nn.init.constant_(w.bias, 0.0)
            net1.add_module('lin'+str(num), w)
            net1.add_module('drop'+str(num), torch.nn.Dropout(p=0.1))
        return net1

class CPMNet_Works(nn.Module): # Main parts of the test code
    def __init__(self, device, view_num, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001], lamb=1):
        """
        :param learning_rate:learning rate of network and h
        :param view_num:view number
        :param layer_size:node of each net
        :param lsd_dim:latent space dimensionality
        :param trainLen:training dataset samples
        :param testLen:testing dataset samples
        """
        super(CPMNet_Works, self).__init__()
        # initialize parameter
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.device = device
        # initialize latent space data
        self.h_train = self.H_init('train')
        self.h_test = self.H_init('test')
        self.h = torch.cat([self.h_train, self.h_test], axis=0).cuda()
        # initialize nets for different views
        self.net, self.train_net_op = self.bulid_model()
        
    def H_init(self, a):
        if a == 'train':
            h = Variable(xavier_init(self.trainLen, self.lsd_dim), requires_grad = True)
        elif a == 'test':
            h = Variable(xavier_init(self.testLen, self.lsd_dim), requires_grad = True)
        return h

    def reconstruction_loss(self,h,x,sn):
        loss = 0
        x_pred = self.calculate(h.cuda())
        for num in range(self.view_num):
            loss = loss + (torch.pow((x_pred[str(num)].cpu() - x[str(num)].cpu()) , 2.0) * sn[str(num)].cpu()).sum()
        return loss

    def classification_loss(self,label_onehot, gt, h_temp):
        h_temp = h_temp.float()
        h_temp = h_temp.cuda()
        F_h_h = torch.mm(h_temp, (h_temp.T))
        F_hn_hn = torch.eye(F_h_h.shape[0],F_h_h.shape[1])
        F_h_h = F_h_h - F_h_h * (F_hn_hn.cuda())
        label_num = label_onehot.sum(0, keepdim=True)  # should sub 1.Avoid numerical errors; the number of samples of per label
        label_onehot = label_onehot.float()
        F_h_h_sum = torch.mm(F_h_h, label_onehot)
        F_h_h_mean = F_h_h_sum / label_num
        gt1 = torch.max(F_h_h_mean, axis=1)[1]  # gt begin from 1
        gt_ = gt1.type(torch.IntTensor) + 1
        F_h_h_mean_max = torch.max(F_h_h_mean, axis=1, keepdim=False)[0]
        gt_ = gt_.cuda()
        gt_ = gt_.reshape([gt_.shape[0],1])
        theta = torch.ne(gt, gt_).type(torch.FloatTensor)
        F_h_hn_mean_ = F_h_h_mean * label_onehot
        F_h_hn_mean = F_h_hn_mean_.sum(axis=1)
        F_h_h_mean_max = F_h_h_mean_max.reshape([F_h_h_mean_max.shape[0],1])
        F_h_hn_mean = F_h_hn_mean.reshape([F_h_hn_mean.shape[0],1])
        theta = theta.cuda()
        return (torch.nn.functional.relu(theta + F_h_h_mean_max - F_h_hn_mean)).sum()

    def train(self, data, sn, label_onehot, gt, epoch, step=[5, 5]):
        global Reconstruction_LOSS
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        gt = gt.cuda()
        label_onehot = label_onehot.cuda()
        sn1 = dict()
        data1 = dict()
        for v_num in range(self.view_num):
            data1[str(v_num)] = torch.from_numpy(data[str(v_num)]).cuda() 
        for i in range(self.view_num):
            sn1[str(i)] = sn[:, i].reshape(self.trainLen, 1).cuda()
        train_hn_op = torch.optim.Adam([self.h_train], self.learning_rate[1])
        for iter in range(epoch):
            for i in range(step[0]):
                Reconstruction_LOSS = self.reconstruction_loss(self.h_train,data1,sn1).float()
                for v_num in range(self.view_num):
                    self.train_net_op[v_num].zero_grad()
                    Reconstruction_LOSS.backward(retain_graph=True)
                    self.train_net_op[v_num].step()
            for i in range(step[1]):
                loss1 = self.reconstruction_loss(self.h_train,data1,sn1).float().cuda() 
                loss2 = self.lamb * self.classification_loss(label_onehot, gt, self.h_train).float().cuda()
                train_hn_op.zero_grad()
                loss1.backward()
                loss2.backward()
                train_hn_op.step()
            Classification_LOSS = self.classification_loss(label_onehot,gt,self.h_train)
            Reconstruction_LOSS = self.reconstruction_loss(self.h_train,data1,sn1)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, Classification Loss = {:.4f} ".format((iter + 1), Reconstruction_LOSS, Classification_LOSS)
            print(output)
        return (self.h_train)

    def bulid_model(self):
        # initialize network
        net = dict()
        train_net_op = []
        for v_num in range(self.view_num):
            net[str(v_num)] = CPMNets(self.view_num, self.trainLen, self.testLen, self.layer_size, v_num,
            self.lsd_dim, self.learning_rate, self.lamb).cuda()
            train_net_op.append(torch.optim.Adam([{"params":net[str(v_num)].parameters()}], self.learning_rate[0]))
        return net,train_net_op

    def calculate(self,h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[str(v_num)] = self.net[str(v_num)](h.cuda())
        return h_views

    def test(self, data, sn, epoch):
        sn1 = dict()
        data1 = dict()
        for v_num in range(self.view_num):
            data1[str(v_num)] = torch.from_numpy(data[str(v_num)]).cuda() 
        for i in range(self.view_num):
            sn1[str(i)] = sn[:, i].reshape(self.testLen, 1).cuda()
        adj_hn_op = torch.optim.Adam([self.h_test], self.learning_rate[0])
        for iter in range(epoch):
            # update the h
            for i in range(5):
                Reconstruction_LOSS = self.reconstruction_loss(self.h_test, data1, sn1).float()
                adj_hn_op.zero_grad()
                Reconstruction_LOSS.backward()
                adj_hn_op.step()
            Reconstruction_LOSS = self.reconstruction_loss(self.h_test, data1, sn1).float()
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}".format((iter + 1), Reconstruction_LOSS)
            print(output)
        return self.h_test

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lsd-dim', type=int, default=150, help='dimensionality of the latent space data [default: 150]')
    parser.add_argument('--epochs-train', type=int, default=30, metavar='N', help='number of epochs to train [default: 30]')
    parser.add_argument('--epochs-test', type=int, default=30, metavar='N', help='number of epochs to test [default: 30]')
    parser.add_argument('--lamb', type=float, default=1, help='trade off parameter [default: 1]')
    parser.add_argument('--missing-rate', type=float, default=0, help='view missing rate [default: 0]')
    args = parser.parse_args()

    # read data
    trainData, testData, view_num = read_data('./data/PIE_face_10.mat', 0.8, 1)
    outdim_size = [trainData.data[str(i)].shape[1] for i in range(view_num)]
    # set layer size
    layer_size = [[300, outdim_size[i]] for i in range(view_num)]
    # set parameter
    epoch = [args.epochs_train, args.epochs_test]
    learning_rate = [0.01, 0.01]
    # Randomly generated missing matrix
    Sn = get_sn(view_num, trainData.num_examples + testData.num_examples, args.missing_rate)
    Sn_train = Sn[np.arange(trainData.num_examples)]
    Sn_test = Sn[np.arange(testData.num_examples) + trainData.num_examples]
    
    Sn = torch.LongTensor(Sn).cuda()
    Sn_train = torch.LongTensor(Sn_train).cuda()
    Sn_test = torch.LongTensor(Sn_test).cuda()

    # Model building
    model = CPMNet_Works(device, view_num, trainData.num_examples, testData.num_examples, layer_size, args.lsd_dim, learning_rate, args.lamb).cuda()
        
    # train
    gt1 = trainData.labels.reshape(trainData.num_examples)
    gt1 = gt1.reshape([gt1.shape[0],1])
    gt1 = torch.LongTensor(gt1)
    class_num = (torch.max(gt1) - torch.min(gt1) + 1).cpu()
    batch_size = torch.tensor(gt1.shape[0])
    label_onehot = (torch.zeros(batch_size,class_num).scatter_(1,gt1 - 1,1)) # gt1 begin from 1 so we need to set the minimum of it to 0
    H_train = model.train(trainData.data, Sn_train, label_onehot, gt1, epoch[0])
    
    # test
    gt2 = testData.labels.reshape(testData.num_examples)
    gt2 = gt2.reshape([gt2.shape[0],1])
    gt2 = torch.LongTensor(gt2)
    H_test = model.test(testData.data, Sn_test, epoch[1])

    label_pre = classfiy.ave(H_train, H_test, label_onehot.cuda(), testData.num_examples)
    print('Accuracy on the test set is {:.4f}'.format(accuracy_score(testData.labels, label_pre)))
    