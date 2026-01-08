import os, random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torchvision import datasets, transforms

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform
        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()
        self.n_data = len(data_list)
        self.img_paths = []
        self.img_labels = []
        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')
        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)
        return imgs, labels

    def __len__(self):
        return self.n_data

def test(dataset_name, epoch):
    assert dataset_name in ['MNIST', 'mnist_m']
    model_root = os.path.join('..', 'models')
    image_root = os.path.join('..', 'dataset', dataset_name)
    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0
    """load data"""
    img_transform_source = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    img_transform_target = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset_name == 'mnist_m':
        test_list = os.path.join(image_root, 'mnist_m_test_labels.txt')
        dataset = GetLoader(data_root=os.path.join(image_root, 'mnist_m_test'), data_list=test_list, transform=img_transform_target)
    else:
        dataset = datasets.MNIST(root='../dataset', train=False, transform=img_transform_source)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    """ training """
    my_net = torch.load(os.path.join(model_root, 'mnist_mnistm_model_epoch_' + str(epoch) + '.pth'))
    my_net = my_net.eval()
    if cuda:
        my_net = my_net.cuda()
    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)
    i = 0
    n_total = 0
    n_correct = 0
    while i < len_dataloader:
        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target
        batch_size = len(t_label)
        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)
        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size
        i += 1
    accu = n_correct.data.numpy() * 1.0 / n_total
    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))

def main():
    print('main function')

if __name__ == '__main__':
    source_dataset_name = 'MNIST'
    target_dataset_name = 'mnist_m'
    source_image_root = os.path.join('..', 'dataset', source_dataset_name)
    target_image_root = os.path.join('..', 'dataset', target_dataset_name)
    model_root = os.path.join('..', 'models')
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 128
    image_size = 28
    n_epoch = 100
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    # load data
    img_transform_source = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
    img_transform_target = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset_source = datasets.MNIST(root='../dataset', train=True, transform=img_transform_source, download=True)
    dataloader_source = torch.utils.data.DataLoader(dataset=dataset_source, batch_size=batch_size, shuffle=True, num_workers=8)
    train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    dataset_target = GetLoader(data_root=os.path.join(target_image_root, 'mnist_m_train'), data_list=train_list, transform=img_transform_target)
    dataloader_target = torch.utils.data.DataLoader(dataset=dataset_target, batch_size=batch_size, shuffle=True, num_workers=8)
    # load model
    my_net = CNNModel()
    # setup optimizer
    optimizer = optim.Adam(my_net.parameters(), lr=lr)
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()
    for p in my_net.parameters():
        p.requires_grad = True
    # training
    for epoch in range(n_epoch):
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)
        i = 0
        while i < len_dataloader:
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source
            my_net.zero_grad()
            batch_size = len(s_label)
            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()
            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()
            input_img.resize_as_(s_img).copy_(s_img)
            class_label.resize_as_(s_label).copy_(s_label)
            class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
            err_s_label = loss_class(class_output, class_label)
            err_s_domain = loss_domain(domain_output, domain_label)
            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target
            batch_size = len(t_img)
            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()
            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                domain_label = domain_label.cuda()
            input_img.resize_as_(t_img).copy_(t_img)
            _, domain_output = my_net(input_data=input_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()
            i += 1
            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(), err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

        torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
        test(source_dataset_name, epoch)
        test(target_dataset_name, epoch)
