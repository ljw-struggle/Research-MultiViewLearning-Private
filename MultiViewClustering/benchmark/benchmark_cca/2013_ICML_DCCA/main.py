import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import time
import logging
from _utils import load_data, svm_classify
import gzip
import numpy as np
import torch.nn as nn
torch.set_default_tensor_type(torch.DoubleTensor)
import numpy

class linear_cca():
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):
        """
        An implementation of linear CCA
        # Arguments:
            H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
            outdim_size: specifies the number of new features
        # Returns
            A and B: the linear transformation matrices
            mean1 and mean2: the means of data for both views
        """
        r1 = 1e-4
        r2 = 1e-4
        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]
        self.m[0] = numpy.mean(H1, axis=0)
        self.m[1] = numpy.mean(H2, axis=0)
        H1bar = H1 - numpy.tile(self.m[0], (m, 1))
        H2bar = H2 - numpy.tile(self.m[1], (m, 1))
        SigmaHat12 = (1.0 / (m - 1)) * numpy.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * numpy.dot(H1bar.T, H1bar) + r1 * numpy.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * numpy.dot(H2bar.T, H2bar) + r2 * numpy.identity(o2)
        [D1, V1] = numpy.linalg.eigh(SigmaHat11)
        [D2, V2] = numpy.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = numpy.dot(numpy.dot(V1, numpy.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = numpy.dot(numpy.dot(V2, numpy.diag(D2 ** -0.5)), V2.T)
        Tval = numpy.dot(numpy.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        [U, D, V] = numpy.linalg.svd(Tval)
        V = V.T
        self.w[0] = numpy.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        self.w[1] = numpy.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]

    def _get_result(self, x, idx):
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = numpy.dot(result, self.w[idx])
        return result

    def test(self, H1, H2):
        return self._get_result(H1, 0), self._get_result(H2, 1)


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False), nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),))
            else:
                layers.append(nn.Sequential(nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]), nn.Sigmoid(), nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DeepCCA(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()
        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss

    def forward(self, x1, x2):
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        return output1, output2
    
class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        """
        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9
        H1, H2 = H1.t(), H2.t()
        # assert torch.isnan(H1).sum().item() == 0
        # assert torch.isnan(H2).sum().item() == 0
        o1 = o2 = H1.size(0)
        m = H1.size(1)
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        # assert torch.isnan(H1bar).sum().item() == 0
        # assert torch.isnan(H2bar).sum().item() == 0
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        # assert torch.isnan(SigmaHat11).sum().item() == 0
        # assert torch.isnan(SigmaHat12).sum().item() == 0
        # assert torch.isnan(SigmaHat22).sum().item() == 0
        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
        # assert torch.isnan(D1).sum().item() == 0
        # assert torch.isnan(D2).sum().item() == 0
        # assert torch.isnan(V1).sum().item() == 0
        # assert torch.isnan(V2).sum().item() == 0
        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())
        SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())
        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
            U, V = torch.symeig(trace_TT, eigenvectors=True)
            U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(self.device))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr

class Solver():
    def __init__(self, model, linear_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=torch.device('cpu')):
        self.model = nn.DataParallel(model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = model.loss
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, weight_decay=reg_par)
        self.device = device
        self.linear_cca = linear_cca
        self.outdim_size = outdim_size
        formatter = logging.Formatter("[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)

    def fit(self, x1, x2, vx1=None, vx2=None, tx1=None, tx2=None, checkpoint='checkpoint.model'):
        x1.to(self.device)
        x2.to(self.device)
        data_size = x1.size(0)
        if vx1 is not None and vx2 is not None:
            best_val_loss = 0
            vx1.to(self.device)
            vx2.to(self.device)
        if tx1 is not None and tx2 is not None:
            tx1.to(self.device)
            tx2.to(self.device)
        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)
            info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            if vx1 is not None and vx2 is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = self.test(vx1, vx2)
                    info_string += " - val_loss: {:.4f}".format(val_loss)
                    if val_loss < best_val_loss:
                        self.logger.info(
                            "Epoch {:d}: val_loss improved from {:.4f} to {:.4f}, saving model to {}".format(epoch + 1, best_val_loss, val_loss, checkpoint))
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), checkpoint)
                    else:
                        self.logger.info("Epoch {:d}: val_loss did not improve from {:.4f}".format(
                            epoch + 1, best_val_loss))
            else:
                torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            self.logger.info(info_string.format(
                epoch + 1, self.epoch_num, epoch_time, train_loss))
        if self.linear_cca is not None:
            _, outputs = self._get_outputs(x1, x2)
            self.train_linear_cca(outputs[0], outputs[1])
        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)
        if vx1 is not None and vx2 is not None:
            loss = self.test(vx1, vx2)
            self.logger.info("loss on validation data: {:.4f}".format(loss))
        if tx1 is not None and tx2 is not None:
            loss = self.test(tx1, tx2)
            self.logger.info('loss on test data: {:.4f}'.format(loss))

    def test(self, x1, x2, use_linear_cca=False):
        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)
            if use_linear_cca:
                print("Linear CCA started!")
                outputs = self.linear_cca.test(outputs[0], outputs[1])
                return np.mean(losses), outputs
            else:
                return np.mean(losses)

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(
                range(data_size)), batch_size=self.batch_size, drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        return losses, outputs

if __name__ == '__main__':
    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")
    # the path to save the final learned features
    save_to = './new_features.gz'
    # the size of the new space learned by the model (number of the new features)
    outdim_size = 10
    # size of the input for view 1 and view 2
    input_shape1 = 784
    input_shape2 = 784
    # number of layers with nodes in each one
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]
    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 1
    batch_size = 800
    # the regularization parameter of the network
    # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
    reg_par = 1e-5
    # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
    # if one option does not work for a network or dataset, try the other one
    use_all_singular_values = False
    # if a linear CCA should get applied on the learned features extracted from the networks
    # it does not affect the performance on noisy MNIST significantly
    apply_linear_cca = True
    # end of parameters section
    ############
    # Each view is stored in a gzip file separately. They will get downloaded the first time the code gets executed.
    # Datasets get stored under the datasets folder of user's Keras folder
    # normally under [Home Folder]/.keras/datasets/
    data1 = load_data('./noisymnist_view1.gz')
    data2 = load_data('./noisymnist_view2.gz')
    # Building, training, and producing the new features by DCCA
    model = DeepCCA(layer_sizes1, layer_sizes2, input_shape1, input_shape2, outdim_size, use_all_singular_values, device=device).double()
    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size, learning_rate, reg_par, device=device)
    train1, train2 = data1[0][0], data2[0][0]
    val1, val2 = data1[1][0], data2[1][0]
    test1, test2 = data1[2][0], data2[2][0]
    solver.fit(train1, train2, val1, val2, test1, test2)
    # TODO: Save l_cca model if needed
    set_size = [0, train1.size(0), train1.size(0) + val1.size(0), train1.size(0) + val1.size(0) + test1.size(0)]
    loss, outputs = solver.test(torch.cat([train1, val1, test1], dim=0), torch.cat([train2, val2, test2], dim=0), apply_linear_cca)
    new_data = []
    # print(outputs)
    for idx in range(3):
        new_data.append([outputs[0][set_size[idx]:set_size[idx + 1], :], outputs[1][set_size[idx]:set_size[idx + 1], :], data1[idx][1]])
    # Training and testing of SVM with linear kernel on the view 1 with new features
    [test_acc, valid_acc] = svm_classify(new_data, C=0.01)
    print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
    print("Accuracy on view 1 (test data) is:", test_acc*100.0)
    # Saving new features in a gzip pickled file specified by save_to
    print('saving new features ...')
    f1 = gzip.open(save_to, 'wb')
    import pickle
    pickle.dump(new_data, f1)
    f1.close()
    d = torch.load('checkpoint.model')
    solver.model.load_state_dict(d)
    solver.model.parameters()

# download the data
# wget https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view1.gz
# wget https://www2.cs.uic.edu/~vnoroozi/noisy-mnist/noisymnist_view2.gz