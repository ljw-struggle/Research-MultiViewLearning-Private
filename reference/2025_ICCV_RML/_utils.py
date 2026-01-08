import random, numpy as np, scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.testing import assert_array_almost_equal
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, contingency_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = clustering_acc(label, pred)
    pur = purity_score(label, pred)
    # asw = silhouette_score(embedding, pred) # silhouette score
    return nmi, ari, acc, pur #, asw

def clustering_acc(y_true, y_pred): # y_pred and y_true are numpy arrays, same shape
    y_true = y_true.astype(np.int64); y_pred = y_pred.astype(np.int64); assert y_pred.size == y_true.size; 
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.array([[sum((y_pred == i) & (y_true == j)) for j in range(D)] for i in range(D)], dtype=np.int64) # shape: (num_pred_clusters, num_true_clusters)
    ind = linear_sum_assignment(w.max() - w) # align clusters using the Hungarian algorithm, ind[0] is the row indices (predicted clusters), ind[1] is the column indices (true clusters)
    return sum([w[i][j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.shape[0] # accuracy

def purity_score(y_true, y_pred):
    contingency_matrix_result = contingency_matrix(y_true, y_pred) # shape: (num_true_clusters, num_pred_clusters)
    return np.sum(np.amax(contingency_matrix_result, axis=0)) / np.sum(contingency_matrix_result) 


def load_data(dataset, trainset_rate=0.7, type='train', seed=0, mode='RML', noise_rate=0.5):
    dataset = MyDataset('./data/', trainset_rate=trainset_rate, type=type, seed=seed, dataname=dataset, mode=mode, noise_rate=noise_rate)
    dims, dimss = dataset.__dims__()
    view = len(dims)
    data_size = dataset.__len__()
    class_num = dataset.__classnum__()
    print('N:'+str(data_size), 'K:'+str(class_num), 'M:'+str(view), 'D:'+str(dims))
    return dataset, dims, view, data_size, class_num, dimss

class MyDataset(Dataset):
    def __init__(self, path, trainset_rate=0.7, type='train', seed=0, dataname='xxx', mode='RML', noise_rate=0.5):
        scaler = MinMaxScaler()
        if dataname == 'WebKB':
            self.V1 = sio.loadmat(path + 'WebKB')['X'][0][0].astype(np.float32)
            self.V2 = sio.loadmat(path + 'WebKB')['X'][0][1].astype(np.float32)
            self.num = self.V1.shape[0]
            self.view_dims = [self.V1.shape[1], self.V2.shape[1]]
            self.Data = [self.V1, self.V2]
            self.Y = sio.loadmat(path + 'WebKB')['gnd'].astype(np.int32).reshape(self.num,) - 1
        if dataname == 'DHA':
            self.V1 = (sio.loadmat(path + 'DHA.mat')['X1'].astype(np.float32))
            self.V2 = (sio.loadmat(path + 'DHA.mat')['X2'].astype(np.float32))
            self.num = self.V1.shape[0]
            self.view_dims = [self.V1.shape[1], self.V2.shape[1]]
            self.Data = [self.V1, self.V2]
            self.Y = sio.loadmat(path + 'DHA.mat')['Y'].astype(np.int32).reshape(self.num, )
        if dataname == 'BDGP':
            data1 = sio.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
            data2 = sio.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
            labels = sio.loadmat(path + 'BDGP.mat')['Y'][0]
            v1 = scale_normalize_matrix(data1)
            v2 = scale_normalize_matrix(data2)
            self.num = v1.shape[0]
            self.view_dims = [v1.shape[1], v2.shape[1]]
            self.Data = [v1, v2]
            self.Y = labels
        if dataname == 'NGs':
            self.Y = sio.loadmat(path + 'NGs')['truelabel'][0][0].astype(np.int32).reshape(500, )
            self.V1 = sio.loadmat(path + 'NGs')['data'][0][0].astype(np.float32)
            self.V2 = sio.loadmat(path + 'NGs')['data'][0][1].astype(np.float32)
            self.V3 = sio.loadmat(path + 'NGs')['data'][0][2].astype(np.float32)
            self.v1 = np.transpose(self.V1)
            self.v2 = np.transpose(self.V2)
            self.v3 = np.transpose(self.V3)
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]
        if dataname == 'VOC':
            self.Y = sio.loadmat(path + 'VOC')['Y'].astype(np.int32).reshape(5649, )
            self.V1 = sio.loadmat(path + 'VOC')['X1'].astype(np.float32)
            self.V2 = sio.loadmat(path + 'VOC')['X2'].astype(np.float32)
            self.num = self.V1.shape[0]
            self.view_dims = [self.V1.shape[1], self.V2.shape[1]]
            self.Data = [self.V1, self.V2]
        if dataname == 'Cora':
            mat = sio.loadmat("./data/Cora.mat")
            self.v1 = (mat['coracites'].astype('float32'))
            self.v2 = (mat['coracontent'].astype('float32'))
            # self.v3 = (mat['corainbound'].astype('float32'))
            # self.v4 = (mat['coraoutbound'].astype('float32'))
            self.Y = np.squeeze(mat['y']).astype('int') - 1
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1]]
            self.Data = [self.v1, self.v2]
        if dataname == 'YoutubeVideo':
            mat = sio.loadmat("./data/Video-3V.mat")
            self.v1 = mat['X1'].astype('float32')
            self.v2 = mat['X2'].astype('float32')
            self.v3 = mat['X3'].astype('float32')
            self.Y = np.squeeze(mat['Y']).astype('int') - 1  # cleaning labels to [0, 1, 2 ... K-1] for visualization
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]
        if dataname == 'Prokaryotic':
            mat = sio.loadmat("./data/Prokaryotic.mat")
            self.v1 = (mat['gene_repert'].astype('float32'))
            self.v2 = scaler.fit_transform(mat['proteome_comp'].astype('float32'))
            self.v3 = scaler.fit_transform(mat['text'].astype('float32'))
            self.Y = np.squeeze(mat['Y']).astype('int') - 1
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]
        if dataname == 'Cifar100':
            mat = sio.loadmat("./data/cifar100.mat")
            self.v1 = mat['data'][0][0].T.astype('float32')
            self.v2 = mat['data'][1][0].T.astype('float32')
            self.v3 = mat['data'][2][0].T.astype('float32')
            self.Y = np.squeeze(mat['truelabel'][0][0].T[0]).astype('int')
            self.num = self.v1.shape[0]
            self.view_dims = [self.v1.shape[1], self.v2.shape[1], self.v3.shape[1]]
            self.Data = [self.v1, self.v2, self.v3]
        self.view_num = len(self.Data)
        index = list(range(0, self.num, 1))
        random.seed(seed); random.shuffle(index)
        for v in range(self.view_num):
            self.Data[v] = self.Data[v][index]
        self.Y = self.Y[index]
        self.classnum = np.unique(self.Y)
        if type == 'train':
            print("Training set...")
            self.num = int(trainset_rate*self.num)
            for v in range(self.view_num):
                self.Data[v] = self.Data[v][:self.num, :]
                print(self.Data[v].shape)
            self.Y = self.Y[:self.num]
            print(self.Y.shape)
            print(self.Y[0:10])
            if mode == 'RML_LCE':
                Ys = np.asarray([[self.Y[i]] for i in range(self.num)])
                if noise_rate > 0:
                    self.train_noisy_labels, _ = noisify(nb_classes=len(self.classnum), train_labels=Ys, noise_type='symmetric', noise_rate=noise_rate, random_state=0)
                else: self.train_noisy_labels = Ys
                print("Noise label rate: " + str(noise_rate))
                self.Y = np.array(self.train_noisy_labels.T[0])
        if type == 'test':
            print("Test set...")
            print(self.Y[0:10])
            self.num = int(trainset_rate*self.num)
            for v in range(self.view_num):
                self.Data[v] = self.Data[v][self.num:, :]
                print(self.Data[v].shape)
            self.Y = self.Y[self.num:]
            self.num = len(self.Y)
            print(self.Y.shape)

    def __len__(self):
        return self.num

    def __dims__(self):
        return self.view_dims, sum(self.view_dims)

    def __classnum__(self):
        return len(self.classnum)

    def __getitem__(self, idx):
        returndata = []
        for v in range(self.view_num):
            returndata.append(torch.from_numpy(self.Data[v][idx]))
        return returndata, self.Y[idx], torch.from_numpy(np.array(idx)).long()

def build_for_cifar100(size, noise):
    """ The noise matrix flips to the "next" class with probability 'noise'."""
    assert(noise >= 0.) and (noise <= 1.)
    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise
    P[size-1, 0] = noise
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T. It expects a number between 0 and the number of classes - 1."""
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]
    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()
    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]
    return new_y

def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes: flip in the pair"""
    P = np.eye(nb_classes); n = noise;
    if n > 0.0:
        P[0, 0], P[0, 1] = 1. - n, n # 0 -> 1
        for i in range(1, nb_classes-1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    return y_train, actual_noise

def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes: flip in the symmetric way"""
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P
    if n > 0.0:
        P[0, 0] = 1. - n # 0 -> 1
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy
    return y_train, actual_noise

def noisify_mnist_asymmetric(y_train, noise, random_state=None):
    nb_classes = 10; P = np.eye(nb_classes); n = noise;
    if n > 0.0:
        P[7, 7], P[7, 1] = 1. - n, n # 1 <- 7
        P[2, 2], P[2, 7] = 1. - n, n # 2 -> 7
        P[5, 5], P[5, 6] = 1. - n, n # 5 <-> 6
        P[6, 6], P[6, 5] = 1. - n, n # 5 <-> 6
        P[3, 3], P[3, 8] = 1. - n, n # 3 -> 8
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy
    return y_train, P

def noisify_cifar10_asymmetric(y_train, noise, random_state=None):
    nb_classes = 10; P = np.eye(nb_classes); n = noise;
    if n > 0.0:
        P[9, 9], P[9, 1] = 1. - n, n # automobile <- truck
        P[2, 2], P[2, 0] = 1. - n, n # bird -> airplane
        P[3, 3], P[3, 5] = 1. - n, n # cat <-> dog
        P[5, 5], P[5, 3] = 1. - n, n # cat <-> dog  
        P[4, 4], P[4, 7] = 1. - n, n # automobile -> truck
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    return y_train, P

def noisify_cifar100_asymmetric(y_train, noise, random_state=None):
    """ mistakes are inside the same superclass of 10 classes, e.g. 'fish' """
    nb_classes = 100; P = np.eye(nb_classes); n = noise; nb_superclasses = 20; nb_subclasses = 5;
    if n > 0.0:
        for i in np.arange(nb_superclasses):
            init, end = i * nb_subclasses, (i+1) * nb_subclasses
            P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)
        y_train_noisy = multiclass_noisify(y_train, P=P, random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    return y_train, P

def noisify(dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    if noise_type == 'symmetric':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    if noise_type == 'asymmetric':
        if dataset == 'mnist':
            train_noisy_labels, actual_noise_rate = noisify_mnist_asymmetric(train_labels, noise_rate, random_state=random_state)
        elif dataset == 'cifar10':
            train_noisy_labels, actual_noise_rate = noisify_cifar10_asymmetric(train_labels, noise_rate, random_state=random_state)
        elif dataset == 'cifar100':
            train_noisy_labels, actual_noise_rate = noisify_cifar100_asymmetric(train_labels, noise_rate, random_state=random_state)
    return train_noisy_labels, actual_noise_rate
    
def classification_evaluation(y_true, y_predict):
    accuracy=classification_report(y_true, y_predict, output_dict=True, zero_division=1)['accuracy']
    s=classification_report(y_true, y_predict, output_dict=True, zero_division=1)['weighted avg']
    precision=s['precision']
    recall=s['recall']
    f1_score=s['f1-score']
    return accuracy, precision, recall, f1_score

def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min()
    max_val = input_matrix.max()
    input_range = max_val - min_val
    scaled_matrix = (input_matrix - min_val) / input_range * (max_value - min_value) + min_value
    return scaled_matrix

def valid(model, device, dataset, view, data_size, class_num, eval_q=False, eval_z=False, eval_x=False):
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    def inference(loader, model, device, view, classification=False, label=0):
        model.eval()
        for step, (xs, y, _) in enumerate(loader):
            xs = [x.to(device) for x in xs]
            with torch.no_grad():
                h, z, q, scores, hs = model.forward(xs)
            z = z.cpu().detach().numpy()
            h = h.cpu().detach().numpy()
            q = q.cpu().detach().numpy()
        q = q.argmax(1); y = y.numpy(); y = y.flatten()
        return y, h, z, q, hs, xs
    labels_vector, h, z, q, hs, xs = inference(test_loader, model, device, view, classification=False, label=0)
    if eval_x == True:
        metric1 = []; metric2 = []; metric3 = []; metric4 = []
        for v in range(len(xs)):
            xs[v] = xs[v].cpu().detach().numpy()
        for i in range(5):
            kmeans = KMeans(n_clusters=class_num)
            x_pred = kmeans.fit_predict(np.concatenate(xs, axis=1))
            nmi_x, ari_x, acc_x, pur_x = evaluate(labels_vector, x_pred)
            print('ACC_x = {:.4f} NMI_x = {:.4f} ARI_x = {:.4f} PUR_x = {:.4f}'.format(acc_x, nmi_x, ari_x, pur_x))
            metric1.append(acc_x); metric2.append(nmi_x); metric3.append(ari_x); metric4.append(pur_x)
        print('%.3f' % np.mean(metric1), '± %.3f' % np.std(metric1), metric1)
        print('%.3f' % np.mean(metric2), '± %.3f' % np.std(metric2), metric2)
        print('%.3f' % np.mean(metric3), '± %.3f' % np.std(metric3), metric3); print('%.3f' % np.mean(metric4), '± %.3f' % np.std(metric4), metric4)
    if eval_z == True:
        kmeans = KMeans(n_clusters=class_num)
        for v in range(len(hs)):
            hs[v] = hs[v].cpu().detach().numpy()
            z_pred = kmeans.fit_predict(hs[v])
            nmi_z, ari_z, acc_z, pur_z = evaluate(labels_vector, z_pred)
            print('ACC_v = {:.4f} NMI_v = {:.4f} ARI_v = {:.4f} PUR_v = {:.4f}'.format(acc_z, nmi_z, ari_z, pur_z))
        z_pred = kmeans.fit_predict(z)
        nmi_z, ari_z, acc_z, pur_z = evaluate(labels_vector, z_pred)
        print('ACC_z = {:.4f} NMI_z = {:.4f} ARI_z = {:.4f} PUR_z = {:.4f}'.format(acc_z, nmi_z, ari_z, pur_z))
        return acc_z, nmi_z, ari_z, pur_z
    if eval_q == True:
        accuracy, precision, recall, f1_score = classification_evaluation(labels_vector, q)
        print('ACC_q = {:.4f} Precision_q = {:.4f} F1-score_q = {:.4f} Recall_q = {:.4f}'.format(accuracy, precision, f1_score, recall))
        return accuracy, precision, f1_score, recall
