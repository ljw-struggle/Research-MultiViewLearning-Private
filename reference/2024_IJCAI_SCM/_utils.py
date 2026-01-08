from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch

def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min()
    max_val = input_matrix.max()
    input_range = max_val - min_val
    scaled_matrix = (input_matrix - min_val) / input_range * (max_value - min_value) + min_value
    return scaled_matrix
class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = scale_normalize_matrix(data1)
        self.x2 = scale_normalize_matrix(data2)
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class DHA(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'DHA.mat')['Y'].astype(np.int32).reshape(483,)
        self.V1 = scipy.io.loadmat(path + 'DHA.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'DHA.mat')['X2'].astype(np.float32)
    def __len__(self):
        return 483
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]

        x1 = scale_normalize_matrix(x1)
        x2 = scale_normalize_matrix(x2)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class WebKB(Dataset):
    def __init__(self,path):
        self.Y = scipy.io.loadmat(path + 'WebKB')['gnd'].astype(np.int32).reshape(1051,)
        self.V1 = scipy.io.loadmat(path + 'WebKB')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'WebKB')['X'][0][1].astype(np.float32)
    def __len__(self):
        return 1051
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]

        return[torch.from_numpy(x1),torch.from_numpy(x2)],self.Y[idx],torch.from_numpy(np.array(idx)).long()
class NGs(Dataset):
    def __init__(self,path):
        self.Y = scipy.io.loadmat(path + 'NGs')['truelabel'][0][0].astype(np.int32).reshape(500,)
        self.V1 = scipy.io.loadmat(path + 'NGs')['data'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'NGs')['data'][0][1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'NGs')['data'][0][2].astype(np.float32)

        self.V1 = np.transpose(self.V1)
        self.V2 = np.transpose(self.V2)
        self.V3 = np.transpose(self.V3)

        self.v1 = scale_normalize_matrix(self.V1)
        self.v2 = scale_normalize_matrix(self.V2)
        self.v3 = scale_normalize_matrix(self.V3)

    def __len__(self):
        return 500
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()
class VOC(Dataset):
    def __init__(self,path):
        self.Y = scipy.io.loadmat(path + 'VOC')['Y'].astype(np.int32).reshape(5649,)
        self.V1 = scipy.io.loadmat(path + 'VOC')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'VOC')['X2'].astype(np.float32)
    def __len__(self):
        return 5649
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Fc_COIL_20(Dataset):
    def __init__(self,path):
        self.Y = scipy.io.loadmat(path + 'Fc_COIL_20')['Y'].astype(np.int32).reshape(1440, )
        self.V1 = scipy.io.loadmat(path + 'Fc_COIL_20')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fc_COIL_20')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fc_COIL_20')['X3'].astype(np.float32)
    def __len__(self):
        return 1440
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], \
            self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        dimss = 1829
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        dimss = 1568
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        dimss = 2352
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "DHA":
        dataset = DHA('./data/')
        dims = [110, 6144]
        dimss = 6254
        view = 2
        data_size = 483
        class_num = 23
    elif dataset == "WebKB":
        dataset = WebKB('./data/')
        dims = [2949, 334]
        dimss = 3283
        view = 2
        data_size = 1051
        class_num = 2
    elif dataset == "NGs":
        dataset = NGs('./data/')
        dims = [2000, 2000 , 2000]
        dimss = 6000
        view = 3
        data_size = 500
        class_num = 5
    elif dataset == "VOC":
        dataset = VOC('./data/')
        dims = [512, 399]
        dimss = 911
        view = 2
        data_size = 5649
        class_num = 20
    elif dataset == "Fc_COIL_20":
        dataset = Fc_COIL_20('./data/')
        dims = [1024, 1024, 1024]
        dimss = 3072
        view = 3
        data_size = 1440
        class_num = 20
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num, dimss


from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

def scale_normalize_matrix(input_matrix, min_value=0, max_value=1):
    min_val = input_matrix.min()
    max_val = input_matrix.max()
    input_range = max_val - min_val
    scaled_matrix = (input_matrix - min_val) / input_range * (max_value - min_value) + min_value
    return scaled_matrix

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def inference(loader, model, device, view):
    model.eval()
    soft_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        xs_all = torch.cat(xs, dim=1)
        with torch.no_grad():
            _, h, z, q = model.forward(xs_all)
        z = z.cpu().detach().numpy()
        h = h.cpu().detach().numpy()
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
    total_pred = np.argmax(np.array(soft_vector), axis=1)

    y = y.numpy()
    y = y.flatten()
    return y, h, z, total_pred
def valid(model, device, dataset, view, data_size, class_num, eval_q = False,eval_z = False):
    test_loader = DataLoader(
            dataset,
            batch_size=data_size,
            shuffle=False,
        )
    labels_vector, h, z, q = inference(test_loader, model, device, view)
    kmeans = KMeans(n_clusters=class_num)
    print(str(len(labels_vector)) + " samples")
    if eval_q == True:
        nmi_q, ari_q, acc_q, pur_q = evaluate(labels_vector, q)
        print('ACC_q = {:.4f} NMI_q = {:.4f} ARI_q = {:.4f} PUR_q = {:.4f}'.format(acc_q, nmi_q, ari_q, pur_q))
        return acc_q, nmi_q, ari_q, pur_q
    if eval_z == True:
        z_pred = kmeans.fit_predict(z)
        nmi_z, ari_z, acc_z, pur_z = evaluate(labels_vector, z_pred)
        print('ACC_z = {:.4f} NMI_z = {:.4f} ARI_z = {:.4f} PUR_z = {:.4f}'.format(acc_z, nmi_z, ari_z, pur_z))
        return acc_z, nmi_z, ari_z, pur_z


