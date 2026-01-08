import random, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from utils import load_data, evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, 2000), nn.ReLU(), nn.Linear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 2000), nn.ReLU(), nn.Linear(2000, 500), nn.ReLU(), nn.Linear(500, 500), nn.ReLU(), nn.Linear(500, input_dim))

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, class_num, device, batch_size):
        super(Network, self).__init__()
        self.avg = (torch.ones((batch_size, batch_size)) * (1 / batch_size)).to(device)
        self.batch_size = batch_size
        self.device = device
        self.class_num = class_num
        self.anchor_num = class_num
        self.view = view
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, self.anchor_num), nn.BatchNorm1d(self.anchor_num))
        self.C = nn.Parameter(torch.randn(self.anchor_num, feature_dim))

    def forward(self, xs, max_view=-1):
        hs = []; zs_pre_align = []; hs_align = []; xrs = []; zs = []; zs_pre = []
        size_x = len(xs[0])
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = F.normalize(self.feature_contrastive_module(z), dim=1)
            h = self.activate_and_normalize(h)
            z_pre_align = 0
            h_align = 0
            z_pre = torch.mm(h, self.C)
            if max_view >= 0:
                z_pre_t = F.normalize(z_pre, dim=1)
                z_pre_t = self.activate_and_normalize(z_pre_t)
                p = (torch.ones((size_x, size_x)) * (1 / size_x)).to(self.device)
                if v != max_view:
                    z_pre_align = torch.mm(p, z_pre_t)
                    h_align = torch.mm(p, h)
                else:
                    z_pre_align = z_pre_t
                    h_align = h
            xr = self.decoders[v](z_pre)
            hs.append(h)
            zs_pre_align.append(z_pre_align)
            hs_align.append(h_align)
            zs.append(z)
            xrs.append(xr)
            zs_pre.append(z_pre)
        return hs, xrs, zs, zs_pre, zs_pre_align, hs_align

    def activate_and_normalize(self, tensor):
        tensor = torch.clamp(tensor, min=0)
        row_sums = tensor.sum(dim=1, keepdim=True)
        row_sums = row_sums + (row_sums == 0).float()
        tensor = tensor / row_sums
        return tensor

    def get_C(self):
        return self.C.clone()

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device
        self.index_dif = torch.combinations(torch.arange(self.class_num))
        self.mask_C = self.remove_index()
        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def remove_index(self):
        N = self.index_dif.size(0)
        M = self.class_num
        mask = torch.ones((N, M), dtype=bool)
        for i in range(N):
            mask[i, self.index_dif[i]] = False
        return mask

    def prototype_dif2(self, C):
        C = F.normalize(C)
        sim = torch.mm(C, C.t())
        f = lambda x: torch.exp(x / self.temperature_l)
        sim = f(sim)
        B1 = sim[self.index_dif[:, 0]]
        B2 = sim[self.index_dif[:, 1]]
        B = B1 - B2
        B = B[self.mask_C].view(B.size(0), -1)
        f2 = lambda x: torch.exp(-x / self.temperature_l)
        loss_sum = torch.sum(f2(B ** 2))/(self.index_dif.size(0) * self.class_num)
        return loss_sum

    def forward_feature(self, h_i, h_j):
        h_i = self.activate_and_normalize(h_i)
        h_j = self.activate_and_normalize(h_j)
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def activate_and_normalize(self, tensor):
        tensor = torch.clamp(tensor, min=0)
        row_sums = tensor.sum(dim=1, keepdim=True)
        row_sums = row_sums + (row_sums == 0).float()
        tensor = tensor / row_sums
        return tensor
    
class GN_Dataset(Dataset):
    def __init__(self, dataset, y, view_num):
        self.dataset = dataset
        self.labels = y
        self.view_num = view_num

    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):
        X = []; Y = []
        for i in range(self.view_num):
            data_tensor = torch.from_numpy(self.dataset[i][index].astype(np.float32).transpose())
            label = self.labels[i][index]; label_tensor = torch.tensor(label)
            X.append(data_tensor); Y.append(label_tensor)
        return X, Y, index
    
def Form_Unaligned_Data(dataset, view):
    X = []; Y = []
    for i in range(view):
        X.append(dataset.get_view(i))
        Y.append(dataset.get_view(-1))
    size = len(Y[0])
    view_num = len(X)
    t = np.linspace(0, size - 1, size, dtype=int)
    random.shuffle(t)
    Xtmp = []; Ytmp = []
    for i in range(view_num):
        xtmp = np.copy(X[i])
        Xtmp.append(xtmp)
        ytmp = np.copy(Y[i])
        Ytmp.append(ytmp)
    ts = []
    for v in range(view_num):
        random.shuffle(t)
        ts.append(t)
        Xtmp[v][:] = X[v][t]
        Ytmp[v][:] = Y[v][t]
    X = Xtmp; Y = Ytmp
    result = GN_Dataset(X, Y, view)
    return result, ts

def valid(model, device, dataset, view, data_size, class_num, max_view, epoch, ts):
    def inference(loader, model, device, view, data_size, class_num, max_view):
        Hs = []; Zs = []; labels_vector_multi = []
        for v in range(view):
            Hs.append([]); Zs.append([]); labels_vector_multi.append([])
        labels_vector = []
        for step, (xs, y, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            with torch.no_grad():
                hs, _, zs, zs_pre, zs_pre_align, hs_align = model.forward(xs, max_view)
            for v in range(view):
                hs[v] = hs[v].detach()
                zs[v] = zs_pre_align[v].detach()
                Hs[v].extend(hs[v].cpu().detach().numpy())
                Zs[v].extend(zs[v].cpu().detach().numpy())
                labels_vector_multi[v].extend(y[v].numpy())
            labels_vector.extend(y[max_view].numpy())
        labels_vector = np.array(labels_vector).reshape(data_size)
        H_avg = np.array(Hs[max_view])
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        total_pred_h = kmeans.fit_predict(H_avg)
        return total_pred_h, labels_vector, H_avg
    test_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    total_pred_h, labels_vector, H_avg = inference(test_loader, model, device, view, data_size, class_num, max_view)
    print("Clustering results on H: " + str(labels_vector.shape[0]))
    nmi, ari, acc, pur = evaluate(labels_vector, total_pred_h)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur, ari

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default="Caltech7")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--lamda2", default=0.1)
    parser.add_argument("--lamda3", default=1.0)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--temperature_l", default=0.4)
    parser.add_argument("--epochs", default=50)
    parser.add_argument("--feature_dim", default=512)
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")
    for iter in range(0, 1):
        print("ROUND:", iter, "dataset: ", args.dataset)
        seed = 10
        def setup_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
        setup_seed(seed)
        dataset, dims, view, data_size, class_num = load_data(args.dataset)
        dataset, ts = Form_Unaligned_Data(dataset, view)
        max_value = max(dims)
        max_view = max(i for i, v in enumerate(dims) if v == max_value)
        print("max_view:", max_view)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        accs = []; nmis = []; purs = []
        T = 1; t_max = 0; t_epoch = 0
        for i in range(T):
            setup_seed(seed)
            model = Network(view, dims, args.feature_dim, class_num, device, args.batch_size)
            model = model.to(device)
            optimizer = torch.optim.Adam([
                {'params':[model.C, *model.feature_contrastive_module.parameters()], 'lr':args.learning_rate * 1},
                {'params':(list(model.encoders.parameters()) + list(model.decoders.parameters())), 'lr': args.learning_rate * 1}
            ])
            criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)
            epoch = 1
            while epoch <= args.epochs:
                tot_loss = 0.
                mse = torch.nn.MSELoss()
                for batch_idx, (xs, _, _) in enumerate(data_loader):
                    for v in range(view):
                        xs[v] = xs[v].to(device)
                    optimizer.zero_grad()
                    hs, xrs, zs, zs_pre, zs_pre_align, hs_align = model(xs, max_view)
                    loss_list = []
                    t1 = hs_align[max_view]
                    for v in range(view):
                        if v != max_view:
                            t2 = hs_align[v]
                            loss_list.append(args.lamda3 * criterion.forward_feature(t1, t2))
                        loss_list.append(mse(xs[v], xrs[v]))
                    C = model.get_C()
                    loss_list.append(args.lamda2 * criterion.prototype_dif2(C))
                    loss = sum(loss_list)
                    loss.backward()
                    optimizer.step()
                    tot_loss += loss.item()
                if (epoch % 5) == 0:
                    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
                if epoch == args.epochs:
                # if epoch % 10 == 0:
                    acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, max_view, epoch, ts)
                epoch += 1
