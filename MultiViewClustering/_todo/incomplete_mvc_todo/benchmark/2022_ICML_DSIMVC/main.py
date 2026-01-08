import math, copy, faiss, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from utils import load_data, MultiviewDataset, RandomSampler, get_mask, evaluate
from torch.autograd import Variable

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
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_submodule = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, high_feature_dim))
        self.label_submodule = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        self.view = view

    def forward(self, xs):
        hs = []; qs = []; xrs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = F.normalize(self.feature_submodule(z), dim=1)
            q = self.label_submodule(z)
            xr = self.decoders[v](z)
            hs.append(h); qs.append(q); xrs.append(xr)
        return hs, qs, xrs

    def forward_mse(self, xs):
        xrs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            xrs.append(self.decoders[v](z))
        return xrs

    def forward_cluster(self, xs):
        qs = []; preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_submodule(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q); preds.append(pred)
        return qs, preds

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module): # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class WNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(WNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)

class MetaEncoder(MetaModule):
    def __init__(self, input_dim, feature_dim):
        super(MetaEncoder, self).__init__()
        self.encoder = nn.Sequential(MetaLinear(input_dim, 500), nn.ReLU(), MetaLinear(500, 500), nn.ReLU(), MetaLinear(500, 2000), nn.ReLU(), MetaLinear(2000, feature_dim))

    def forward(self, x):
        return self.encoder(x)

class MetaDecoder(MetaModule):
    def __init__(self, input_dim, feature_dim):
        super(MetaDecoder, self).__init__()
        self.decoder = nn.Sequential(MetaLinear(feature_dim, 2000), nn.ReLU(), MetaLinear(2000, 500), nn.ReLU(), MetaLinear(500, 500), nn.ReLU(), MetaLinear(500, input_dim))

    def forward(self, x):
        return self.decoder(x)

class SafeNetwork(MetaModule):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(SafeNetwork, self).__init__()
        self.encoders = []
        for v in range(view):
            self.encoders.append(MetaEncoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.feature_submodule = nn.Sequential(MetaLinear(feature_dim, feature_dim), nn.ReLU(), MetaLinear(feature_dim, high_feature_dim))
        self.label_submodule = nn.Sequential(MetaLinear(feature_dim, feature_dim), nn.ReLU(), MetaLinear(feature_dim, class_num), nn.Softmax(dim=1))
        self.view = view

    def forward(self, xs, xs_incomplete):
        qs = []; qs_incomplete = []; zs = []; zs_incomplete = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.feature_submodule(z)
            q = self.label_submodule(z)
            zs.append(h); qs.append(q)
            x_ = xs_incomplete[v]
            z_ = self.encoders[v](x_)
            h_ = self.feature_submodule(z_)
            q_ = self.label_submodule(z_)
            zs_incomplete.append(h_); qs_incomplete.append(q_)
        return zs, qs, zs_incomplete, qs_incomplete

    def forward_xs(self, xs):
        hs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            hs.append(self.feature_submodule(z))
        return hs, None, None

    def forward_s(self, xs):
        qs = []; zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = self.feature_submodule(z)
            q = self.label_submodule(z)
            zs.append(h); qs.append(q)
        return zs, qs

    def forward_cluster(self, xs):
        qs = []; preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_submodule(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q); preds.append(pred)
        return qs, preds

class Online(MetaModule):
    def __init__(self, view, input_size, feature_dim):
        super(Online, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(MetaEncoder(input_size[v], feature_dim))
            self.decoders.append(MetaDecoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.view = view

    def forward(self, xs):
        xrs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            xrs.append(self.decoders[v](z))
        return xrs

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, view, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device
        self.view = view
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

    def mask_correlated_samples2(self, N):
        m1 = torch.ones((N//2, N//2))
        m1 = m1.fill_diagonal_(0)
        m2 = torch.zeros((N//2, N//2))
        mask1 = torch.cat([m1, m2], dim=1)
        mask2 = torch.cat([m2, m1], dim=1)
        mask = torch.cat([mask1, mask2], dim=0)
        mask = mask.bool()
        return mask

    def mask_correlated_samples3(self, N):
        m1 = torch.ones((N//2, N//2))
        m1 = m1.fill_diagonal_(0)
        m2 = torch.zeros((N//2, N//2))
        mask1 = torch.cat([m2, m1], dim=1)
        mask2 = torch.cat([m1, m2], dim=1)
        mask = torch.cat([mask1, mask2], dim=0)
        mask = mask.bool()
        return mask

    def forward_feature(self, z1, z2, r=3.0):
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * z1.shape[0] / (z1.shape[0] - 1)
        return loss_part1 + loss_part2

    def forward_feature2(self, z1, z2, r=3.0):
        mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(r)).float().unsqueeze(1)
        z1 = mask1 * z1 + (1 - mask1) * F.normalize(z1, dim=1) * np.sqrt(r)
        z2 = mask2 * z2 + (1 - mask2) * F.normalize(z2, dim=1) * np.sqrt(r)
        loss_part1 = -2 * torch.sum(z1*z2, dim=1, keepdim=True)/z1.shape[0]
        square_term = torch.matmul(z1, z2.T) ** 2
        loss_part2 = torch.sum(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1), dim=1, keepdim=True) / (z1.shape[0] * (z1.shape[0] - 1))
        return loss_part1 + loss_part2

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j
        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)
        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0))
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples2(N)
        negative_clusters = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

def get_model():
    return SafeNetwork(view, dims, args.feature_dim, args.high_feature_dim, class_num).to(device)

def pretrain(com_dataset):
    print("Initializing network parameters...")
    pretrain_model = Online(view, dims, args.feature_dim).to(device)
    loader = DataLoader(com_dataset, batch_size=args.batch_size, shuffle=True)
    opti = torch.optim.Adam(pretrain_model.params(), lr=0.0003)
    criterion = torch.nn.MSELoss()
    for epoch in range(args.pretrain_epochs):
        for batch_idx, (xs, _, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            xrs = pretrain_model(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list)
            opti.zero_grad()
            loss.backward()
            opti.step()
    return pretrain_model.state_dict()

def bi_level_train(model, criterion, optimizer, class_num, view, com_loader, full_loader, mask, incomplete_ind):
    wnet_label = WNet(class_num, 100, 1).to(device)
    memory = Memory()
    memory.bi = True
    wnet_label.train()
    iteration = 0
    optimizer_wnet_label = torch.optim.Adam(wnet_label.params(), lr=args.lr_wnet)
    for com_batch, incomplete_batch in zip(com_loader, incomplete_loader):
        xs, _, _ = com_batch
        incomplete_xs, _, _ = incomplete_batch
        iteration += 1
        for v in range(view):
            xs[v] = xs[v].to(device)
            incomplete_xs[v] = incomplete_xs[v].to(device)
        model.train()
        meta_net = get_model()
        meta_net.load_state_dict(model.state_dict())
        com_hs, com_qs, incomplete_hs, incomplete_qs = meta_net(xs, incomplete_xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(criterion.forward_label(com_qs[v], com_qs[w]))
        loss_hat = sum(loss_list)
        cost_w_labels = []
        cost_w_features = []
        for v in range(view):
            for w in range(v+1, view):
                l_f, l_l = criterion.forward_feature2(incomplete_hs[v], incomplete_hs[w]), criterion.forward_label(incomplete_qs[v], incomplete_qs[w])
                cost_w_labels.append(l_l)
                cost_w_features.append(l_f)
        weight_label = wnet_label(sum(incomplete_qs)/view)
        norm_label = torch.sum(weight_label)
        for v in range(len(cost_w_labels)):
            if norm_label != 0:
                loss_hat += (torch.sum(cost_w_features[v] * weight_label)/norm_label + torch.sum(cost_w_labels[v]*weight_label) / norm_label)
            else:
                loss_hat += torch.sum(cost_w_labels[v] * weight_label + cost_w_features[v]*weight_label)
        meta_net.zero_grad()
        grads = torch.autograd.grad(loss_hat, (meta_net.params()), create_graph=True)
        meta_net.update_params(lr_inner=args.meta_lr, source_params=grads)
        del grads
        com_hs, com_qs, _, _ = meta_net(xs, incomplete_xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(criterion.forward_label(com_qs[v], com_qs[w]))
        l_g_meta = sum(loss_list)
        optimizer_wnet_label.zero_grad()
        l_g_meta.backward()
        optimizer_wnet_label.step()
        com_hs, com_qs, incomplete_hs, incomplete_qs = model(xs, incomplete_xs)
        loss_list = []
        for v in range(view):
            for w in range(v + 1, view):
                loss_list.append(criterion.forward_feature(com_hs[v], com_hs[w]))
                loss_list.append(criterion.forward_label(com_qs[v], com_qs[w]))
        loss = sum(loss_list)
        cost_w_labels = []
        cost_w_features = []
        for v in range(view):
            for w in range(v+1, view):
                l_f, l_l = criterion.forward_feature2(incomplete_hs[v], incomplete_hs[w]), criterion.forward_label(incomplete_qs[v], incomplete_qs[w])
                cost_w_labels.append(l_l)
                cost_w_features.append(l_f)
        with torch.no_grad():
            weight_label = wnet_label(sum(incomplete_qs)/view)
            norm_label = torch.sum(weight_label)
        for v in range(len(cost_w_labels)):
            if norm_label != 0:
                loss += (torch.sum(cost_w_labels[v] * weight_label)/norm_label + torch.sum(cost_w_features[v]*weight_label) / norm_label)
            else:
                loss += torch.sum(cost_w_labels[v] * weight_label + cost_w_features[v]*weight_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        memory.update_feature(model, full_loader, mask, incomplete_ind, iteration)
    acc, nmi, pur = valid(model, mask)
    return acc, nmi, pur

def valid(model, mask):
    pred_vec = []
    with torch.no_grad():
        input_data = []
        for v in range(view):
            data_v = torch.from_numpy(record_data_list[v]).to(device)
            input_data.append(data_v)
        output, _ = model.forward_cluster(input_data)
        for v in range(view):
            miss_ind = mask[:, v] == 0
            output[v][miss_ind] = 0
        sum_ind = np.sum(mask, axis=1, keepdims=True)
        output = sum(output)/torch.from_numpy(sum_ind).to(device)
        pred_vec.extend(output.detach().cpu().numpy())

    pred_vec = np.argmax(np.array(pred_vec), axis=1)
    acc, nmi, pur = evaluate(Y, pred_vec)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(acc, nmi, pur))
    return acc, nmi, pur

class Memory:
    def __init__(self):
        self.features = None
        self.alpha = args.alpha
        self.interval = args.interval
        self.bi = False

    def cal_cur_feature(self, model, loader):
        features = []
        for v in range(view):
            features.append([])
        for _, (xs, y, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            with torch.no_grad():
                if self.bi:
                    hs, _, _ = model.forward_xs(xs)
                else:
                    hs, _, _ = model(xs)
                for v in range(view):
                    fea = hs[v].detach().cpu().numpy()
                    features[v].extend(fea)
        for v in range(view):
            features[v] = np.array(features[v])
        return features

    def update_feature(self, model, loader, mask, incomplete_ind, epoch):
        topK = 600
        model.eval()
        cur_features = self.cal_cur_feature(model, loader)
        indices = []
        if epoch == 1:
            self.features = cur_features
            for v in range(view):
                fea = np.array(self.features[v])
                n, dim = fea.shape[0], fea.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(fea)
                _, ind = index.search(fea, topK + 1)  # Sample itself is included
                indices.append(ind[:, 1:])
            return indices
        elif epoch % self.interval == 0:
            for v in range(view):
                f_v = (1-self.alpha)*self.features[v] + self.alpha*cur_features[v]
                self.features[v] = f_v/np.linalg.norm(f_v, axis=1, keepdims=True)
                n, dim = self.features[v].shape[0], self.features[v].shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(self.features[v])
                _, ind = index.search(self.features[v], topK + 1)  # Sample itself is included
                indices.append(ind[:, 1:])
            if self.bi:
                make_imputation(mask, indices, incomplete_ind)
            return indices

def make_imputation(mask, indices, incomplete_ind):
    global data_list
    for v in range(view):
        for i in range(data_size):
            if mask[i, v] == 0:
                predicts = []
                for w in range(view):
                    # only the available views are selected as neighbors
                    if w != v and mask[i, w] != 0:
                        neigh_w = indices[w][i]
                        for n_w in range(neigh_w.shape[0]):
                            if mask[neigh_w[n_w], v] != 0 and mask[neigh_w[n_w], w] != 0:
                                predicts.append(data_list[v][neigh_w[n_w]])
                            if len(predicts) >= args.K:
                                break
                assert len(predicts) >= args.K
                fill_sample = np.mean(predicts, axis=0)
                data_list[v][i] = fill_sample
    global incomplete_loader
    incomplete_data = []
    for v in range(view):
        incomplete_data.append(data_list[v][incomplete_ind])
    incomplete_label = Y[incomplete_ind]
    incomplete_dataset = MultiviewDataset(view, incomplete_data, incomplete_label)
    incomplete_loader = DataLoader(incomplete_dataset, args.batch_size, drop_last=True, sampler=RandomSampler(len(incomplete_dataset), args.iterations * args.batch_size))

def initial(com_dataset, full_loader, criterion, mask, incomplete_ind):
    print("Initializing neighbors...")
    online_net = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num).to(device)
    loader = DataLoader(com_dataset, batch_size=256, shuffle=True, drop_last=True)
    mse_loader = DataLoader(com_dataset, batch_size=256, shuffle=True)
    opti = torch.optim.Adam(online_net.parameters(), lr=0.0003, weight_decay=0.)
    mse = torch.nn.MSELoss()
    memory = Memory()
    memory.interval = 1
    epochs = args.initial_epochs
    # pretraining on complete data
    for e in range(1, 201):
        for xs, _, _ in mse_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            xrs = online_net.forward_mse(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(mse(xrs[v], xs[v]))
            loss = sum(loss_list)
            opti.zero_grad()
            loss.backward()
            opti.step()
    for e in range(1, epochs+1):
        for xs, _, _ in loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            hs, qs, _ = online_net(xs)
            loss_list = []
            for v in range(view):
                for w in range(v+1, view):
                    loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                    loss_list.append(criterion.forward_label(qs[v], qs[w]))
            loss = sum(loss_list)
            opti.zero_grad()
            loss.backward()
            opti.step()
    # initial neighbors by the pretrain model
    indices = memory.update_feature(online_net, full_loader, mask, incomplete_ind, epoch=1)
    make_imputation(mask, indices, incomplete_ind)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--dataset', default='bdgp')
    parser.add_argument("--view", type=int, default=2)
    parser.add_argument("--feature_dim", default=512)
    parser.add_argument("--high_feature_dim", type=int, default=128)
    parser.add_argument('--lr_wnet', type=float, default=0.0004)
    parser.add_argument('--meta_lr', type=float, default=0.001)
    parser.add_argument("--epochs", default=120)
    parser.add_argument('--lr_decay_factor', type=float, default=0.2)
    parser.add_argument('--lr_decay_iter', type=int, default=20)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--initial_epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--miss_rate', default=0.1, type=float)
    parser.add_argument('--T', default=10, type=int)
    parser.add_argument('--iterations', default=200, type=int)
    args = parser.parse_args()
    data_list, Y, dims, total_view, data_size, class_num = load_data(args.dataset)
    view = total_view
    miss_rate = args.miss_rate
    incomplete_loader = None
    if args.dataset not in ['ccv']:
        for v in range(total_view):
            min_max_scaler = MinMaxScaler()
            data_list[v] = min_max_scaler.fit_transform(data_list[v])
    record_data_list = copy.deepcopy(data_list)
    if args.dataset == 'bdgp': args.initial_epochs = 30; args.pretrain_epochs = 100; args.iterations = 100;
    if args.dataset == 'mnist_usps': args.initial_epochs = 80; args.pretrain_epochs = 100; args.iterations = 200;
    if args.dataset == 'ccv': args.initial_epochs = 30; args.pretrain_epochs = 100; args.iterations = 300;
    if args.dataset == 'multi-fashion': args.initial_epochs = 100; args.pretrain_epochs = 200; args.iterations = 300;
    result_record = {"ACC": [], "NMI": [], "PUR": []}
    for t in range(1, args.T+1):
        print("--------Iter:{}--------".format(t))
        data_list = copy.deepcopy(record_data_list)
        mask = get_mask(view, data_size, miss_rate)
        sum_vec = np.sum(mask, axis=1, keepdims=True)
        complete_index = (sum_vec[:, 0]) == view
        mv_data = []
        for v in range(view):
            mv_data.append(data_list[v][complete_index])
        mv_label = Y[complete_index]
        com_dataset = MultiviewDataset(view, mv_data, mv_label)
        com_loader = DataLoader(com_dataset, args.batch_size, drop_last=True, sampler=RandomSampler(len(com_dataset), args.iterations * args.batch_size))
        full_dataset = MultiviewDataset(view, data_list, Y)
        full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
        incomplete_ind = (sum_vec[:, 0]) != view
        model = get_model()
        state_dict = pretrain(com_dataset)
        model.load_state_dict(state_dict, strict=False)
        optimizer = torch.optim.Adam(model.params(), lr=0.0003, weight_decay=0.)
        criterion = Loss(args.batch_size, class_num, view, device)
        initial(com_dataset, full_loader, criterion, mask, incomplete_ind)
        acc, nmi, pur = bi_level_train(model, criterion, optimizer, class_num, view, com_loader, full_loader, mask, incomplete_ind)
        result_record["ACC"].append(acc)
        result_record["NMI"].append(nmi)
        result_record["PUR"].append(pur)
    print("ACC (mean) = {:.4f} ACC (std) = {:.4f}".format(np.mean(result_record["ACC"]), np.std(result_record["ACC"])))
    print("NMI (mean) = {:.4f} NMI (std) = {:.4f}".format(np.mean(result_record["NMI"]), np.std(result_record["NMI"])))
    print("PUR (mean) = {:.4f} PUR (std) = {:.4f}".format(np.mean(result_record["PUR"]), np.std(result_record["PUR"])))
