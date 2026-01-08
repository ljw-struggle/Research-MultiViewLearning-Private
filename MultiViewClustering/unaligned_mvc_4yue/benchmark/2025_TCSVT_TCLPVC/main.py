import math, argparse
from utils import set_seed, getMvKNNGraph, get_dataset, MvDataset, valid
from configure import get_default_config
from alignment import get_alignment
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim), nn.Softmax(dim=1))
        self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        self.view = view

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = F.normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        # self.mask = self.mask_correlated_samples(batch_size)
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

    def forward_feature(self, h_i, h_j):
        self.batch_size = h_i.shape[0]
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

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

    def graph_loss(self, sub_graph, sub_x, all_x):
        if len(sub_graph.shape) == 2:

            diag0_graph0 = torch.diag(sub_graph.sum(1))  # m*m for a m*n matrix
            diag1_graph0 = torch.diag(sub_graph.sum(0))  # n*n for a m*n matrix
            graph_loss = torch.trace(sub_x.t().mm(diag0_graph0).mm(sub_x)) + torch.trace(
                all_x.t().mm(diag1_graph0).mm(all_x)) - 2 * torch.trace(sub_x.t().mm(sub_graph).mm(all_x))
            return graph_loss / (sub_graph.shape[0] * sub_graph.shape[1])
        else:
            graphs_loss = 0
            for v, graph in enumerate(sub_graph):
                diag0_graph0 = torch.diag(graph.sum(1))  # m*m for a m*n matrix
                diag1_graph0 = torch.diag(graph.sum(0))  # n*n for a m*n matrix
                graphs_loss += torch.trace(sub_x[v].t().mm(diag0_graph0).mm(sub_x[v])) + torch.trace(
                    all_x[v].t().mm(diag1_graph0).mm(all_x[v])) - 2 * torch.trace(sub_x[v].t().mm(graph).mm(all_x[v]))
            return graphs_loss / (sub_graph.shape[0] * sub_graph.shape[1] * sub_graph.shape[2])

class RunModule:
    def __init__(self, config, device):
        self.cfg = config
        self.device = device
        self.dataset_aligned, self.dataset_shuffle, self.aligned_idx = get_dataset(self.cfg, self.device)
        self.model_path = "./pretrain/" + self.cfg['Dataset']["name"]

    def pretrain_ae(self):
        num_views = self.cfg['Dataset']['num_views']
        epochs = self.cfg['training']['mse_epochs']
        lambda_graph = self.cfg['training']['lambda_graph']
        model = Network(num_views,
                        self.cfg['Module']['in_dim'],
                        self.cfg['Module']['feature_dim'],
                        self.cfg['Module']['high_feature_dim'],
                        self.cfg['Dataset']['num_classes'], self.device).to(self.device)
        data_loader = torch.utils.data.DataLoader(
            self.dataset_shuffle,
            batch_size=self.cfg['Dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.cfg['training']['lr'],
                                     weight_decay=self.cfg['training']['weight_decay'])
        criterion = Loss(self.cfg['Dataset']['batch_size'],
                         self.cfg['Dataset']['num_classes'],
                         self.cfg['training']['temperature_f'],
                         self.cfg['training']['temperature_l'],
                         self.device).to(self.device)
        all_new_x = [torch.from_numpy(v_data) for v_data in self.dataset_shuffle.fea]
        all_new_z = torch.ones((self.cfg['Dataset']['num_views'], self.cfg['Dataset']['num_sample'],
                                self.cfg['Module']['feature_dim'])).to(self.device)
        all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device,
                                 dtype=torch.float32)
        for epoch in range(epochs):
            for xs, _, _, idx in data_loader:
                _, _, xrs, zs = model(xs)
                loss_list = []
                for v in range(num_views):
                    loss_list.append(torch.nn.MSELoss()(xs[v], xrs[v]))
                loss = sum(loss_list)
                if epoch > 0:
                    loss = loss + lambda_graph * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for v in range(num_views):
                    all_new_z[v][idx] = zs[v].detach().clone()
        torch.save({'model': model.state_dict()}, self.model_path + '_ae.pth')

    def pretrain_cl(self):
        num_views = self.cfg['Dataset']['num_views']
        num_classes = self.cfg['Dataset']['num_classes']
        epochs = self.cfg['training']['con_epochs']
        lambda_graph = self.cfg['training']['lambda_graph']
        model = Network(num_views,
                        self.cfg['Module']['in_dim'],
                        self.cfg['Module']['feature_dim'],
                        self.cfg['Module']['high_feature_dim'],
                        num_classes, self.device).to(self.device)
        model.load_state_dict(torch.load(self.model_path + '_ae.pth')['model'])
        data_loader = torch.utils.data.DataLoader(
            self.dataset_shuffle,
            batch_size=self.cfg['Dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.cfg['training']['lr'],
                                     weight_decay=self.cfg['training']['weight_decay'])
        criterion = Loss(self.cfg['Dataset']['batch_size'],
                         self.cfg['Dataset']['num_classes'],
                         self.cfg['training']['temperature_f'],
                         self.cfg['training']['temperature_l'],
                         self.device).to(self.device)
        # training
        all_new_x = [torch.from_numpy(v_data) for v_data in self.dataset_shuffle.fea]
        all_new_z = torch.ones((self.cfg['Dataset']['num_views'], self.cfg['Dataset']['num_sample'],
                                self.cfg['Module']['feature_dim'])).to(self.device)
        all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device,
                                 dtype=torch.float32)
        for epoch in range(epochs):
            for xs, _, aligned_idx, idx in data_loader:
                hs, qs, xrs, zs = model(xs)
                loss_list = []
                for v in range(num_views):
                    for w in range(v + 1, num_views):
                        loss_list.append(criterion.forward_feature(hs[v][aligned_idx == 1], hs[w][aligned_idx == 1]))
                        loss_list.append(criterion.forward_label(qs[v][aligned_idx == 1], qs[w][aligned_idx == 1]))
                    loss_list.append(torch.nn.MSELoss()(xs[v], xrs[v]))
                loss = sum(loss_list)
                if epoch > 0:
                    loss = loss + lambda_graph * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for v in range(num_views):
                    all_new_z[v][idx] = zs[v].detach().clone()
        # save model
        torch.save({'model': model.state_dict()}, self.model_path + '_cl.pth')

    def train1(self):
        # get param
        num_views = self.cfg['Dataset']['num_views']
        num_classes = self.cfg['Dataset']['num_classes']
        epochs = self.cfg['training']['tune_epochs']
        lambda_graph = self.cfg['training']['lambda_graph']
        # get model
        model = Network(num_views,
                        self.cfg['Module']['in_dim'],
                        self.cfg['Module']['feature_dim'],
                        self.cfg['Module']['high_feature_dim'],
                        num_classes, self.device).to(self.device)
        model.load_state_dict(torch.load(self.model_path + '_cl.pth')['model'])
        # valid aligned data and get info
        print("valid on aligned data")
        _, _, _, pre_labels, hs, labels, qs, zs = valid(model, self.device, self.dataset_aligned, num_views,
                                                        self.cfg['Dataset']['num_sample'], num_classes, eval_h=False)
        # performer alignment
        fea_realigned, labels_realigned, realigned_idx = get_alignment(self.dataset_aligned.fea, hs, qs, zs, pre_labels,
                                                                       self.dataset_aligned.labels, self.aligned_idx,
                                                                       self.device)
        dataset_realigned = MvDataset(fea_realigned, labels_realigned, realigned_idx, self.device)
        print("valid on realigned data")
        valid(model, self.device, dataset_realigned, num_views, self.cfg['Dataset']['num_sample'], num_classes, eval_h=False)
        data_loader = torch.utils.data.DataLoader(
            dataset_realigned,
            batch_size=self.cfg['Dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
        )
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.cfg['training']['lr'],
                                     weight_decay=self.cfg['training']['weight_decay'])
        criterion = Loss(self.cfg['Dataset']['batch_size'],
                         self.cfg['Dataset']['num_classes'],
                         self.cfg['training']['temperature_f'],
                         self.cfg['training']['temperature_l'],
                         self.device).to(self.device)
        all_new_x = [torch.from_numpy(v_data) for v_data in dataset_realigned.fea]
        all_new_z = torch.ones((self.cfg['Dataset']['num_views'], self.cfg['Dataset']['num_sample'],
                                self.cfg['Module']['feature_dim'])).to(self.device)
        all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device,
                                 dtype=torch.float32)
        for epoch in range(epochs):
            for xs, _, aligned_idx, idx in data_loader:
                hs, qs, xrs, zs = model(xs)
                loss_list = []
                for v in range(num_views):
                    for w in range(v + 1, num_views):
                        loss_list.append(criterion.forward_feature(hs[v][aligned_idx == 1], hs[w][aligned_idx == 1]))
                        loss_list.append(criterion.forward_label(qs[v][aligned_idx == 1], qs[w][aligned_idx == 1]))
                    loss_list.append(torch.nn.MSELoss()(xs[v], xrs[v]))
                loss = sum(loss_list)
                if epoch > 0:
                    loss = loss + lambda_graph * criterion.graph_loss(all_graph[:, idx], zs, all_new_z)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for v in range(num_views):
                    all_new_z[v][idx] = zs[v].detach().clone()
            if epoch % 5 == 0:
                print("epoch: " + str(epoch))
                _, _, _, pre_labels, hs, labels, qs, zs = valid(model, self.device, dataset_realigned, num_views,
                                                                self.cfg['Dataset']['num_sample'], num_classes,
                                                                eval_h=False)
                fea_realigned, labels_realigned, realigned_idx = get_alignment(dataset_realigned.fea, hs, qs, zs,
                                                                               pre_labels,
                                                                               dataset_realigned.labels,
                                                                               dataset_realigned.aligned_idx,
                                                                               self.device)
                dataset_realigned = MvDataset(fea_realigned, labels_realigned, realigned_idx, self.device)
                valid(model, self.device, dataset_realigned, num_views, self.cfg['Dataset']['num_sample'], num_classes, eval_con=False)
                data_loader = torch.utils.data.DataLoader(dataset_realigned, batch_size=self.cfg['Dataset']['batch_size'], shuffle=True, drop_last=True)
                all_new_x = [torch.from_numpy(v_data) for v_data in dataset_realigned.fea]
                all_graph = torch.tensor(getMvKNNGraph(all_new_x, k=self.cfg['training']['knn']), device=self.device, dtype=torch.float32)


if __name__ == '__main__':
    dataset = {0: "Fashion", 1: "BDGP", 2: "HandWritten", 3: "Reuters_dim10", 4: "WebKB", 5: "Caltech101-7"}
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default='2', help='dataset id')
    parser.add_argument('--print_num', type=int, default='10', help='gap of print evaluations')
    parser.add_argument('--test_time', type=int, default='20', help='number of test times')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config = get_default_config(dataset)
    config['print_num'] = args.print_num
    config['Dataset']['name'] = dataset
    set_seed(config['training']['seed'])
    run = RunModule(config, device)
    # run.pretrain_ae()
    # run.pretrain_cl()
    # for run.cfg['training']['knn'] in [5, 8, 10, 15, 20, 25, 30, 35, 50, 60]:
    #     for run.cfg['training']['lambda_graph'] in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
    #         print("knn: " + str(run.cfg['training']['knn']) + ", lambda_graph: " + str(run.cfg['training']['lambda_graph']))
    #         run.pretrain_ae()
    #         run.pretrain_cl()
    run.train1()
