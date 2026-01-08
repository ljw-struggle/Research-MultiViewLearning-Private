import os, random, copy, math, time, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from _utils import load_data, valid
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class Mlp(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim); self.fc2 = nn.Linear(mlp_dim, out_dim); self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x):
        out = self.fc1(x); out = self.act(out); out = self.dropout1(out) if self.dropout1 else out
        out = self.fc2(out); out = self.dropout2(out) if self.dropout2 else out
        return out

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model)); self.bias = nn.Parameter(torch.zeros(d_model)); self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2) # transpose to get dimensions bs * N * sl * d_model/h
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        output_scores, scores = self.attention(q, k, v, self.d_k, self.dropout)
        concat = output_scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, scores
    
    @staticmethod
    def attention(q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scores shape is [bs heads view view]
        scores = F.softmax(scores, dim=-1)
        scores = dropout(scores) if dropout is not None else scores
        output = torch.matmul(scores, v)
        return output, scores

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model); self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        output, scores = self.attn(x2, x2, x2)
        x = x + self.dropout_1(output)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, scores

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(d_model, heads, dropout)) for i in range(N)])
        self.norm = Norm(d_model)

    def forward(self, src):
        x = src
        for i in range(self.N):
            x, scores = self.layers[i](x)
        return self.norm(x), scores

class Network(nn.Module):
    def __init__(self, class_num, feature_dim, contrastive_feature_dim, device, data_dim_list, view_num, multi_blocks=1, multi_heads=1):
        super(Network, self).__init__()
        self.view_num = view_num
        self.embeddinglayers_in = nn.ModuleList([Mlp(d, d, feature_dim) for d in data_dim_list]).to(device)
        self.MMLEncoder = TransformerEncoder(d_model=feature_dim, N=multi_blocks, heads=multi_heads, dropout=0.3).to(device)        # N=3, H=4
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, contrastive_feature_dim))
        self.label_module = nn.Sequential(nn.Linear(contrastive_feature_dim, class_num), nn.Softmax(dim=1))

    def forward(self, x): # [x1 xv] -> embeddings -> Hs -> fusion_H -> fusion_Z -> fusion_Q
        embeddings = []
        for v in range(self.view_num):
            embeddings.append(self.embeddinglayers_in[v](x[v]))
        Tensor = torch.stack(embeddings, dim=1)  # B,view,d
        H, scores = self.MMLEncoder(Tensor)
        fusion_H = torch.einsum('bvd->bd', H)
        fusion_Z = normalize(self.feature_contrastive_module(fusion_H), dim=1)
        hs = []
        for v in range(self.view_num):
            hs.append(H.chunk(self.view_num, dim=1)[v][:, 0, :])
        fusion_Q = self.label_module(fusion_Z)
        return fusion_H, fusion_Z, fusion_Q, scores, hs

class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
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

    def forward_feature(self, h_i, h_j):
        num_rows = h_i.shape[0]
        N = 2 * num_rows
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, num_rows)
        sim_j_i = torch.diag(sim, -num_rows)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    
def mask(rows, cols, p):
    tensor = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        if i < int(rows * p):
            while True:
                row = np.random.randint(0, 2, size=cols)
                if np.count_nonzero(row) < cols and np.count_nonzero(row) > 0:
                    tensor[i, :] = row
                    break
        else:
            tensor[i, :] = 1
    np.random.shuffle(tensor)
    tensor = torch.tensor(tensor)
    return tensor

def add_noise(matrix, std, p):
    rows, cols = matrix.shape
    noisy_matrix = matrix.clone()
    for i in range(rows):
        if random.random() < p:
            noise = torch.randn(cols, device=device) * std
            noisy_matrix[i] += noise
    return noisy_matrix

def RML(iteration, model, mode, miss_rate, noise_rate, Gaussian_noise, data_loader):
    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
            y = y.to(device)
        break
    masked_xs = []; noised_xs = []; num_rows = xs[0].shape[0]
    mask_tensor = mask(num_rows, view, miss_rate).to(device)
    for v in range(view):
        masked_x = mask_tensor[:, v].unsqueeze(1) * xs[v]
        masked_xs.append(masked_x)
    for v in range(view):
        noised_x = add_noise(xs[v], Gaussian_noise, noise_rate)
        noised_xs.append(noised_x)
    optimizer.zero_grad()
    _, xs_z, q, scores, hs = model(xs)
    _, mask_z, mask_q, _, _ = model(masked_xs)
    _, noise_z, noise_q, _, _ = model(noised_xs)
    if mode == 'RML' or mode == 'RML_LCE':
        loss_con_1 = criterion.forward_feature(noise_z, mask_z)
        loss_con_2 = criterion.forward_feature(mask_z, noise_z)
        loss_con = loss_con_1 + loss_con_2
        loss = loss_con
    if mode == 'RML_LCE':
        crossentropyloss = nn.CrossEntropyLoss()
        loss_ce_x = crossentropyloss(q, y.long())
        loss_ce_mask = crossentropyloss(mask_q, y.long())
        loss_ce_noise = crossentropyloss(noise_q, y.long())
        loss_y = loss_ce_x + loss_ce_mask + loss_ce_noise
        loss = loss_y + Lambda * loss_con
    loss.backward()
    optimizer.step()
    print('\r', 'Iteration {}'.format(iteration), 'Loss:{:.6f}'.format(loss), end='') # \r: overwrite the same line

if __name__ == '__main__':
    Datasets = ['DHA', 'BDGP', 'Prokaryotic', 'Cora', 'YoutubeVideo', 'WebKB', 'VOC', 'NGs', 'Cifar100']
    Dataname = 'BDGP'; MODE = 'RML'; tsne = False; T = 5
    if MODE == 'RML':
        train_rate = 1.0; noise_label_rate = 0
    if MODE == 'RML_LCE':
        train_rate = 0.7; noise_label_rate = 0.5
    multi_blocks = 1; multi_heads = 1; Lambda = 1 if noise_label_rate < 0.3 else 1000
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--learning_rate", default=0.0003)
    parser.add_argument("--weight_decay", default=0.)
    parser.add_argument("--workers", default=8)
    parser.add_argument("--con_iterations", default=50)
    parser.add_argument("--tune_iterations", default=50)
    parser.add_argument("--feature_dim", default=256)
    parser.add_argument("--contrastive_feature_dim", default=256)
    parser.add_argument('--mode', type=str, default=MODE)
    parser.add_argument('--miss_rate', type=str, default=0.25)
    parser.add_argument('--noise_rate', type=str, default=0.25)
    parser.add_argument('--Gaussian_noise', type=str, default=0.4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset == "BDGP":
        args.miss_rate = 0.25; args.noise_rate = 0.25; args.Gaussian_noise = 0.4; args.con_iterations = 1000; seed = 0
    if args.dataset == "Cora":
        args.miss_rate = 0.25; args.noise_rate = 0.25; args.Gaussian_noise = 0.4; args.con_iterations = 500; seed = 9
    if args.dataset == "DHA":
        args.miss_rate = 0.25; args.noise_rate = 0.25; args.Gaussian_noise = 0.4; args.con_iterations = 800; seed = 4
    if args.dataset == "WebKB":
        args.miss_rate = 0.75; args.noise_rate = 0.75; args.Gaussian_noise = 0.4; args.con_iterations = 400; seed = 1
    if args.dataset == "NGs":
        args.miss_rate = 0.50; args.noise_rate = 0.50; args.Gaussian_noise = 0.4; args.con_iterations = 400; seed = 1
    if args.dataset == "VOC":
        args.miss_rate = 0.25; args.noise_rate = 0.25; args.Gaussian_noise = 0.4; args.con_iterations = 500; seed = 1
    if args.dataset == "YoutubeVideo":
        args.miss_rate = 0.25; args.noise_rate = 0.25; args.Gaussian_noise = 0.4; args.con_iterations = 200000; seed = 0
    if args.dataset == "Prokaryotic":
        args.miss_rate = 0.50; args.noise_rate = 0.50; args.Gaussian_noise = 0.4; args.con_iterations = 1000; seed = 9
    if args.dataset == "Cifar100":
        args.miss_rate = 0.25; args.noise_rate = 0.25; args.Gaussian_noise = 0.4; args.con_iterations = 10000; seed = 4
    metric1 = []; metric2 = []; metric3 = []; metric4 = []
    for i in range(T):
        print("ROUND:{}".format(i + 1))
        seed = seed + i
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        dataset, dims, view, data_size, class_num, dimss = load_data(args.dataset, trainset_rate=train_rate, type='train', seed=i, mode=MODE, noise_rate=noise_label_rate)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        epochs = int(args.con_iterations / (data_size / args.batch_size))
        model = Network(class_num, args.feature_dim, args.contrastive_feature_dim, device, dims, view, multi_blocks=multi_blocks, multi_heads=multi_heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
        mode = args.mode
        miss_rate = args.miss_rate
        noise_rate = args.noise_rate
        Gaussian_noise = args.Gaussian_noise
        time0 = time.time()
        iteration = 1
        while iteration <= args.con_iterations:
            RML(iteration, model, mode, miss_rate, noise_rate, Gaussian_noise, data_loader)
            iteration += 1
        if mode == 'RML':
            dataset, dims, view, data_size, class_num, dimss = load_data(args.dataset, trainset_rate=train_rate, type='train', seed=i, noise_rate=0)
            m1, m2, m3, m4 = valid(model, device, dataset, view, data_size, class_num, eval_z=True)
        if mode == 'RML_LCE':
            dataset, dims, view, data_size, class_num, dimss = load_data(args.dataset, trainset_rate=train_rate, type='test', seed=i, noise_rate=noise_label_rate)
            m1, m2, m3, m4 = valid(model, device, dataset, view, data_size, class_num, eval_q=True)
        metric1.append(m1); metric2.append(m2); metric3.append(m3); metric4.append(m4)
    print('%.3f'% np.mean(metric1), '± %.3f'% np.std(metric1), metric1)
    print('%.3f'% np.mean(metric2), '± %.3f'% np.std(metric2), metric2)
    print('%.3f'% np.mean(metric3), '± %.3f'% np.std(metric3), metric3)
    print('%.3f'% np.mean(metric4), '± %.3f'% np.std(metric4), metric4)
    if tsne == True:
        mask_x = []; noise_x = []; model.eval()
        ALL_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
        for step, (xs, ys, _) in enumerate(ALL_loader):
            ys = ys.numpy()
            xs = [x.to(device) for x in xs]
            num_rows = xs[0].shape[0]
            miss = mask(num_rows, view, miss_rate).to(device)
            for v in range(view):
                miss = miss[:, v].unsqueeze(1)*xs[v]
                mask_x.append(miss)
            for v in range(view):
                noisedx = add_noise(xs[v], Gaussian_noise, noise_rate)
                noise_x.append(noisedx)
            with torch.no_grad():
                xr, _, z, _, _, _ = model.forward(xs)
                xmr, _, zm, _, _, _ = model.forward(mask_x)
                xnr, _, zn, _, _, _ = model.forward(noise_x)
                z = z.cpu().detach().numpy()
                zm = zm.cpu().detach().numpy()
                zn = zn.cpu().detach().numpy()
