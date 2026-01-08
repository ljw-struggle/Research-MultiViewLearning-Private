import argparse, random, os, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from _utils import load_data, evaluate

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
    
class MFLVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(MFLVC, self).__init__()
        # self.encoders = nn.ModuleList([Encoder(input_size[v], feature_dim) for v in range(view)])
        # self.decoders = nn.ModuleList([Decoder(input_size[v], feature_dim) for v in range(view)])
        # self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        # self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        # self.view = view
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.label_contrastive_module = nn.Sequential(nn.Linear(feature_dim, class_num), nn.Softmax(dim=1))
        self.view = view

    def forward(self, x_list):
        h_list = []; z_list = []; q_list = []; r_list = []
        for v in range(self.view):
            z = self.encoders[v](x_list[v])
            h = F.normalize(self.feature_contrastive_module(z), dim=1) # L2 normalization
            q = self.label_contrastive_module(z)
            r = self.decoders[v](z)
            z_list.append(z); h_list.append(h); q_list.append(q); r_list.append(r)
        return z_list, h_list, q_list, r_list
    
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, batch_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear_1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, src): # shape: (batch_size, feature_dim * view)
        # Self-attention
        src2, attn_weights = self.self_attn(src, src, src, average_attn_weights=True) # shape: (batch_size, feature_dim * view); attn_weights: (batch_size, batch_size)
        src = src + self.dropout_1(src2) # shape: (batch_size, feature_dim * view)
        src = self.norm_1(src) # shape: (batch_size, feature_dim * view)
        # Feedforward
        src2 = self.linear_2(self.dropout(F.relu(self.linear_1(src)))) # shape: (batch_size, feature_dim * view)
        src = src + self.dropout_2(src2) # shape: (batch_size, feature_dim * view)
        src = self.norm_2(src) # shape: (batch_size, feature_dim * view)
        return src, attn_weights

class GCFAggMVC(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim):
        super(GCFAggMVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.spec_view = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.comm_view = nn.Sequential(nn.Linear(feature_dim * view, high_feature_dim))
        self.transformer_encoder_layer = CustomTransformerEncoderLayer(d_model=feature_dim * view, nhead=1, dim_feedforward=256, batch_first=False)
        self.view = view
        
    def forward(self, xs):
        zs = []; hs = []; rs = []
        for v in range(self.view):
            z = self.encoders[v](xs[v])
            h = F.normalize(self.spec_view(z), dim=1)
            r = self.decoders[v](z)
            hs.append(h); zs.append(z); rs.append(r)
        return zs, hs, rs
    
    def GCFAgg(self, xs):
        zs = [self.encoders[v](xs[v]) for v in range(self.view)] # shape: (batch_size, feature_dim) * view
        z_concat = torch.cat(zs, 1) # shape: (batch_size, feature_dim * view)
        z_concat, attn_weights = self.transformer_encoder_layer(z_concat) # shape: (batch_size, feature_dim * view); attn_weights: (batch_size, batch_size)
        z_concat = F.normalize(self.comm_view(z_concat), dim=1) # shape: (batch_size, high_feature_dim)
        return z_concat, attn_weights # shape: (batch_size, high_feature_dim); attn_weights: (batch_size, batch_size)
    
class MVCLLoss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(MVCLLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # input logits and labels, output the loss

    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask

    def forward_feature_structure_guided_contrastive_Loss(self, h_i, h_j, attn_weights):
        attn_weights_1 = attn_weights.repeat(2, 2) # shape: (2*batch_size, 2*batch_size)
        attn_weights_2 = torch.ones_like(attn_weights_1).to(self.device) - attn_weights_1 # shape: (2*batch_size, 2*batch_size)
        
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0) # shape: (2*batch_size, feature_dim), h is the concatenation of h_i and h_j
        sim = torch.matmul(h, h.T) / self.temperature_f # shape: (2*batch_size, 2*batch_size), sim is the cosine similarity matrix
        sim_1 = torch.multiply(sim, attn_weights_2) # shape: (2*batch_size, 2*batch_size)
        sim_i_j = torch.diag(sim, self.batch_size) # shape: (2*batch_size,), sim_i_j is the cosine similarity between h_i and h_j
        sim_j_i = torch.diag(sim, -self.batch_size) # shape: (2*batch_size,), sim_j_i is the cosine similarity between h_j and h_i
        sim_of_positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive samples
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative samples
        sim_of_negative_samples = sim_1[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative samples
        labels = torch.zeros(N).to(self.device).long() # shape: (N,), labels is the labels for the positive samples
        logits = torch.cat((sim_of_positive_samples, sim_of_negative_samples), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative samples
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for logits and labels
        loss /= N # shape: (1,), loss is the average loss for logits and labels, N is the number of samples
        return loss # shape: (1,)

def validation(model, dataset, view, data_size, class_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    model.eval()
    z_concat_list = []; label_list = []
    for xs, y, _ in test_loader:
        xs = [x.to(device) for x in xs] # shape: (batch_size, input_dim) * view
        with torch.no_grad():
            z_concat, attn_weights = model.GCFAgg(xs)
            z_concat_list.extend(z_concat.detach().cpu().numpy())
            label_list.extend(y.detach().cpu().numpy())
    z_concat_list = np.array(z_concat_list) # shape: (batch_size, feature_dim * view)
    label_list = np.array(label_list) # shape: (batch_size,)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(z_concat_list)
    nmi, ari, acc, pur = evaluate(label_list, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCFAggMVC')
    parser.add_argument('--dataset', default='Hdigit', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--tune_epochs", default=100, type=int)
    parser.add_argument("--low_feature_dim", default=512, type=int)
    parser.add_argument("--high_feature_dim", default=128, type=int)
    parser.add_argument("--seed", default=10, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset; batch_size = args.batch_size; temperature_f = args.temperature_f; learning_rate = args.learning_rate; weight_decay = args.weight_decay
    mse_epochs = args.mse_epochs; tune_epochs = args.tune_epochs; low_feature_dim = args.low_feature_dim; high_feature_dim = args.high_feature_dim; seed = args.seed
    tune_epochs_dict = {"MNIST-USPS": 100, "CCV": 100, "Hdigit": 100, "YouTubeFace": 100, "Cifar10": 10, "Cifar100": 200, "Prokaryotic": 50, "Synthetic3d": 100, "Caltech-2V": 100, "Caltech-3V": 100, "Caltech-4V": 150, "Caltech-5V": 200}
    seed_dict = {"MNIST-USPS": 10, "CCV": 3, "Hdigit": 10, "YouTubeFace": 10, "Cifar10": 10, "Cifar100": 10, "Prokaryotic": 10, "Synthetic3d": 10, "Caltech-2V": 10, "Caltech-3V": 10, "Caltech-4V": 10, "Caltech-5V": 5}
    tune_epochs = tune_epochs_dict[dataset_name]; seed = seed_dict[dataset_name]
    os.makedirs('./result', exist_ok=True)

    ## 1. Set seed for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(dataset_name)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = GCFAggMVC(view, dims, low_feature_dim, high_feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = MVCLLoss(batch_size, temperature_f, device)
    criterion_mse = torch.nn.MSELoss()
    model.train()
    for epoch in range(mse_epochs):
        tot_loss = 0.
        for xs, _, _ in dataloader:
            xs = [x.to(device) for x in xs] # shape: (batch_size, input_dim) * view
            optimizer.zero_grad()
            zs, hs, rs = model(xs)
            loss = sum([criterion_mse(xs[v], rs[v]) for v in range(view)])
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Pre-training Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(dataloader)))
    for epoch in range(tune_epochs):
        tot_loss = 0.
        for xs, _, _ in dataloader:
            xs = [x.to(device) for x in xs] # shape: (batch_size, input_dim) * view
            optimizer.zero_grad()
            zs, hs, rs = model(xs)
            z_concat, attn_weights = model.GCFAgg(xs)
            loss_list = []
            for v in range(view):
                loss_list.append(criterion.forward_feature_structure_guided_contrastive_Loss(hs[v], z_concat, attn_weights))
                loss_list.append(criterion_mse(xs[v], rs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print('Fine-tuning Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(dataloader)))
    validation(model, dataset, view, data_size, class_num)
    torch.save(model.state_dict(), './result/' + dataset_name + '.pth')
