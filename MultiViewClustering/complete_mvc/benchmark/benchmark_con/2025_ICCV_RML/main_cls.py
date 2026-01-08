import random, copy, math, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from _utils_cls import load_data, evaluate

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(in_dim, hidden_dim); self.act_1 = nn.GELU(); self.dropout_1 = nn.Dropout(dropout_rate)
        self.fc_2 = nn.Linear(hidden_dim, out_dim); self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout_2(self.fc_2(self.dropout_1(self.act_1(self.fc_1(x)))))

class NORM(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model)); self.bias = nn.Parameter(torch.zeros(d_model)); self.eps = eps

    def forward(self, x): # x shape: (batch_size, sequence_length, d_model)
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model; self.d_k = d_model // heads; self.h = heads
        self.q_linear = nn.Linear(d_model, d_model); self.v_linear = nn.Linear(d_model, d_model); self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout); self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v): # q, k, v shape: (batch_size, view, feature_dim)
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k) # shape: (batch_size, view, heads, d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k) # shape: (batch_size, view, heads, d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k) # shape: (batch_size, view, heads, d_k)
        k = k.transpose(1, 2) # shape: (batch_size, heads, view, d_k)
        q = q.transpose(1, 2) # shape: (batch_size, heads, view, d_k)
        v = v.transpose(1, 2) # shape: (batch_size, heads, view, d_k)
        output_scores, scores = self.attention(q, k, v, self.d_k, self.dropout) # shape: (batch_size, heads, view, d_k)
        concat = output_scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model) # shape: (batch_size, view, d_model)
        output = self.out(concat) # shape: (batch_size, view, d_model)
        return output, scores # output shape: (batch_size, view, d_model), scores shape: (batch_size, view, view)
    
    @staticmethod
    def attention(q, k, v, d_k, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scores shape: (batch_size, heads, view, view)
        scores = F.softmax(scores, dim=-1) # shape: (batch_size, heads, view, view)
        scores = dropout(scores) if dropout is not None else scores # shape: (batch_size, heads, view, view)
        output = torch.matmul(scores, v) # shape: (batch_size, heads, view, d_k)
        return output, scores # output shape: (batch_size, heads, view, d_k), scores shape: (batch_size, heads, view, view)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.2):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff); self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model); self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x): # x shape: (batch_size, view, feature_dim)
        return self.dropout_2(self.linear_2(self.dropout_1(F.relu(self.linear_1(x))))) # shape: (batch_size, view, feature_dim)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = NORM(d_model); self.norm_2 = NORM(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x): # x shape: (batch_size, view, feature_dim)
        x_temp = self.norm_1(x) # shape: (batch_size, view, feature_dim)
        output, scores = self.attn(x_temp, x_temp, x_temp) # shape: (batch_size, view, feature_dim), scores shape: (batch_size, view, view)
        x = x + self.dropout_1(output) # shape: (batch_size, view, feature_dim)
        x_temp = self.norm_2(x) # shape: (batch_size, view, feature_dim)
        x = x + self.dropout_2(self.ff(x_temp)) # shape: (batch_size, view, feature_dim)
        return x, scores # shape: (batch_size, view, feature_dim), scores shape: (batch_size, view, view)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N # N is the number of encoder layers
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(d_model, heads, dropout)) for i in range(N)]) # copy.deepcopy is used to create a deep copy of the encoder layer
        self.norm = NORM(d_model) # L2 normalization

    def forward(self, x): # x shape: (batch_size, view, feature_dim)
        for i in range(self.N):
            x, scores = self.layers[i](x) # x shape: (batch_size, view, feature_dim), scores shape: (batch_size, view, view)
        return self.norm(x) # shape: (batch_size, view, feature_dim)
    
class RML(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, multi_blocks=1, multi_heads=1):
        super(RML, self).__init__()
        self.view = view
        self.embedding_layers = nn.ModuleList([MLP(input_size[v], input_size[v], feature_dim) for v in range(view)])
        self.TransEncoder = TransformerEncoder(d_model=feature_dim, N=multi_blocks, heads=multi_heads, dropout=0.3) # N=1, H=1
        self.feature_contrastive_module = nn.Sequential(nn.Linear(feature_dim, high_feature_dim))
        self.label_contrastive_module = nn.Sequential(nn.Linear(high_feature_dim, class_num), nn.Softmax(dim=1))

    def forward(self, x): # [x1 xv] -> embeddings -> Hs -> fusion_H -> fusion_Z -> fusion_Q
        embeddings = [self.embedding_layers[v](x[v]) for v in range(self.view)] # shape: (batch_size, feature_dim)
        embeddings = torch.stack(embeddings, dim=1)  # shape: (batch_size, view, feature_dim)
        Z = self.TransEncoder(embeddings) # shape: (batch_size, view, feature_dim)
        fusion_Z = Z.sum(dim=1) # shape: (batch_size, feature_dim), sum over the view dimension
        fusion_H = F.normalize(self.feature_contrastive_module(fusion_Z), dim=1) # shape: (batch_size, high_feature_dim), L2 normalization
        fusion_Q = self.label_contrastive_module(fusion_H) # shape: (batch_size, class_num)
        return fusion_Z, fusion_H, fusion_Q
    
class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.device = device
        self.similarity = nn.CosineSimilarity(dim=2) # input two vectors, output the cosine similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # input logits and labels, output the loss
    
    def get_correlated_mask(self, N):
        mask = torch.ones((N//2, N//2)).fill_diagonal_(0) # shape: (N//2, N//2) with diagonal elements set to 0
        mask = torch.concat([mask, mask], dim=1) # shape: (N//2, N)
        mask = torch.concat([mask, mask], dim=0) # shape: (N, N)
        mask = mask.bool() # shape: (N, N)
        return mask

    def forward_feature(self, h_i, h_j):
        # Compute the feature contrastive loss: 1/2*(L_fc^(i,j) + L_fc^(j,i))
        # h_i and h_j are the feature vectors of the positive samples, shape: (batch_size, feature_dim) unit vectors
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0) # shape: (N, feature_dim), h is the concatenation of h_i and h_j
        sim = torch.matmul(h, h.T) / self.temperature_f # shape: (N, N), sim is the cosine similarity matrix
        sim_i_j = torch.diag(sim, diagonal=self.batch_size) # shape: (N//2), sim_i_j is the cosine similarity between h_i and h_j
        sim_j_i = torch.diag(sim, diagonal=-self.batch_size) # shape: (N//2), sim_j_i is the cosine similarity between h_j and h_i
        sim_of_positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) # shape: (N, 1), similarity between the positive samples
        mask = self.get_correlated_mask(N) # shape: (N, N), mask is the mask for the negative samples
        sim_of_negative_samples = sim[mask].reshape(N, -1) # shape: (N, N-2), similarity between the negative samples
        labels = torch.zeros(N).to(self.device).long() # shape: (N,), labels is the labels for the positive samples
        logits = torch.cat((sim_of_positive_samples, sim_of_negative_samples), dim=1) # shape: (N, N-1), logits is the logits for the positive and negative samples
        loss = self.criterion(logits, labels) # shape: (1,), loss is the loss for logits and labels
        loss /= N # shape: (1,), loss is the average loss for logits and labels, N is the number of samples
        return loss

def missing_mask(sample_num, view_num, missing_rate):
    mask = np.ones((sample_num, view_num)) # shape: (sample_num, view_num), mask is the mask for the samples
    num_partial_mask = int(sample_num * missing_rate) # num_partial_mask is the number of samples to be masked
    for i in range(num_partial_mask): # generate the partial mask for the first num_partial_mask samples
        num_zero_indices = np.random.randint(1, view_num) # 1 < num_zero_indices < view_num, num_zero_indices is the number of zeros to be set to 0
        zero_indices = np.random.choice(view_num, size=num_zero_indices, replace=False) # randomly choose num_zero_indices indices from view_num to set to 0
        mask[i, zero_indices] = 0 # set the zeros to 0
    np.random.shuffle(mask) # shuffle the mask to ensure the randomness across samples
    return mask.astype(np.float32) # shape: (sample_num, view_num), mask is the mask for the samples

def noise_addition(sample_num, feature_num, noise_std, noise_rate):
    noise_matrix = []
    for i in range(sample_num):
        if np.random.random() < noise_rate:
            noise = np.random.randn(feature_num) * noise_std
            noise_matrix.append(noise)
        else:
            noise_matrix.append(np.zeros(feature_num))
    noise_matrix = np.array(noise_matrix)
    return noise_matrix.astype(np.float32) # shape: (sample_num, feature_num), noise_matrix is the noise matrix

def validation(model, dataset, view, data_size, class_num, eval_h=False, eval_q=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_list, y, idx = next(iter(torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False, drop_last=False))) # Get the whole dataset
    x_list = [x_list[v].to(device) for v in range(view)]; y = y.numpy()
    with torch.no_grad():
        fusion_Z, fusion_H, fusion_Q = model.forward(x_list)
    fusion_Z = fusion_Z.cpu().detach().numpy(); fusion_H = fusion_H.cpu().detach().numpy(); fusion_Q = fusion_Q.cpu().detach().numpy()
    fusion_Q = fusion_Q.argmax(1)
    if eval_h == True:
        y_pred = KMeans(n_clusters=class_num).fit_predict(fusion_H)
        nmi, ari, acc, pur = evaluate(y, y_pred)
        print('Evaluation on feature vectors: ACC = {:.4f}; NMI = {:.4f}; ARI = {:.4f}; PUR = {:.4f}'.format(acc, nmi, ari, pur))
    if eval_q == True:
        classification_result = classification_report(y, fusion_Q, output_dict=True, zero_division=1)
        accuracy = classification_result['accuracy']
        precision = classification_result['weighted avg']['precision']
        f1_score = classification_result['weighted avg']['f1-score']
        recall = classification_result['weighted avg']['recall']
        print('Evaluation on cluster assignments: ACC = {:.4f}; weighted Precision = {:.4f}; weighted F1-score = {:.4f}; weighted Recall = {:.4f}'.format(accuracy, precision, f1_score, recall))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RML_LCE')
    parser.add_argument('--dataset', default='BDGP', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--con_iterations", default=50, type=int)
    parser.add_argument("--feature_dim", default=256, type=int)
    parser.add_argument("--high_feature_dim", default=256, type=int)
    parser.add_argument('--miss_rate', default=0.25, type=float)
    parser.add_argument('--noise_rate', default=0.25, type=float)
    parser.add_argument('--noise_std', default=0.4, type=float)
    parser.add_argument('--seed', default=10, type=int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset; batch_size = args.batch_size; temperature_f = args.temperature_f; learning_rate = args.learning_rate; weight_decay = args.weight_decay; 
    con_iterations = args.con_iterations; feature_dim = args.feature_dim; high_feature_dim = args.high_feature_dim; 
    miss_rate = args.miss_rate; noise_rate = args.noise_rate; noise_std = args.noise_std; seed = args.seed
    if dataset_name == "BDGP": miss_rate = 0.25; noise_rate = 0.25; noise_std = 0.4; con_iterations = 1000; seed = 0
    if dataset_name == "Cora": miss_rate = 0.25; noise_rate = 0.25; noise_std = 0.4; con_iterations = 500; seed = 9
    if dataset_name == "DHA": miss_rate = 0.25; noise_rate = 0.25; noise_std = 0.4; con_iterations = 800; seed = 4
    if dataset_name == "WebKB": miss_rate = 0.75; noise_rate = 0.75; noise_std = 0.4; con_iterations = 400; seed = 1
    if dataset_name == "NGs": miss_rate = 0.50; noise_rate = 0.50; noise_std = 0.4; con_iterations = 400; seed = 1
    if dataset_name == "VOC": miss_rate = 0.25; noise_rate = 0.25; noise_std = 0.4; con_iterations = 500; seed = 1
    if dataset_name == "YoutubeVideo": miss_rate = 0.25; noise_rate = 0.25; noise_std = 0.4; con_iterations = 200000; seed = 0
    if dataset_name == "Prokaryotic": miss_rate = 0.50; noise_rate = 0.50; noise_std = 0.4; con_iterations = 1000; seed = 9
    if dataset_name == "Cifar100": miss_rate = 0.25; noise_rate = 0.25; noise_std = 0.4; con_iterations = 10000; seed = 4
    train_rate = 0.7; noise_label_rate = 0.5; Lambda = 1000
    
    ## 1. Set seed for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    ## 2. Load data and initialize model.
    dataset, dims, view, data_size, class_num = load_data(args.dataset, type='train', trainset_rate=train_rate, noise_rate=noise_label_rate, seed=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = RML(view, dims, args.feature_dim, args.high_feature_dim, class_num, multi_blocks=1, multi_heads=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device)
    criterion_ce = nn.CrossEntropyLoss()
    mode = args.mode; miss_rate = args.miss_rate; noise_rate = args.noise_rate; noise_std = args.noise_std 
    
    ## 3. Train and evaluate the model.
    for epoch in range(args.con_iterations):
        xs, y, _ = next(iter(dataloader))
        xs = [xs[v].to(device) for v in range(view)]
        mask = missing_mask(args.batch_size, view, miss_rate)
        xs_masked = [torch.from_numpy(np.expand_dims(mask[:, v], axis=1) * xs[v].cpu().detach().numpy()).to(device) for v in range(view)]
        xs_noised = [torch.from_numpy(noise_addition(xs[v].shape[0], xs[v].shape[1], noise_std, noise_rate) + xs[v].cpu().detach().numpy()).to(device) for v in range(view)]
        optimizer.zero_grad()
        _, h_origin, q_origin = model(xs)
        _, h_masked, q_masked = model(xs_masked)
        _, h_noised, q_noised = model(xs_noised)
        loss_contrastive_1 = criterion.forward_feature(h_noised, h_masked)
        loss_contrastive_2 = criterion.forward_feature(h_masked, h_noised)
        loss_crossentropy_origin = criterion_ce(q_origin, y.long())
        loss_crossentropy_masked = criterion_ce(q_masked, y.long())
        loss_crossentropy_noised = criterion_ce(q_noised, y.long())
        loss = (loss_crossentropy_origin + loss_crossentropy_masked + loss_crossentropy_noised) + Lambda * (loss_contrastive_1 + loss_contrastive_2)
        loss.backward()
        optimizer.step()
        print('\r', 'Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(loss), end='') # \r: overwrite the same line
    dataset, dims, view, data_size, class_num = load_data(args.dataset, type='test', trainset_rate=train_rate, noise_rate=noise_label_rate, seed=0)
    validation(model, dataset, view, data_size, class_num, eval_h=False, eval_q=True)
