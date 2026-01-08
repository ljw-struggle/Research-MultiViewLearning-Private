import argparse, random, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import RandomSampler, SequentialSampler
from utils import load_data, evaluate

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, latent_dim))
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, input_dim))
        
    def forward(self, x):
        return self.decoder(x)

class PVC(nn.Module):
    def __init__(self, view_num, input_sizes, latent_dim):
        super(PVC, self).__init__()
        self.view_num = view_num
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for view in range(view_num):
            self.encoders.append(Encoder(input_sizes[view], latent_dim))
            self.decoders.append(Decoder(input_sizes[view], latent_dim))

    def forward(self, inputs_list):
        encoded_list = [self.encoders[view](inputs_list[view]) for view in range(self.view_num)]
        decoded_list = [self.decoders[view](encoded_list[view]) for view in range(self.view_num)]
        return encoded_list, decoded_list
    
class PVC_Loss(nn.Module):
    def __init__(self):
        super(PVC_Loss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input_list, encoded_list, decoded_list, permutation_matrix):
        loss_1 = self.mse(input_list[0], decoded_list[0]) + self.mse(input_list[1], decoded_list[1])
        aligned_encoded = torch.mm(permutation_matrix, encoded_list[1]) # (num_samples, latent_dim)
        loss_2 = self.mse(encoded_list[0], aligned_encoded)
        return loss_1 + loss_2  
    
def row2one(P):
    P_sum = P.sum(dim=1, keepdim=True)
    one = torch.ones(1, P.shape[1]).to(P.device)
    return P - (P_sum - 1).mm(one) / P.shape[1]

def col2one(P):
    P_sum = P.sum(dim=0, keepdim=True)
    one = torch.ones(P.shape[0], 1).to(P.device)
    return P - (one).mm(P_sum - 1) / P.shape[0]  

def alignment(encoded_1, encoded_2): # This function is a torch function, so it can be backpropagated.
    # This function is used to find the optimal permutation matrix P according to minimizing <P,D>= sum_i sum_j P[i,j] * D[i,j], where D is the squared euclidean distance matrix.
    # P is a doubly stochastic matrix, where row sum and column sum are all 1, and P[i,j] is the probability of the i-th sample being aligned with the j-th sample. 
    # Permutation matrix is a special case of doubly stochastic matrix, where P[i,j] is either 0 or 1.
    # Input: encoded_1 and encoded_2 are the encoded features of the two views. Output: the optimal doubly stochastic matrix P 
    assert encoded_1.shape == encoded_2.shape, "The shape of encoded_1 and encoded_2 must be the same"
    num_samples = encoded_1.shape[0]; num_features = encoded_1.shape[1]
    
    # Step 1: calculate the squared euclidean distance matrix
    D = torch.pow(encoded_1, 2).sum(1, keepdim=True) + torch.pow(encoded_2, 2).sum(1, keepdim=True).t() - 2 * (encoded_1 @ encoded_2.t()) # squared euclidean distance matrix: (num_samples, num_samples)
    
    # Step 2: initialize the permutation matrix P using the Brute force solution
    row_ind = torch.argmin(D, dim=0) # (num_samples,)
    col_ind = torch.arange(num_samples, device=D.device)
    P = torch.full((num_samples, num_samples), D.max().item(), device=D.device)
    P[row_ind, col_ind] = D[row_ind, col_ind]  # put the minimum value of each column back to (row_ind[j], j), otherwise P[i,j] is the maximum value of D[i,j]
    col_ind = torch.argmin(P, dim=1)
    row_ind = torch.arange(num_samples, device=D.device)
    P = torch.zeros((num_samples, num_samples), device=D.device)
    P[row_ind, col_ind] = 1.0 # set the minimum value of each column to 1, otherwise P[i,j] is 0
    
    # Step 3: optimize the doubly stochastic matrix P according to minimizing <P,D>= sum_i sum_j P[i,j] * D[i,j] using the Dykstra's projection algorithm (or Sinkhorn-Knopp algorithm)
    tau_1 = 30; tau_2 = 10; lr = 0.1
    d = [torch.zeros_like(D) for _ in range(3)] 
    for i in range(tau_1):               
        P = P - lr * D 
        for j in range(tau_2):
                P_initial = P.clone()  
                # 3.1 alternatively update P by d[0], and project to row sum = 1 according minus a constant 1/num_samples
                P = P + d[0]; 
                # Y = P - (P.sum(dim=1, keepdim=True) - 1) / P.shape[1]; # exist floating point difference with row2one
                Y = row2one(P)
                d[0] = P - Y
                # 3.2 alternatively update P by d[1], and project to column sum = 1 according minus a constant 1/num_samples
                P = Y + d[1]; 
                # Y = P - (P.sum(dim=0, keepdim=True) - 1) / P.shape[0]; # exist floating point difference with col2one
                Y = col2one(P)
                d[1] = P - Y
                # 3.3 alternatively update P by d[2], and project to non-negative
                P = Y + d[2]; 
                Y = F.relu(P); 
                d[2] = P - Y
                # 3.4 check if P is converged
                P = Y # update P to the new value
                if (P - P_initial).norm().item() == 0: break # check if P is converged
    return P # optimal doubly stochastic matrix P (row sum and column sum are all 1, and P[i,j] is the probability of the i-th sample being aligned with the j-th sample)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, X_list):
        self.X_list = X_list
        self.view_num = len(X_list)

    def __getitem__(self, index):
        current_x_list = [self.X_list[view][index] for view in range(self.view_num)]
        # permutation_index = np.random.permutation(len(index))
        permutation_index = random.sample(range(len(index)), len(index))
        current_x_list[1] = current_x_list[1][permutation_index] # permutation second view: (num_samples, input_size[1])
        permutation_matrix = np.eye(len(index)).astype('float32') # permutation matrix
        permutation_matrix = permutation_matrix[:, permutation_index] # permutation matrix: (num_samples, num_samples), row: original index, column: permuted index
        return current_x_list, permutation_matrix

    def __len__(self):
        return  self.X_list[0].shape[0]
    
class MyBatchSampler(object):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.n = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.random.permutation(self.n) if self.shuffle else np.arange(self.n)
        if self.drop_last:
            indices = indices[: self.n // self.batch_size * self.batch_size]
        for i in range(0, len(indices), self.batch_size):
            # yield [[i1, i2, ...]] so __getitem__ is called once with the full list (same as DataSampler)
            yield [indices[i : i + self.batch_size].tolist()]

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size
    
class BatchSampler(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n = len(dataset)
        
    def __iter__(self):
        indices = list(self.sampler)
        if self.drop_last:
            indices = indices[: len(indices) // self.batch_size * self.batch_size]
        for i in range(0, len(indices), self.batch_size):
            yield [indices[i : i + self.batch_size]]

    def __len__(self):
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset_name', default="Caltech101-20")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--ae_epochs', default=2000, type=int)
    parser.add_argument('--pretrain_epoch', default=2000, type=int)
    parser.add_argument('--training_epoch', default=200, type=int)
    parser.add_argument('--aligned_ratio', default=0.5, type=float)
    parser.add_argument('--lambda_alignment', default=10, type=float, dest='lambda_alignment')
    parser.add_argument('--latent_dim', default=20, type=int)
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    # 1. Load dataset
    dataset, dims, view_num, data_size, class_num = load_data(args.dataset_name)
    X_list = [dataset.x_1, dataset.x_2]; Y_list = [dataset.y, dataset.y]
    ########### Train data preparation ###########
    # index = np.random.permutation(np.arange(data_size)); num_aligned = int(data_size * args.aligned_ratio)
    index = random.sample(range(data_size), data_size); num_aligned = int(data_size * args.aligned_ratio)
    train_X_list = [X_list[view][index[:num_aligned]] for view in range(view_num)]; train_Y_list = [Y_list[view][index[:num_aligned]] for view in range(view_num)]
    ########### Test data preparation ###########
    # permutation_index = np.random.permutation(np.arange(data_size)) # permutation index for second view
    permutation_index = random.sample(range(data_size), data_size)
    permutation_matrix = np.eye(data_size).astype('float32') # permutation matrix
    permutation_matrix = permutation_matrix[:, permutation_index] # permutation matrix: (num_samples, num_samples), row: original index, column: permuted index
    var_X_list = [torch.from_numpy(X_list[0]).to(device), torch.from_numpy(X_list[1][permutation_index]).to(device)]; var_Y_list = [Y_list[0], Y_list[1][permutation_index]]
    
    # 2. Build model
    model = PVC(view_num, dims, args.latent_dim).to(device)
    criterion = PVC_Loss().to(device)
    
    # 3. Pretraining
    train_dataset = TrainDataset(train_X_list)
    batch_sampler = BatchSampler(train_dataset, shuffle=True, batch_size=args.batch_size, drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=batch_sampler)
    optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model.train()
    for epoch in range(args.ae_epochs + args.pretrain_epoch):
        loss_list = []
        for i, (batch_X_list, batch_P) in enumerate(train_loader):
            batch_X_list = [batch_X_list[view].squeeze(0).to(device) for view in range(view_num)]
            batch_P = batch_P.squeeze(0).to(device)
            encoded_list, decoded_list = model(batch_X_list)
            loss = criterion(batch_X_list, encoded_list, decoded_list, batch_P)
            if(epoch>=args.ae_epochs):
                batch_P_pred = alignment(encoded_list[0], encoded_list[1])
                loss = loss + args.lambda_alignment * F.mse_loss(batch_P_pred, batch_P) # can be backpropagated to optimize the encoded features
            loss_list.append(loss.item())
            optimizer_pretrain.zero_grad()
            loss.backward()
            optimizer_pretrain.step()
        print(f'epoch {epoch} : loss {np.mean(loss_list):.6f}')

    # 4. Training
    with torch.no_grad():
        encoded_list, decoded_list = model(var_X_list)
        P_pred = alignment(encoded_list[0], encoded_list[1])
    optimizer_training = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.train()
    for epoch in range(args.training_epoch):
        encoded_list, decoded_list = model(var_X_list)
        loss = criterion(var_X_list, encoded_list, decoded_list, P_pred)
        optimizer_training.zero_grad()
        loss.backward()
        optimizer_training.step()
        print(f'epoch {epoch} : loss {loss.item():.6f}')
    
    # 5. Evaluation
    model.eval()
    encoded_list, decoded_list = model(var_X_list)
    encoded_1 = encoded_list[0].cpu().detach().numpy()
    encoded_2 = encoded_list[1].cpu().detach().numpy()
    P_pred = P_pred.cpu().detach().numpy()
    aligned_encoded_2 = np.dot(P_pred, encoded_2)
    encoded_all = np.concatenate((encoded_1, aligned_encoded_2), axis=1)
    kmeans = KMeans(n_clusters=class_num, n_init=20)
    y_preds = kmeans.fit_predict(encoded_all)
    nmi, ari, acc, pur = evaluate(var_Y_list[0], y_preds)
    print(f'NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, PUR: {pur:.4f}')
