import os, h5py, scipy, random, argparse
import pandas as pd, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_data(data_path):
    data = h5py.File(data_path, 'r')['matrix/data']
    data = np.array(data).transpose() # shape: (cell_num, gene_num)
    return data

def read_label(label_path):
    label_df = pd.read_csv(label_path, header=0, index_col=0)
    label_codes = pd.Categorical(label_df['x']).codes # categorical data to numerical data, shape: (cell_num,)
    label_categories = pd.Categorical(label_df['x']).categories # get the categories of the categorical data, shape: (cell_type_num,)
    return label_codes, label_categories

def preprocess_data(data): # log2 normalization, z-score normalization
    temp = torch.sum(data, 1, keepdim=True) # shape: (n,1)
    temp = temp/torch.mean(temp) # shape: (n,1)
    data = torch.log2(data/temp + 1) # shape: (n, m)
    temp_mean = torch.mean(data, 1, keepdim=True) # shape: (n,1)
    temp_std = torch.std(data, 1, keepdim=True) # shape: (n,1)
    data = (data - temp_mean)/temp_std # shape: (n, m)
    return data

def preprocess_data(data): # normalize total UMI counts to the mean of all rows, log2 normalization, z-score normalization
    temp = torch.sum(data, 1, keepdim=True) # shape: (n,1), calculate the sum of each row
    temp = temp/torch.mean(temp) # shape: (n,1), sum of each row / mean(sum of all rows)
    data = data/temp # shape: (n, m), make the sum of each row to be the mean of all rows
    data = torch.log2(data + 1) # shape: (n, m), log2 transformation
    temp_mean = torch.mean(data, 1, keepdim=True) # shape: (n,1), calculate the mean of each row (every row is same)
    temp_std = torch.std(data, 1, keepdim=True) # shape: (n,1), calculate the standard deviation of each row
    data = (data - temp_mean)/temp_std # shape: (n, m), z-score normalization
    return data

class MMDataset(Dataset):
    def __init__(self, data, label):
        self.data = data; self.label = label

    def __getitem__(self, index):
        return {'data': self.data[index,:], 'label': self.label[index]}

    def __len__(self):
        return len(self.data)

def get_simulated_data_from_sampling(model, dl):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    data = []; label = []; decodings = []
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            x = torch.tensor(batch_sample['data'], dtype=torch.float32).reshape(x.size(0),-1)
            label = torch.tensor(batch_sample['label'], dtype=torch.long)
            decoding, x_cell_type, mu, var = model(x.to(device))
            data.append(x); label.append(label); decodings.append(decoding.cpu().numpy())
    return torch.cat(decodings, dim=0), torch.cat(label,dim=0), torch.cat(data,dim=0)

class AverageMeter(object): # Computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name; self.fmt = fmt; 
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

    def __str__(self):
        return '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'.format(**self.__dict__)

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes=17, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes; self.epsilon = epsilon; self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets): # inputs: (n, num_classes), targets: (n,)
        log_probs = self.logsoftmax(inputs) # shape: (n, num_classes)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1) # shape: (n, num_classes)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes # do label smoothing, shape: (n, num_classes)
        loss = (- targets * log_probs).mean(0).sum() # shape: (1,)
        return loss

def accuracy(output, target, topk=(1,)): # Computes the accuracy over the k top predictions for the specified values of k, returns the topk accuracy list
    _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True) # shape: (n, max(topk))
    pred = pred.t(); target = target.view(1, -1).expand_as(pred); correct = pred.eq(target) # shape: (max(topk), n)
    res = []
    for k in topk:
        correct_k = correct[:k].sum().float(); res.append(correct_k / target.size(0) * 100.0) # shape: (1,)
    return res

##############################################################################################################################################################################

class LinBnDrop(nn.Sequential): # Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers
    def __init__(self, n_in, n_out, bn=True, p=0, act=None, linear_first=True):
        linear = [nn.Linear(n_in, n_out, bias=not bn)]; linear.append(act) if act is not None else None
        norm = [nn.BatchNorm1d(n_out if linear_first else n_in)] if bn else []; norm.append(nn.Dropout(p)) if p != 0 else None
        layers = linear + norm if linear_first else norm + linear
        super().__init__(*layers)
        
class AutoEncoder(nn.Module): # Autoencoder for RNA data
    def __init__(self, nfeatures_rna=0, hidden_rna=185, z_dim=20, classify_dim=17):
        super().__init__()
        self.encoder_modality = LinBnDrop(nfeatures_rna, hidden_rna, p=0.2, act=nn.ReLU())
        self.encoder = LinBnDrop(hidden_rna, z_dim, p=0.2, act=nn.ReLU())
        self.weights_modality = nn.Parameter(torch.rand((1, nfeatures_rna)) * 0.001, requires_grad=True)
        self.fc_mu = LinBnDrop(z_dim,z_dim, p=0.2)
        self.fc_var = LinBnDrop(z_dim,z_dim, p=0.2)
        self.decoder = LinBnDrop(z_dim, nfeatures_rna, act=nn.ReLU())
        self.classify = nn.Linear(z_dim, classify_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        x = self.encoder(self.encoder_modality(x * self.weights_modality))
        mu = self.fc_mu(x); var = self.fc_var(x); x = self.reparameterize(mu, var)
        x_recon = self.decoder(x); x_class = self.classify(x)
        return x_recon, x_class, mu, var

class AutoEncoder_Two_Modality(nn.Module): # CITE-seq or SHARE-seq
    def __init__(self, nfeatures_modality_1=0, nfeatures_modality_2=0,  hidden_modality_1=185,  hidden_modality_2=15, z_dim=20, classify_dim=17):
        super().__init__()
        self.nfeatures_modality_1 = nfeatures_modality_1; self.nfeatures_modality_2 = nfeatures_modality_2
        self.weights_modality_1 = nn.Parameter(torch.rand((1, nfeatures_modality_1)) * 0.001, requires_grad=True)
        self.weights_modality_2 = nn.Parameter(torch.rand((1, nfeatures_modality_2)) * 0.001, requires_grad=True)
        self.encoder_modality_1 = LinBnDrop(nfeatures_modality_1, hidden_modality_1, p=0.2, act=nn.ReLU())
        self.encoder_modality_2 = LinBnDrop(nfeatures_modality_2, hidden_modality_2, p=0.2, act=nn.ReLU())
        self.encoder = LinBnDrop(hidden_modality_1 + hidden_modality_2, z_dim,  p=0.2, act=nn.ReLU())
        self.fc_mu = LinBnDrop(z_dim,z_dim, p=0.2); self.fc_var = LinBnDrop(z_dim,z_dim, p=0.2)
        self.classify = nn.Linear(z_dim, classify_dim)
        self.decoder_modality_1 = LinBnDrop(z_dim, nfeatures_modality_1, act=nn.ReLU())
        self.decoder_modality_2 = LinBnDrop(z_dim, nfeatures_modality_2,  act=nn.ReLU())
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    def forward(self, x):
        x_modality_1 = self.encoder_modality_1(x[:, :self.nfeatures_modality_1]*self.weights_modality_1)
        x_modality_2 = self.encoder_modality_2(x[:, self.nfeatures_modality_1:]*self.weights_modality_2)
        x = torch.cat([x_modality_1, x_modality_2], 1)
        x = self.encoder(x); mu = self.fc_mu(x); var = self.fc_var(x); x = self.reparameterize(mu, var)
        x_class = self.classify(x)        
        x_modality_1_recon = self.decoder_modality_1(x)
        x_modality_2_recon = self.decoder_modality_2(x)
        x_recon = torch.cat((x_modality_1_recon, x_modality_2_recon), 1)
        return x_recon, x_class, mu, var

class AutoEncoder_Three_Modality(nn.Module): # TEA-seq
    def __init__(self, nfeatures_modality_1=10000, nfeatures_modality_2=30, nfeatures_modality_3=10000, hidden_modality_1=185, hidden_modality_2=30,  hidden_modality_3=185, z_dim=100, classify_dim=17):
        super().__init__()
        self.nfeatures_modality_1 = nfeatures_modality_1; self.nfeatures_modality_2 = nfeatures_modality_2; self.nfeatures_modality_3 = nfeatures_modality_3
        self.weights_modality_1 = nn.Parameter(torch.rand((1, nfeatures_modality_1)) * 0.001, requires_grad=True)
        self.weights_modality_2 = nn.Parameter(torch.rand((1, nfeatures_modality_2)) * 0.001, requires_grad=True)
        self.weights_modality_3 = nn.Parameter(torch.rand((1, nfeatures_modality_3)) * 0.001, requires_grad=True)
        self.encoder_modality_1 = LinBnDrop(nfeatures_modality_1, hidden_modality_1, p=0.2, act=nn.ReLU())
        self.encoder_modality_2 = LinBnDrop(nfeatures_modality_2, hidden_modality_2, p=0.2, act=nn.ReLU())
        self.encoder_modality_3 = LinBnDrop(nfeatures_modality_3, hidden_modality_3, p=0.2, act=nn.ReLU())
        self.encoder = LinBnDrop(hidden_modality_1 + hidden_modality_2 + hidden_modality_3, z_dim,  p=0.2, act=nn.ReLU())
        self.fc_mu = LinBnDrop(z_dim, z_dim); self.fc_var = LinBnDrop(z_dim, z_dim)
        self.classify = nn.Linear(z_dim, classify_dim)
        self.decoder_modality_1 = LinBnDrop(z_dim, nfeatures_modality_1, act=nn.ReLU())
        self.decoder_modality_2 = LinBnDrop(z_dim, nfeatures_modality_2, act=nn.ReLU())
        self.decoder_modality_3 = LinBnDrop(z_dim, nfeatures_modality_3, act=nn.ReLU())
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x_modality_1 = self.encoder_modality_1(x[:, :self.nfeatures_modality_1]*self.weights_modality_1)
        x_modality_2 = self.encoder_modality_2(x[:, self.nfeatures_modality_1:(self.nfeatures_modality_1 + self.nfeatures_modality_2)]*self.weights_modality_2)
        x_modality_3 = self.encoder_modality_3(x[:, (self.nfeatures_modality_1 + self.nfeatures_modality_2):]*self.weights_modality_3)
        x = torch.cat([x_modality_1, x_modality_2, x_modality_3], 1)
        x = self.encoder(x); mu = self.fc_mu(x); var = self.fc_var(x); x = self.reparameterize(mu, var)
        x_class = self.classify(x)
        x_modality_1_recon = self.decoder_modality_1(x)
        x_modality_2_recon = self.decoder_modality_2(x)
        x_modality_3_recon = self.decoder_modality_3(x)
        x_recon = torch.cat((x_modality_1_recon, x_modality_2_recon, x_modality_3_recon), 1)
        return x_recon, x_class, mu, var
    
##############################################################################################################################################################################

def train_model(model, train_dl, test_dl, lr, epochs, classify_dim=17, save_path = '', feature_num=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss().to(device)
    criterion_smooth_cty = CrossEntropyLabelSmooth().to(device)
    criterion_KLD = lambda mu, logvar: -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
    best_top_1_acc=0 # record the best top 1 accuracy on the test data
    train_each_celltype_num = [0 for i in range(classify_dim)] # record the number of each cell type in the training data only for the first epoch
    best_each_celltype_top_1_acc = [0 for i in range(classify_dim)]; best_each_celltype_num = [0 for i in range(classify_dim)] # record the best top 1 accuracy of each cell type on the test data
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        train_top_1_acc = AverageMeter('Acc@1', ':6.2f')
        for i, batch_sample in enumerate(train_dl):
            optimizer.zero_grad()
            x = torch.tensor(batch_sample['data'], dtype=torch.float32).reshape(x.size(0),-1)
            train_label = torch.tensor(batch_sample['label'], dtype=torch.long)
            x_prime, x_label, mu, var = model(x.to(device))
            loss_1 = criterion(x_prime, x.to(device)) + 1/feature_num*(criterion_KLD(mu,var)) # simulation loss
            loss_2 = criterion_smooth_cty(x_label, train_label.to(device)) # classification loss
            loss = 0.9 * loss_1 + 0.1 * loss_2 # sum up the loss together
            loss.backward()
            optimizer.step()
            train_top_1_acc.update(accuracy(x_label, train_label, topk=(1, ))[0], 1)
            if epoch == 1:
                for j in range(classify_dim):
                    train_each_celltype_num[j] = train_each_celltype_num[j] + len(train_label[train_label==j])                        
        model = model.eval()
        test_top_1_acc = AverageMeter('Acc@1', ':6.2f')
        each_celltype_top_1_acc = [AverageMeter('Acc@1', ':6.2f') for i in range(classify_dim)]; each_celltype_num = [0 for i in range(classify_dim)]
        if test_dl != 'NULL':
            with torch.no_grad():
                for i, batch_sample in enumerate(test_dl):
                    x = torch.tensor(batch_sample['data'], dtype=torch.float32).reshape(x.size(0), -1)
                    test_label = torch.tensor(batch_sample['label'], dtype=torch.long)
                    x_recon, x_label, mu, var = model(x.to(device))
                    test_top_1_acc.update(accuracy(x_label, test_label, topk=(1, ))[0], 1)
                    for j in range(classify_dim):
                        each_celltype_top_1_acc[j].update(accuracy(x_label[test_label==j,:], test_label[test_label==j], topk=(1, ))[0], 1)
                        each_celltype_num[j] = each_celltype_num[j] + len(test_label[test_label==j])
        # if test_top_1.avg > best_top_1_acc: # save the best model according to the test data
        #     best_top_1_acc = test_top_1.avg
        if epoch==epochs: # save the model according to the last epoch
            for j in range(classify_dim):
                best_each_celltype_top_1_acc[j] = each_celltype_top_1_acc[j].avg; best_each_celltype_num[j] = each_celltype_num[j]
                print('cell type : ', j, ' prec :', best_each_celltype_top_1_acc[j], ' number:', best_each_celltype_num[j], ' train_cell_type_num:', train_each_celltype_num[j])
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join(save_path,'model_best.pth.tar'))

    return model, best_each_celltype_top_1_acc, best_each_celltype_num, train_each_celltype_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Matilda')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--augmentation', type=bool, default= True, help='if augmentation or not')
    parser.add_argument('--rna', metavar='DIR', default='NULL', help='path to train rna data')
    parser.add_argument('--adt', metavar='DIR', default='NULL', help='path to train adt data')
    parser.add_argument('--atac', metavar='DIR', default='NULL', help='path to train atac data')
    parser.add_argument('--label', metavar='DIR', default='NULL', help='path to train cell type label')
    parser.add_argument('--z_dim', type=int, default=100, help='the number of neurons in latent space')
    parser.add_argument('--hidden_rna', type=int, default=185, help='the number of neurons for RNA layer')
    parser.add_argument('--hidden_adt', type=int, default=30, help='the number of neurons for ADT layer')
    parser.add_argument('--hidden_atac', type=int, default=185, help='the number of neurons for ATAC layer')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
    parser.add_argument('--lr', type=float, default=0.02, help='init learning rate')
    args = parser.parse_args()
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.adt != 'NULL' and args.atac != 'NULL':
        train_rna_data = read_data(args.rna); train_adt_data = read_data(args.adt); train_atac_data = read_data(args.atac); train_label, label_name_list = read_label(args.label); classify_dim = len(label_name_list)
        nfeatures_rna = train_rna_data.shape[1]; nfeatures_adt = train_adt_data.shape[1]; nfeatures_atac = train_atac_data.shape[1]; feature_num = nfeatures_rna + nfeatures_adt + nfeatures_atac
        mode = 'TEAseq'; train_data = torch.cat((preprocess_data(train_rna_data), preprocess_data(train_adt_data), preprocess_data(train_atac_data)), 1) 
        train_dl = DataLoader(MMDataset(train_data, train_label), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    if args.adt == 'NULL' and args.atac != 'NULL':
        train_rna_data = read_data(args.rna); train_atac_data = read_data(args.atac); train_label, label_name_list = read_label(args.label); classify_dim = len(label_name_list)
        nfeatures_rna = train_rna_data.shape[1]; nfeatures_atac = train_atac_data.shape[1]; feature_num = nfeatures_rna + nfeatures_atac
        mode = 'SHAREseq'; train_data = torch.cat((preprocess_data(train_rna_data), preprocess_data(train_atac_data)), 1)
        train_dl = DataLoader(MMDataset(train_data, train_label), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    if args.adt != 'NULL' and args.atac == 'NULL':
        train_rna_data = read_data(args.rna); train_adt_data = read_data(args.adt); train_label, label_name_list = read_label(args.label); classify_dim = len(label_name_list)
        nfeatures_rna = train_rna_data.shape[1]; nfeatures_adt = train_adt_data.shape[1]; feature_num = nfeatures_rna + nfeatures_adt
        mode = 'CITEseq'; train_data = torch.cat((preprocess_data(train_rna_data), preprocess_data(train_adt_data)), 1)
        train_dl = DataLoader(MMDataset(train_data, train_label), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    if args.adt == 'NULL' and args.atac == 'NULL':
        train_rna_data = read_data(args.rna); train_label, label_name_list = read_label(args.label); classify_dim = len(label_name_list)
        nfeatures_rna = train_rna_data.shape[1]; feature_num = nfeatures_rna
        mode = 'RNAseq'; train_data = preprocess_data(train_rna_data)
        train_dl = DataLoader(MMDataset(train_data, train_label), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    test_dl = 'NULL'; print('The dataset is', mode); output_v = []
    model_save_path = './result/trained_model/{}/'.format(mode); model_save_path_1_stage = './result/trained_model/{}/simulation_'.format(mode); save_fs_eachcell = '../result/marker/{}/'.format(mode)
    os.makedirs(model_save_path, exist_ok=True); os.makedirs(save_fs_eachcell, exist_ok=True); os.makedirs(model_save_path_1_stage, exist_ok=True)
    if mode == 'RNAseq': model = AutoEncoder(nfeatures_rna, args.hidden_rna, args.z_dim, classify_dim)
    if mode == 'CITEseq': model = AutoEncoder_Two_Modality(nfeatures_rna, nfeatures_adt, args.hidden_rna, args.hidden_adt, args.z_dim, classify_dim)
    if mode == 'SHAREseq': model = AutoEncoder_Two_Modality(nfeatures_rna, nfeatures_atac, args.hidden_rna, args.hidden_atac, args.z_dim, classify_dim)
    if mode == 'TEAseq': model = AutoEncoder_Three_Modality(nfeatures_rna, nfeatures_adt, nfeatures_atac, args.hidden_rna, args.hidden_adt, args.hidden_atac, args.z_dim, classify_dim)
    model = model.to(device) # model = nn.DataParallel(model).to(device) for multi gpu
    # Stage 1: train the model with original data
    model, acc1, num1, train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=args.epochs, classify_dim=classify_dim, save_path=model_save_path, feature_num=feature_num)
    # Stage 2: train the model with augmented data. 
    if args.augmentation == True:
        stage_1_list = []
        for i in np.arange(0, classify_dim):
            stage_1_list.append([i, train_num[i]])
        stage_1_df = pd.DataFrame(stage_1_list) 
        train_median = np.sort(train_num)[int(classify_dim/2)-1] if classify_dim%2==0 else np.median(train_num)
        median_anchor = stage_1_df[stage_1_df[1] == train_median][0]; train_major = stage_1_df[stage_1_df[1] > train_median]; train_minor = stage_1_df[stage_1_df[1] < train_median]
        anchor_fold = np.array((train_median)/(train_minor[:][1]))
        minor_anchor_cts = train_minor[0].to_numpy(); major_anchor_cts = train_major[0].to_numpy()
        index = (train_label == int(np.array(median_anchor))).nonzero(as_tuple=True)[0]; anchor_data = train_data[index.tolist(),:]; anchor_label = train_label[index.tolist()]
        new_data = anchor_data; new_label = anchor_label
        for j, anchor in enumerate(major_anchor_cts): # random downsample major cell types
            anchor_num = np.array(train_major[1])[j]; ds_index = random.sample(range(anchor_num),int(train_median))
            index = (train_label == anchor).nonzero(as_tuple=True)[0]
            anchor_data = train_data[index.tolist(),:]; anchor_label = train_label[index.tolist()]; anchor_data = anchor_data[ds_index,:]; anchor_label = anchor_label[ds_index]
            new_data = torch.cat((new_data,anchor_data),0); new_label = torch.cat((new_label,anchor_label.to(device)),0)
        for j, anchor in enumerate(minor_anchor_cts): # augment for minor cell types
            aug_fold = int((anchor_fold[j])); remaining_cell = int(train_median - (int(anchor_fold[j]))*np.array(train_minor[1])[j])
            index = (train_label == anchor).nonzero(as_tuple=True)[0]; anchor_data = train_data[index.tolist(),:]; anchor_label = train_label[index.tolist()]
            anchor_transfomr_dataset = MMDataset(anchor_data, anchor_label); anchor_dl = DataLoader(anchor_transfomr_dataset, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
            reconstructed_data, reconstructed_label, real_data = get_simulated_data_from_sampling(model, anchor_dl)
            reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data); reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data); reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
            new_data = torch.cat((new_data,reconstructed_data),0); new_label = torch.cat((new_label, reconstructed_label),0)
            for i in range(aug_fold-1):
                reconstructed_data, reconstructed_label, real_data = get_simulated_data_from_sampling(model, anchor_dl)
                reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data); reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data); reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
                new_data = torch.cat((new_data, reconstructed_data),0); new_label = torch.cat((new_label, reconstructed_label.to(device)),0)
            reconstructed_data, reconstructed_label, real_data = get_simulated_data_from_sampling(model, anchor_dl)
            reconstructed_data[reconstructed_data>torch.max(real_data)]=torch.max(real_data); reconstructed_data[reconstructed_data<torch.min(real_data)]=torch.min(real_data); reconstructed_data[torch.isnan(reconstructed_data)]=torch.max(real_data)
            N = range(np.array(train_minor[1])[j]); ds_index = random.sample(N, remaining_cell)
            reconstructed_data = reconstructed_data[ds_index,:]; reconstructed_label = reconstructed_label[ds_index]
            new_data = torch.cat((new_data, reconstructed_data),0); new_label = torch.cat((new_label, reconstructed_label.to(device)),0)
    train_dl = DataLoader(MMDataset(new_data, new_label), batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=False)
    # train the model with augmented data based on the model trained with original data and original learning rate
    checkpoint = torch.load(os.path.join(model_save_path, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model, acc2, num1, train_num = train_model(model, train_dl, test_dl, lr=args.lr, epochs=int(args.epochs/2), classify_dim=classify_dim, save_path=model_save_path, feature_num=feature_num)
    # train the model with augmented data based on the model trained with augmented data and 1/10 learning rate
    checkpoint = torch.load(os.path.join(model_save_path, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model, acc2, num1, train_num = train_model(model, train_dl, test_dl, lr=args.lr/10, epochs=int(args.epochs/2), classify_dim=classify_dim, save_path=model_save_path, feature_num=feature_num)