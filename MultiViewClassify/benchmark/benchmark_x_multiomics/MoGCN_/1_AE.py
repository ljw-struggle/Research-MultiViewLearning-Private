import torch
import torch.nn as nn
import torch.utils.data as Data
import os, argparse, pandas as pd, numpy as np
from tqdm import tqdm   
from matplotlib import pyplot as plt

class MMAE(nn.Module):
    def __init__(self, in_feas_dim, latent_dim, a=0.4, b=0.3, c=0.3):
        '''
        :param in_feas_dim: a list, input dims of omics data
        :param latent_dim: dim of latent layer
        :param a, b, c: weight of omics data type 1, 2, 3
        '''
        super(MMAE, self).__init__()
        self.a = a; self.b = b; self.c = c
        self.in_feas = in_feas_dim; self.latent = latent_dim
        self.encoder_omics_1 = nn.Sequential(nn.Linear(self.in_feas[0], self.latent), nn.BatchNorm1d(self.latent), nn.Sigmoid()); self.decoder_omics_1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        self.encoder_omics_2 = nn.Sequential(nn.Linear(self.in_feas[1], self.latent), nn.BatchNorm1d(self.latent), nn.Sigmoid()); self.decoder_omics_2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        self.encoder_omics_3 = nn.Sequential(nn.Linear(self.in_feas[2], self.latent), nn.BatchNorm1d(self.latent), nn.Sigmoid()); self.decoder_omics_3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))
        self.init_weights()
    
    def init_weights(self):
        # [torch.nn.init.normal_(param, mean=0, std=0.1) if name.endswith('weight') else torch.nn.init.constant_(param, val=0) if name.endswith('bias') else None for name, param in self.named_parameters()]
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.normal_(param, mean=0, std=0.1)
            elif name.endswith('bias'):
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_1, omics_2, omics_3):
        encoded_omics_1 = self.encoder_omics_1(omics_1); encoded_omics_2 = self.encoder_omics_2(omics_2); encoded_omics_3 = self.encoder_omics_3(omics_3)
        latent_data = torch.mul(encoded_omics_1, self.a) + torch.mul(encoded_omics_2, self.b) + torch.mul(encoded_omics_3, self.c)
        decoded_omics_1 = self.decoder_omics_1(latent_data); decoded_omics_2 = self.decoder_omics_2(latent_data); decoded_omics_3 = self.decoder_omics_3(latent_data)
        return latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3
           
def prepare_data(data_dir):
    data_train_list = []
    data_test_list = []
    for i in range(1, 4): # num_view = 3
        data_train = np.loadtxt(os.path.join(data_dir, str(i) + "_tr.csv"), delimiter=',')
        data_test = np.loadtxt(os.path.join(data_dir, str(i) + "_te.csv"), delimiter=',')
        data_train_min = np.min(data_train, axis=0, keepdims=True)
        data_train_max = np.max(data_train, axis=0, keepdims=True)
        data_train = (data_train - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_test = (data_test - data_train_min)/(data_train_max - data_train_min + 1e-10)
        data_train_list.append(data_train.astype(float))
        data_test_list.append(data_test.astype(float))
    label_train = np.loadtxt(os.path.join(data_dir, "labels_tr.csv"), delimiter=',').astype(int)
    label_test = np.loadtxt(os.path.join(data_dir, "labels_te.csv"), delimiter=',').astype(int)
    return data_train_list, data_test_list, label_train, label_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='./data/BRCA/', help='The data dir.')
    parser.add_argument('-o', '--output_dir', default='./result/BRCA/', help='The output dir.')
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('-bs', '--batchsize', default=32, type=int, help='Training batchszie.')
    parser.add_argument('-lr', '--learningrate', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('-e', '--epoch', default=100, type=int, help='Training epochs.')
    parser.add_argument('-l', '--latent', default=100, type=int, help='The latent layer dim.')
    parser.add_argument('-a', '--a', type=float, default=0.6, help='[0,1], float, weight for the first omics data')
    parser.add_argument('-b', '--b', type=float, default=0.1, help='[0,1], float, weight for the second omics data.')
    parser.add_argument('-c', '--c', type=float, default=0.3, help='[0,1], float, weight for the third omics data.')
    parser.add_argument('-n', '--topn', default=100, type=int, help='Extract top N features every 10 epochs.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert args.a + args.b + args.c == 1.0, 'The sum of weights must be 1.'
    data_train_list, data_test_list, label_train, label_test = prepare_data(args.data_dir)
    data_list = [np.concatenate([data_train_list[i], data_test_list[i]], axis=0) for i in range(3)]
    label_list = [np.concatenate([label_train, label_test], axis=0)]
    dataset = Data.TensorDataset(torch.FloatTensor(data_list[0]), torch.FloatTensor(data_list[1]), torch.FloatTensor(data_list[2]), torch.FloatTensor(label_list[0]))
    train_loader = Data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

    print('Training model...')
    mmae = MMAE([data_list[i].shape[1] for i in range(3)], latent_dim=args.latent, a=args.a, b=args.b, c=args.c)
    mmae.to(device)
    
    from thop import profile, clever_format
    params = sum([sum(p.numel() for p in mmae.parameters())])
    flops, params = profile(mmae.to(device), inputs=(torch.FloatTensor(data_list[0]).to(device), torch.FloatTensor(data_list[1]).to(device), torch.FloatTensor(data_list[2]).to(device)), verbose=False) # return MACs and params; flops = 2 * MACs
    flops, params = clever_format([flops*2, params], "%.3f")
    print(f'Input data shape: {data_list[0].shape}, {data_list[1].shape}, {data_list[2].shape}')
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"Params by manual calculation: {sum(p.numel() for p in mmae.parameters())}")
    print('Begin training model...')
    exit()
    mmae.train()
    optimizer = torch.optim.Adam(mmae.parameters(), lr=args.learningrate)
    loss_fn = nn.MSELoss()
    loss_ls = []
    for epoch in range(args.epoch):
        train_loss_sum = 0.0
        for (omics_1, omics_2, omics_3, label) in train_loader:
            omics_1 = omics_1.to(device); omics_2 = omics_2.to(device); omics_3 = omics_3.to(device); label = label.to(device)
            latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = mmae.forward(omics_1, omics_2, omics_3)
            loss = mmae.a * loss_fn(decoded_omics_1, omics_1) + mmae.b * loss_fn(decoded_omics_2, omics_2) + mmae.c * loss_fn(decoded_omics_3, omics_3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        loss_ls.append(train_loss_sum)
        print('epoch: %d | loss: %.4f' % (epoch + 1, train_loss_sum))
        #save the model every 10 epochs, used for feature extraction
        if (epoch+1) % 10 ==0:
            torch.save(mmae, os.path.join(args.output_dir, 'ae_model_{}.pkl'.format(epoch+1)))
    plt.plot([i + 1 for i in range(args.epoch)], loss_ls)
    plt.xlabel('epochs'); plt.ylabel('loss'); plt.savefig(os.path.join(args.output_dir, 'AE_train_loss.png'))
    # torch.save(mmae, os.path.join(args.output_dir, 'ae_model.pkl'))
    # mmae = torch.load(os.path.join(args.output_dir, 'ae_model.pkl'))
    mmae.eval()
    omics_1 = data_list[0]; omics_2 = data_list[1]; omics_3 = data_list[2]
    omics_1 = torch.FloatTensor(omics_1); omics_2 = torch.FloatTensor(omics_2); omics_3 = torch.FloatTensor(omics_3)
    omics_1 = omics_1.to(device); omics_2 = omics_2.to(device); omics_3 = omics_3.to(device)
    latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = mmae.forward(omics_1, omics_2, omics_3)
    latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
    latent_df.to_csv(os.path.join(args.data_dir, 'latent_data.csv'), header=True, index=False)
    print('Extract features...')
    feas_omics_1 = pd.read_csv(os.path.join(args.data_dir, '1_featname.csv'), header=None).iloc[:, 0].tolist()
    feas_omics_2 = pd.read_csv(os.path.join(args.data_dir, '2_featname.csv'), header=None).iloc[:, 0].tolist()
    feas_omics_3 = pd.read_csv(os.path.join(args.data_dir, '3_featname.csv'), header=None).iloc[:, 0].tolist()
    std_omics_1 = omics_1.std(axis=0).detach().cpu().numpy()
    std_omics_2 = omics_2.std(axis=0).detach().cpu().numpy()
    std_omics_3 = omics_3.std(axis=0).detach().cpu().numpy()
    # record top N features every 10 epochs
    topn_omics_1 = pd.DataFrame()
    topn_omics_2 = pd.DataFrame()
    topn_omics_3 = pd.DataFrame()
    epoch_ls = list(range(10, args.epoch+10, 10))
    for epoch in tqdm(epoch_ls):
        mmae = torch.load(os.path.join(args.output_dir, 'ae_model_{}.pkl'.format(epoch)), weights_only=False)
        model_dict = mmae.state_dict()
        # get the absolute value of weights, the shape of matrix is (n_features, latent_layer_dim). manual = x @ fc.weight.t()
        weight_omics1 = np.abs(model_dict['encoder_omics_1.0.weight'].detach().cpu().numpy().T) # (n_features, latent_layer_dim)
        weight_omics2 = np.abs(model_dict['encoder_omics_2.0.weight'].detach().cpu().numpy().T) # (n_features, latent_layer_dim)
        weight_omics3 = np.abs(model_dict['encoder_omics_3.0.weight'].detach().cpu().numpy().T) # (n_features, latent_layer_dim)
        weight_omics1_df = pd.DataFrame(weight_omics1, index=feas_omics_1) # (n_features, latent_layer_dim)
        weight_omics2_df = pd.DataFrame(weight_omics2, index=feas_omics_2) # (n_features, latent_layer_dim)
        weight_omics3_df = pd.DataFrame(weight_omics3, index=feas_omics_3) # (n_features, latent_layer_dim)
        # calculate the weight sum of each feature --> sum of each row
        weight_omics1_df['Weight_sum'] = weight_omics1_df.apply(lambda x:x.sum(), axis=1) # (n_features,)
        weight_omics2_df['Weight_sum'] = weight_omics2_df.apply(lambda x:x.sum(), axis=1) # (n_features,)
        weight_omics3_df['Weight_sum'] = weight_omics3_df.apply(lambda x:x.sum(), axis=1) # (n_features,)
        weight_omics1_df['Std'] = std_omics_1 # (n_features,)
        weight_omics2_df['Std'] = std_omics_2 # (n_features,)
        weight_omics3_df['Std'] = std_omics_3 # (n_features,)
        # importance = Weight * Std
        weight_omics1_df['Importance'] = weight_omics1_df['Weight_sum'] * weight_omics1_df['Std'] # (n_features,)
        weight_omics2_df['Importance'] = weight_omics2_df['Weight_sum'] * weight_omics2_df['Std'] # (n_features,)
        weight_omics3_df['Importance'] = weight_omics3_df['Weight_sum'] * weight_omics3_df['Std'] # (n_features,)
        # select top N features
        fea_omics_1_top = weight_omics1_df.nlargest(args.topn, 'Importance').index.tolist() # (topN,)
        fea_omics_2_top = weight_omics2_df.nlargest(args.topn, 'Importance').index.tolist() # (topN,)
        fea_omics_3_top = weight_omics3_df.nlargest(args.topn, 'Importance').index.tolist() # (topN,)
        #save top N features in a dataframe
        col_name = 'epoch_'+str(epoch) # (topN,)
        topn_omics_1[col_name] = fea_omics_1_top
        topn_omics_2[col_name] = fea_omics_2_top
        topn_omics_3[col_name] = fea_omics_3_top
    #all of top N features
    topn_omics_1.to_csv(os.path.join(args.output_dir, 'topn_omics_1.csv'), header=True, index=False)
    topn_omics_2.to_csv(os.path.join(args.output_dir, 'topn_omics_2.csv'), header=True, index=False)
    topn_omics_3.to_csv(os.path.join(args.output_dir, 'topn_omics_3.csv'), header=True, index=False)
    print('Success! Results can be seen in result file')
    