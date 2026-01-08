
import os, random, argparse, json
import scipy as sp, numpy as np, pandas as pd, anndata as ad, scanpy as sc
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from itertools import repeat
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.utils.data as Data
import warnings
warnings.filterwarnings("ignore")
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def compute_metrics(y_true, y_pred):
    y_true = y_true[y_pred.columns]
    y_true = pd.melt(y_true)['value']
    y_pred = pd.melt(y_pred)['value']
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    PCC = np.corrcoef(y_true, y_pred)[0, 1] # Pearson correlation coefficient (using scipy.stats.pearsonr: pearsonr(y_true, y_pred)[0])
    CCC = (2 * PCC * np.std(y_true) * np.std(y_pred)) / (np.var(y_true) + np.var(y_pred) + (np.mean(y_true) - np.mean(y_pred))**2)
    return RMSE, PCC, CCC
    
def save_loss_plot(loss_logger, loss_type, result_dir, output_prex):
    plt.figure(figsize=(18, 10))
    for i in range(len(loss_type)):
        plt.subplot(2, 3, i+1)
        plt.plot(loss_logger[loss_type[i]], 'b-')
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel(loss_type[i])
        plt.title(loss_type[i], x = 0.5, y = 0.5)
    plt.savefig(os.path.join(result_dir, output_prex +'.png'))
    
def save_tsne_plot(ann_data, result_dir, output_prex):
    os.makedirs(result_dir, exist_ok=True)
    sc.tl.pca(ann_data, svd_solver='arpack')
    sc.pp.neighbors(ann_data, n_neighbors=10, n_pcs=20)
    sc.tl.tsne(ann_data)
    with plt.rc_context({'figure.figsize': (8, 8)}):
        sc.pl.tsne(ann_data, color=list(ann_data.obs.columns), color_map='viridis', frameon=False, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, output_prex + "_tsne_plot.jpg"))
    ann_data.write(os.path.join(result_dir, output_prex + ".h5ad"))
    
def save_pred_plot(target_preds, ground_truth, result_dir, output_prex):
    plt.figure(figsize = (5*(target_preds.shape[1] + 1), 5))
    # Plot results for all cell types
    RMSE, PCC, CCC = compute_metrics(target_preds, ground_truth)
    eval_metric = [CCC, RMSE, PCC]
    x = pd.melt(target_preds)['value']
    y = pd.melt(ground_truth)['value']
    plt.subplot(1, target_preds.shape[1] + 1, 1)
    plt.scatter(x, y, s=2)
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), 'r--')
    plt.xlim(0, max(y))
    plt.ylim(0, max(y))
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('All samples')
    plt.text(0.05, max(y)-0.05, f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$", fontsize=8, verticalalignment='top')
    # Plot results for each cell type
    celltypes = list(target_preds.columns)
    for i in range(ground_truth.shape[1]):
        RMSE, PCC, CCC = compute_metrics(target_preds, ground_truth)
        eval_metric = [CCC, RMSE, PCC]
        x = target_preds[celltypes[i]]
        y = ground_truth[celltypes[i]]
        plt.subplot(1, ground_truth.shape[1] + 1, i + 2)
        plt.scatter(x, y, s=2)
        plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), "r--")
        plt.xlim(0, max(y))
        plt.ylim(0, max(y))
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
        plt.title(celltypes[i])
        plt.text(0.05, max(y)-0.05, f"$CCC = {eval_metric[0]:0.3f}$\n$RMSE = {eval_metric[1]:0.3f}$\n$Corr = {eval_metric[2]:0.3f}$", fontsize=8, verticalalignment='top')
    plt.savefig(os.path.join(result_dir, output_prex + '.jpg'))

###############################################################################################################################################################################

class ReferMixup(object):
    def __init__(self, option_dict):  
        self.seed = 2021; self.data_dir = option_dict['data_dir']; self.result_dir = option_dict['result_dir']; self.normalize_method = 'min_max'
        self.source_dataset_name = option_dict['source_dataset_name']; self.source_metadata_name = option_dict['source_metadata_name']
        self.target_dataset_name = option_dict['target_dataset_name']; self.target_metadata_name = option_dict['target_metadata_name']
        self.source_sample_num = option_dict['source_sample_num']; self.target_sample_num = option_dict['target_sample_num']
        self.cell_type_name = option_dict['cell_type_name']; self.cell_type_list = option_dict['cell_type_list']
        self.target_type = option_dict['target_type']; self.mixup_sample_size = option_dict['mixup_sample_size']; self.hvp_num = option_dict['hvp_num']
        set_reproducibility(self.seed)
    
    def load_data(self): # return AnnData object of bulk source data and target data, input single cell source data and target data
        if os.path.exists(os.path.join(self.result_dir, 'pseudo_bulk_source_' + str(self.source_sample_num) + '.h5ad')) and os.path.exists(os.path.join(self.result_dir, 'pseudo_bulk_target_' + str(self.target_sample_num) + '.h5ad')):
            source_data = ad.read_h5ad(os.path.join(self.result_dir, 'pseudo_bulk_source_' + str(self.source_sample_num) + '.h5ad'))
            target_data = ad.read_h5ad(os.path.join(self.result_dir, 'pseudo_bulk_target_' + str(self.target_sample_num) + '.h5ad'))
            source_data.uns['cell_type_list'] = self.cell_type_list
            target_data.uns['cell_type_list'] = self.cell_type_list
            return source_data, target_data
        
        # 1\ Mixup source data to simulate pseudo bulk source data
        source_data_x, source_data_y = self.mixup_dataset(self.source_dataset_name, self.source_metadata_name, self.source_sample_num)
        
        # 2\ Mixup target data to simulate pseudo bulk target data
        if self.target_type == "simulated":
            target_data_x, target_data_y = self.mixup_dataset(self.target_dataset_name, self.target_metadata_name, self.target_sample_num)
            target_data = ad.AnnData(X=target_data_x, obs=target_data_y) # auto generate obs_names and var_names from the index and columns of target_data_x
        if self.target_type == "real":
            if ".h5ad" in self.target_dataset_name:
                target_data = ad.read_h5ad(os.path.join(self.data_dir, self.target_dataset_name))
                target_data.X = pd.DataFrame(target_data.X.todense()).fillna(0) if sp.sparse.issparse(target_data.X) else pd.DataFrame(target_data.X).fillna(0)
                target_data.X = self.sample_normalize(target_data.X, normalize_method=self.normalize_method) if self.normalize_method else target_data.X
            elif ".csv" in self.target_dataset_name:
                target_data_x = pd.read_csv(os.path.join(self.data_dir, self.target_dataset_name), header=0, index_col=0).fillna(0)
                target_data_x_array = self.sample_normalize(target_data_x, normalize_method=self.normalize_method) if self.normalize_method else target_data_x.values
                target_data_x = pd.DataFrame(target_data_x_array, columns=target_data_x.columns, index=target_data_x.index)
                target_data = ad.AnnData(X=target_data_x) # auto generate obs_names and var_names from the index and columns of target_data_x
        
        # 3\ Get overlapped features between source and target data. (add the highly variable genes of target data as the missing features in source data)
        used_features = set(source_data_x.columns.tolist()).intersection(set(target_data.var_names.tolist())) # overlapped features between reference and target
        if self.hvp_num > 0:
            sc.pp.highly_variable_genes(target_data, n_top_genes=self.hvp_num)
            HVPs = set(target_data.var[target_data.var.highly_variable].index)
            used_features = list(used_features.union(HVPs))
        used_features = list(used_features)
            
        # 4\ Prepare train data and target data with aligned features
        missing_features = [feature for feature in used_features if feature not in list(source_data_x.columns)] # missing features in source data (subset of HVPs in target data)
        if len(missing_features) > 0:
            missing_data_x = pd.DataFrame(np.zeros((source_data_x.shape[0], len(missing_features))), columns=missing_features, index=source_data_x.index)
            source_data_x = pd.concat([source_data_x, missing_data_x], axis=1)
        source_data = ad.AnnData(X=source_data_x[used_features], obs=source_data_y) # auto generate obs_names and var_names from the index and columns of source_data_x
        source_data.uns["cell_type_list"] = self.cell_type_list
        target_data = target_data[:, used_features]
        target_data.uns["cell_type_list"] = self.cell_type_list
        save_tsne_plot(source_data, self.result_dir, output_prex='pseudo_bulk_source_' + str(self.source_sample_num))
        save_tsne_plot(target_data, self.result_dir, output_prex='pseudo_bulk_target_' + str(self.target_sample_num))
        return source_data, target_data

    def mixup_dataset(self, dataset_name, metadata_name, sample_num): # return dataframe object
        if ".h5ad" in dataset_name:
            data = ad.read_h5ad(os.path.join(self.data_dir, dataset_name))
            self.cell_type_list = list(set(data.obs[self.cell_type_name].tolist())) if self.cell_type_list == None else self.cell_type_list
            data = data[data.obs[self.cell_type_name].isin(self.cell_type_list)]
            data_x = pd.DataFrame(data.X.todense(), index=data.obs_names, columns=data.var_names).fillna(0) if sp.sparse.issparse(data.X) else pd.DataFrame(data.X, index=data.obs_names, columns=data.var_names).fillna(0)
            data_y = pd.DataFrame(data.obs.values, index=data.obs.index, columns=data.obs.columns)
        if ".csv" in dataset_name:
            data_x = pd.read_csv(os.path.join(self.data_dir, dataset_name), header=0, index_col=0).fillna(0)
            data_y = pd.read_csv(os.path.join(self.data_dir, metadata_name), header=0, index_col=0)
            self.cell_type_list = list(set(data_y[self.cell_type_name].tolist())) if self.cell_type_list == None else self.cell_type_list
            data_x = data_x.loc[data_y[self.cell_type_name].isin(self.cell_type_list), :]
            data_y = data_y.loc[data_y[self.cell_type_name].isin(self.cell_type_list), :]
        mixup_data_x = []
        mixup_data_y = []
        for i in range(int(sample_num)):
            data_ratio_list = np.random.rand(len(self.cell_type_list))
            data_ratio_list = data_ratio_list / data_ratio_list.sum()
            data_num_list = list(map(round, data_ratio_list * self.mixup_sample_size))
            sample_temp_list = []
            for i in range(len(data_num_list)):
                cell_type_subset = data_x.loc[np.array(data_y[self.cell_type_name] == self.cell_type_list[i]), :]
                sample_temp_list.append(cell_type_subset.iloc[np.random.randint(0, cell_type_subset.shape[0], data_num_list[i]), :])
            mixup_data_x.append(np.concatenate(sample_temp_list, axis=0).sum(axis=0))
            mixup_data_y.append(data_ratio_list)
        mixup_data_x = self.sample_normalize(np.array(mixup_data_x), normalize_method=self.normalize_method) if self.normalize_method else np.array(mixup_data_x)
        mixup_data_x = pd.DataFrame(mixup_data_x, columns=data_x.columns)
        mixup_data_y = pd.DataFrame(np.array(mixup_data_y), columns=self.cell_type_list) # shape: (sample_num, cell_type_num)
        return mixup_data_x, mixup_data_y
    
    def sample_normalize(self, data, normalize_method = 'min_max'): # data: numpy array
        normalized_data = pp.MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(data.T).T if normalize_method == 'min_max' else data
        normalized_data = pp.StandardScaler(copy=True).fit_transform(data) if normalize_method == 'z_score' else normalized_data
        # normalized_data = (data - data.mean(axis=0))/(data.std(axis=0) + 1e-10) if normalize_method == 'z_score' else normalized_data
        return normalized_data
    
##############################################################################################################################################################################

class ReferImpute(object):
    def __init__(self, option_dict):
        self.seed = 2021; self.result_dir = option_dict['result_dir']
        self.epoch_num = 200; self.batch_size = option_dict['batch_size']; self.learning_rate = option_dict['learning_rate']
        set_reproducibility(self.seed)

    def impute_data(self, source_data, target_data):
        source_dataset = Data.TensorDataset(torch.FloatTensor(source_data.X.astype(np.float32)), torch.FloatTensor(source_data.obs.values.astype(np.float32))) # source_data.obs.columns = source_data.uns['cell_type_list']
        train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=self.batch_size, shuffle=True)
        test_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=self.batch_size, shuffle=False)
        target_dataset = Data.TensorDataset(torch.FloatTensor(target_data.X.astype(np.float32)), torch.FloatTensor(np.zeros((target_data.shape[0], len(source_data.uns['cell_type_list'])))))
        train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=self.batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Block = lambda in_dim, out_dim: nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.LeakyReLU(0.2, inplace=True))
        model = nn.ModuleDict({'encoder': nn.Sequential(Block(len(source_data.var_names), 512), Block(512, 256)), 
                               'predictor': nn.Sequential(nn.Linear(256, len(source_data.uns['cell_type_list'])), nn.Softmax(dim=-1)), 
                               'decoder': nn.Sequential(Block(256, 512), Block(512, len(source_data.var_names)))}).to(device)
        optimizer = torch.optim.Adam([{'params': model['encoder'].parameters()}, 
                                      {'params': model['predictor'].parameters()}, 
                                      {'params': model['decoder'].parameters()}], lr=self.learning_rate)
        criterion_pred = nn.MSELoss()
        criterion_recon = nn.MSELoss()
        loss_logger = defaultdict(list)
        for epoch in range(self.epoch_num):
            model.train()
            train_target_iterator = iter(inf_loop(train_target_loader)) # endless data loader iterator
            loss_epoch, ratio_loss_epoch, recon_loss_epoch = 0., 0., 0.
            for batch_idx, (source_x, source_y) in enumerate(train_source_loader):
                optimizer.zero_grad()
                target_x, _ = next(train_target_iterator)
                input = torch.cat((source_x, target_x), 0).to(device) # source data and target data
                source_x_batch_size = source_x.shape[0]
                source_y = source_y.to(device)
                embedding = model['encoder'](input)
                pred_ratio = model['predictor'](embedding)
                pred_recon = model['decoder'](embedding)
                ratio_loss = criterion_pred(pred_ratio[:source_x_batch_size,], source_y)
                recon_loss = criterion_recon(pred_recon, input)
                loss = ratio_loss + recon_loss
                loss.backward()
                optimizer.step()
                ratio_loss_epoch += ratio_loss.cpu().data.numpy(); recon_loss_epoch += recon_loss.cpu().data.numpy(); loss_epoch += loss.cpu().data.numpy()
            loss_epoch = loss_epoch/(batch_idx + 1); loss_logger['total_loss'].append(loss_epoch)
            ratio_loss_epoch = ratio_loss_epoch/(batch_idx + 1); loss_logger['ratio_loss'].append(ratio_loss_epoch)
            recon_loss_epoch = recon_loss_epoch/(batch_idx + 1); loss_logger['recon_loss'].append(recon_loss_epoch)
            print('Epoch {:02d}/{:02d} in Stage 2: total_loss={:.4f}, ratio_loss={:.4f}, recon_loss={:.4f}'.format(epoch + 1, self.epoch_num, loss_epoch, ratio_loss_epoch, recon_loss_epoch)) if (epoch+1) % 10 == 0 else None
        save_loss_plot(loss_logger, loss_type = ['total_loss','ratio_loss','recon_loss'], result_dir=self.result_dir, output_prex = 'loss_plot_stage_2')
        model.eval()
        source_recon, source_label = None, None
        for batch_idx, (source_x, source_y) in enumerate(test_source_loader):
            source_x = source_x.to(device)
            embedding = model['encoder'](source_x)
            pred_ratio = model['predictor'](embedding)
            pred_recon = model['decoder'](embedding).detach().cpu().numpy()
            label = source_y.detach().numpy()
            source_recon = pred_recon if source_recon is None else np.concatenate((source_recon, pred_recon), axis=0)
            source_label = label if source_label is None else np.concatenate((source_label, label), axis=0)
        source_recon = pd.DataFrame(source_recon, columns=source_data.var_names)
        source_label = pd.DataFrame(source_label, columns=source_data.uns['cell_type_list'])
        source_recon_data = ad.AnnData(X=source_recon, obs=source_label)
        source_recon_data.uns['cell_type_list'] = source_data.uns['cell_type_list']
        save_tsne_plot(source_recon_data, result_dir=self.result_dir, output_prex='ae_impute_source')
        sc.pp.filter_genes(source_data, min_cells=0) # Calculate the n_cells of each gene
        missing_features = list(source_data.var[source_data.var['n_cells']==0].index) # genes with zero expression in all cells
        if len(missing_features) > 0:
            recon_source_data_missing_features = source_recon_data[:,missing_features]
            save_tsne_plot(recon_source_data_missing_features, self.result_dir, output_prex='ae_impute_source_missing_features')
        return source_recon_data
    
##############################################################################################################################################################################

class DANN(object):
    def __init__(self, option_dict):
        self.seed = 2021; self.result_dir = option_dict['result_dir']; self.target_type = option_dict['target_type']
        self.epoch_num = option_dict['epoch_num']; self.batch_size = option_dict['batch_size']; self.learning_rate = option_dict['learning_rate']
        self.model = None; self.source_data = None; self.target_data = None; self.test_target_loader = None
        set_reproducibility(self.seed)

    def train(self, source_data, target_data):
        source_dataset = Data.TensorDataset(torch.FloatTensor(source_data.X.astype(np.float32)), torch.FloatTensor(source_data.obs.values.astype(np.float32)))
        train_source_loader = Data.DataLoader(dataset=source_dataset, batch_size=self.batch_size, shuffle=True)
        target_data_x = target_data.X.astype(np.float32)
        target_data_y = np.array(target_data.obs.values, dtype=np.float32) if self.target_type == "simulated" else np.zeros(target_data.shape[0], len(source_data.uns['cell_type_list']))
        target_dataset = Data.TensorDataset(torch.FloatTensor(target_data_x), torch.FloatTensor(target_data_y))
        train_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=self.batch_size, shuffle=True)
        self.source_data = source_data; self.target_data = target_data; self.test_target_loader = Data.DataLoader(dataset=target_dataset, batch_size=self.batch_size, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Block = lambda in_dim, out_dim, do_rates: nn.Sequential(nn.Linear(in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(p=do_rates, inplace=False))
        self.model = nn.ModuleDict({'encoder': nn.Sequential(Block(len(source_data.var_names), 512, 0), Block(512, 256, 0.3)),
                                    'predictor': nn.Sequential(Block(256, 128, 0.2), nn.Linear(128, len(source_data.uns['cell_type_list'])), nn.Softmax(dim=1)),
                                    'discriminator': nn.Sequential(Block(256, 128, 0.2), nn.Linear(128, 1), nn.Sigmoid())}).to(device)
        optimizer_da_1 = torch.optim.Adam([{'params': self.model['encoder'].parameters()},
                                           {'params': self.model['predictor'].parameters()},
                                           {'params': self.model['discriminator'].parameters()}], lr=self.learning_rate)
        optimizer_da_2 = torch.optim.Adam([{'params': self.model['encoder'].parameters()},
                                           {'params': self.model['discriminator'].parameters()}], lr=self.learning_rate)
        criterion = nn.MSELoss()
        criterion_da = nn.BCELoss()
        loss_logger = defaultdict(list) 
        source_label = torch.ones(self.batch_size).unsqueeze(1).to(device)  # define the source domain label as 1
        target_label = torch.zeros(self.batch_size).unsqueeze(1).to(device)  # define the target domain label as 0
        for epoch in range(self.epoch_num):
            self.model.train()
            train_target_iterator = iter(inf_loop(train_target_loader))
            pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch = 0., 0., 0.
            for batch_idx, (source_x, source_y) in enumerate(train_source_loader):
                # 1\ Train the predictor
                optimizer_da_1.zero_grad()
                target_x, _ = next(train_target_iterator)
                embedding_source = self.model['encoder'](source_x.to(device))
                embedding_target = self.model['encoder'](target_x.to(device))
                frac_pred = self.model['predictor'](embedding_source)
                domain_pred_source = self.model['discriminator'](embedding_source)
                domain_pred_target = self.model['discriminator'](embedding_target)
                pred_loss = criterion(frac_pred, source_y.to(device))    
                disc_loss = criterion_da(domain_pred_source, source_label[0:domain_pred_source.shape[0],]) + \
                    criterion_da(domain_pred_target, target_label[0:domain_pred_target.shape[0],])
                loss = pred_loss + disc_loss
                pred_loss_epoch += pred_loss.data.item()
                disc_loss_epoch += disc_loss.data.item()
                loss.backward(retain_graph=True)
                optimizer_da_1.step()
                # 2\ Train the discriminator
                optimizer_da_2.zero_grad()
                embedding_source = self.model['encoder'](source_x.to(device))
                embedding_target = self.model['encoder'](target_x.to(device))
                domain_pred_source = self.model['discriminator'](embedding_source)
                domain_pred_target = self.model['discriminator'](embedding_target)
                disc_loss_DA = criterion_da(domain_pred_target, source_label[0:domain_pred_target.shape[0],]) + \
                    criterion_da(domain_pred_source, target_label[0:domain_pred_source.shape[0],]) # reverse the label
                disc_loss_DA_epoch += disc_loss_DA.data.item()
                disc_loss_DA.backward(retain_graph=True)
                optimizer_da_2.step()
            pred_loss_epoch = pred_loss_epoch/(batch_idx + 1); loss_logger['pred_loss'].append(pred_loss_epoch)
            disc_loss_epoch = disc_loss_epoch/(batch_idx + 1); loss_logger['disc_loss'].append(disc_loss_epoch)
            disc_loss_DA_epoch = disc_loss_DA_epoch/(batch_idx + 1); loss_logger['disc_loss_DA'].append(disc_loss_DA_epoch)
            print('Epoch {:02d}/{:02d} in Stage 3: ratio_loss={:.4f}, disc_loss={:.4f}, disc_loss_DA={:.4f}'.format(epoch + 1, self.epoch_num, pred_loss_epoch, disc_loss_epoch, disc_loss_DA_epoch)) if (epoch+1) % 10 == 0 else None
            if self.target_type == "simulated":
                target_preds, ground_truth = self.prediction()
                epoch_rmse, epoch_pcc, epoch_ccc = compute_metrics(target_preds, ground_truth)
                loss_logger['target_ccc'].append(epoch_ccc); loss_logger['target_rmse'].append(epoch_rmse); loss_logger['target_pcc'].append(epoch_pcc)
        if self.target_type == "simulated":
            save_loss_plot(loss_logger, loss_type = ['pred_loss','disc_loss','disc_loss_DA','target_ccc','target_rmse','target_corr'], result_dir=self.result_dir, output_prex = 'loss_plot_stage_3')
        elif self.target_type == "real":
            save_loss_plot(loss_logger, loss_type = ['pred_loss','disc_loss','disc_loss_DA'], result_dir=self.result_dir, output_prex = 'loss_metric_plot_stage_3')
            
    def prediction(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        target_preds, ground_truth = None, None
        for batch_idx, (target_x, target_y) in enumerate(self.test_target_loader):
            target_x = target_x.to(device)
            preds = self.model['predictor'](self.model['encoder'](target_x)).detach().cpu().numpy()
            target_preds = preds if target_preds is None else np.concatenate((target_preds, preds), axis=0)
            ground_truth = target_y.detach().numpy() if ground_truth is None else np.concatenate((ground_truth, target_y.detach().numpy()), axis=0)
        target_preds = pd.DataFrame(target_preds, columns=self.source_data.uns['cell_type_list'])
        ground_truth = pd.DataFrame(ground_truth, columns=self.source_data.uns['cell_type_list'])
        return target_preds, ground_truth
    
##############################################################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/murine_cellline/', help='data directory')
    parser.add_argument('--result_dir', type=str, default='./result/murine_cellline/', help='result directory')
    parser.add_argument('--source_dataset_name', type=str, default='murine_N2_SCP_exp.csv', help='reference dataset name')
    parser.add_argument('--source_metadata_name', type=str, default='murine_N2_SCP_meta.csv', help='reference metadata name')
    parser.add_argument('--target_dataset_name', type=str, default='murine_nanoPOTS_SCP_exp.csv', help='target dataset name')
    parser.add_argument('--target_metadata_name', type=str, default='murine_nanoPOTS_SCP_meta.csv', help='target metadata name')
    parser.add_argument('--source_sample_num', type=int, default=4000, help='reference sample number')
    parser.add_argument('--target_sample_num', type=int, default=1000, help='target sample number')
    parser.add_argument('--cell_type_name', type=str, default='CellType', help='cell type name')
    parser.add_argument('--cell_type_list', nargs='+', default=['C10','SVEC','RAW'], help='cell type list')
    parser.add_argument('--target_type', type=str, default='simulated', help='target type')
    parser.add_argument('--mixup_sample_size', type=int, default=15, help='mixup sample size')
    parser.add_argument('--hvp_num', type=int, default=500, help='high variable protein number')
    parser.add_argument('--epoch_num', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    args = parser.parse_args()
    options = vars(args)
    json.dump(options, open(os.path.join(options['result_dir'], 'parameters.json'), "w"), indent=4)
    print("------ Running Stage 1: Reference Data Mixup ------")
    source_data, target_data = ReferMixup(options).load_data()
    print('The dim of source data is :', source_data.shape, '\n', 'The dim of target data is :', target_data.shape)
    print("------ Running Stage 2: Reference Data Imputation ------")
    source_recon_data = ReferImpute(options).impute_data(source_data, target_data)
    print("------ Running Stage 3: Training DANN Model ------")
    model = DANN(options)
    model.train(source_recon_data, target_data) 
    print("------ Running Stage 4: Inference for Target Data ------")
    final_preds_target, ground_truth_target = model.prediction()
    final_preds_target.to_csv(os.path.join(options['result_dir'], "target_predicted_fractions.csv"))
    ground_truth_target.to_csv(os.path.join(options['result_dir'], "target_ground_truth_fractions.csv")) if options['target_type'] == "simulated" else None
    save_pred_plot(final_preds_target, ground_truth_target, result_dir=options['result_dir'], output_prex='pred_fraction_target_scatter') if options['target_type'] == "simulated" else None
    