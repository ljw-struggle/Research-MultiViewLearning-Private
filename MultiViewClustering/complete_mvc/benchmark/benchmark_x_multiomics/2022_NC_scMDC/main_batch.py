import math, os, h5py, argparse
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset
from time import time
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from main import gene_selection, get_cluster_number, best_map, read_dataset, normalize, NBLoss, ZINBLoss, activation_dict, build_block, cluster_acc

class scMultiClusterBatch(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, n_batch, encode_layer=[], decode_layer_1=[], decode_layer_2=[], 
                 activation='elu', tau=1.0, sigma_1=2.5, sigma_2=0.1, alpha=1.0, gamma=1.0, phi_1=0.0001, phi_2=0.0001, cutoff=0.5, save_dir='./result'):
        super(scMultiClusterBatch, self).__init__()
        self.input_dim_1 = input_dim_1; self.input_dim_2 = input_dim_2; self.tau = tau; self.activation = activation; self.z_dim = encode_layer[-1]; self.zinb_loss = ZINBLoss(); self.mse = nn.MSELoss(); 
        self.sigma_1 = sigma_1; self.sigma_2 = sigma_2; self.alpha = alpha; self.gamma = gamma; self.phi_1 = phi_1; self.phi_2 = phi_2; self.tau=tau; self.cutoff = cutoff; self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = build_block([input_dim_1+input_dim_2+n_batch]+encode_layer, activation=activation)
        self.decoder1 = build_block([decode_layer_1[0]+n_batch]+decode_layer_1[1:], activation=activation)
        self.decoder2 = build_block([decode_layer_2[0]+n_batch]+decode_layer_2[1:], activation=activation)       
        self.dec_mean1 = nn.Sequential(nn.Linear(decode_layer_1[-1], input_dim_1), activation_dict['mean_act'])
        self.dec_disp1 = nn.Sequential(nn.Linear(decode_layer_1[-1], input_dim_1), activation_dict['disp_act'])
        self.dec_mean2 = nn.Sequential(nn.Linear(decode_layer_2[-1], input_dim_2), activation_dict['mean_act'])
        self.dec_disp2 = nn.Sequential(nn.Linear(decode_layer_2[-1], input_dim_2), activation_dict['disp_act'])
        self.dec_pi1 = nn.Sequential(nn.Linear(decode_layer_1[-1], input_dim_1), activation_dict['sigmoid'])
        self.dec_pi2 = nn.Sequential(nn.Linear(decode_layer_2[-1], input_dim_2), activation_dict['sigmoid'])
        
    def forward(self, x1, x2, b, stage='pretrain'):
        x_noise = torch.cat([x1 + torch.randn_like(x1) * self.sigma_1, x2 + torch.randn_like(x2) * self.sigma_2], dim=-1) # add gaussian noise
        h_noise = self.encoder(torch.cat([x_noise, b], dim=-1))
        h_noise = torch.cat([h_noise, b], dim=-1)
        h_1 = self.decoder_1(h_noise); mean_1 = self.decoder_mean_1(h_1); disp_1 = self.decoder_disp_1(h_1); pi_1 = self.decoder_pi_1(h_1)
        h_2 = self.decoder_2(h_noise); mean_2 = self.decoder_mean_2(h_2); disp_2 = self.decoder_disp_2(h_2); pi_2 = self.decoder_pi_2(h_2)
        x = torch.cat([x1, x2], dim=-1)
        z = self.encoder(torch.cat([x, b], dim=-1)) # latent representation, shape: (batch_size, z_dim)
        z_square = torch.sum(torch.square(z), dim=1, keepdim=True) # shape: (batch_size, 1)
        euclidean_distance = z_square + z_square.t() - 2.0 * torch.matmul(z, z.t()) # shape: (batch_size, batch_size)
        matrix = euclidean_distance / self.alpha # shape: (batch_size, batch_size)
        matrix = torch.pow(1.0 + matrix, -(self.alpha + 1.0) / 2.0) # shape: (batch_size, batch_size)
        zerodiag_matrix = matrix - torch.diag(torch.diag(matrix)) # shape: (batch_size, batch_size), remove diagonal elements
        latent_q = zerodiag_matrix / torch.sum(zerodiag_matrix, dim=1, keepdim=True) # the probability of each pair of cells, shape: (batch_size, batch_size)
        if stage == 'pretrain':
            return z, matrix, latent_q, mean_1, mean_2, disp_1, disp_2, pi_1, pi_2
        elif stage == 'finetune': # soft assign
            # z: latent representation, shape: (batch_size, z_dim), self.mu: cluster centers, shape: (n_clusters, z_dim)
            q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) # shape: (batch_size, n_clusters)
            q = q**((self.alpha+1.0)/2.0) # shape: (batch_size, n_clusters)
            q = q / torch.sum(q, dim=1, keepdim=True) # shape: (batch_size, n_clusters), represent the probability of each cell belonging to each cluster
            return z, q, matrix, latent_q, mean_1, mean_2, disp_1, disp_2, pi_1, pi_2
        
    def encode(self, X1, X2, B, batch_size=256, stage='pretain'):
        self.eval()
        encoded = []
        num_sample = X1.shape[0]; num_batch = int(math.ceil(1.0 * X1.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            inputs_1 = X1[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
            inputs_2 = X2[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
            inputs_b = B[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
            z, _, _, _, _, _, _, _, _ = self.forward(inputs_1, inputs_2, inputs_b, stage=stage)
            encoded.append(z.data)
        return torch.cat(encoded, dim=0)
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0, keepdim=True) # shape: (batch_size, batch_size)
        return p / p.sum(1, keepdim=True) # shape: (batch_size, batch_size)

    def kld_loss(self, p, q): # KL divergence between two distributions
        return torch.mean(torch.sum(p * torch.log(p/q), dim=-1)) # torch.mean(torch.sum(p * torch.log(p), dim=-1) - torch.sum(p * torch.log(q), dim=-1))
    
    def kmeans_loss(self, z): # z: latent representation, shape: (batch_size, z_dim), self.mu: cluster centers, shape: (n_clusters, z_dim)
        dist_1 = self.tau * torch.sum(torch.square(z.unsqueeze(1) - self.mu), dim=2) # shape: (batch_size, n_clusters)
        temp_dist_1 = dist_1 - torch.mean(dist_1, dim=1, keepdim=True) # shape: (batch_size, n_clusters), remove mean
        q = torch.exp(- temp_dist_1) # shape: (batch_size, n_clusters) 
        q = q / torch.sum(q, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        q = torch.pow(q, 2); # shape: (batch_size, n_clusters)
        q = q / torch.sum(q, dim=1, keepdim=True) # shape: (batch_size, n_clusters)
        dist_2 = dist_1 * q # shape: (batch_size, n_clusters)
        return dist_1, torch.mean(torch.sum(dist_2, dim=1)) # shape: (batch_size, n_clusters), shape: (1)
    
    def pretrain_autoencoder(self, X_1, X_1_raw, X_1_size_factor, X_2, X_2_raw, X_2_size_factor, B, batch_size=256, lr=0.001, epoch_num=400):
        print('Pretraining stage for scMultiCluster.')
        dataset = TensorDataset(torch.Tensor(X_1), torch.Tensor(X_1_raw), torch.Tensor(X_1_size_factor), torch.Tensor(X_2), torch.Tensor(X_2_raw), torch.Tensor(X_2_size_factor), torch.Tensor(B))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        num = X_1.shape[0] # number of cells
        self.train()
        for epoch in range(epoch_num):
            loss_val = 0; loss_val_recon_1 = 0; loss_val_recon_2 = 0; loss_val_kld = 0
            for batch_idx, (x_1_batch, x_1_raw_batch, x_1_size_factor_batch, x_2_batch, x_2_raw_batch, x_2_size_factor_batch, b_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                x_1 = x_1_batch.to(self.device); x_1_raw = x_1_raw_batch.to(self.device); x_1_size_factor = x_1_size_factor_batch.to(self.device)
                x_2 = x_2_batch.to(self.device); x_2_raw = x_2_raw_batch.to(self.device); x_2_size_factor = x_2_size_factor_batch.to(self.device); b = b_batch.to(self.device)
                z_batch, z_batch_matrix, z_batch_latent_q, mean_1_batch, mean_2_batch, disp_1_batch, disp_2_batch, pi_1_batch, pi_2_batch = self.forward(x_1, x_2, b, stage='pretrain')
                # print(torch.isnan(z_batch).sum(), torch.isnan(z_batch_matrix).sum(), torch.isnan(z_batch_latent_q).sum(), torch.isnan(mean_1_batch).sum(), torch.isnan(mean_2_batch).sum(), torch.isnan(disp_1_batch).sum(), torch.isnan(disp_2_batch).sum(), torch.isnan(pi_1_batch).sum(), torch.isnan(pi_2_batch).sum())
                recon_loss_1 = self.zinb_loss(x=x_1_raw, mean=mean_1_batch, r=disp_1_batch, pi=pi_1_batch, scale_factor=x_1_size_factor)
                recon_loss_2 = self.zinb_loss(x=x_2_raw, mean=mean_2_batch, r=disp_2_batch, pi=pi_2_batch, scale_factor=x_2_size_factor)
                latent_p_batch = self.target_distribution(z_batch_latent_q); latent_p_batch = latent_p_batch + torch.diag(torch.diag(z_batch_matrix)); latent_q_batch = z_batch_latent_q + torch.diag(torch.diag(z_batch_matrix))
                kld_loss = self.kld_loss(latent_p_batch, latent_q_batch)
                # print(torch.isnan(recon_loss_1).sum(), torch.isnan(recon_loss_2).sum(), torch.isnan(kld_loss).sum())
                loss = recon_loss_1 + recon_loss_2 + kld_loss * self.phi_1 if epoch + 1 >= epoch_num * self.cutoff else recon_loss_1 + recon_loss_2
                loss.backward()
                # print(torch.isnan(self.encoder[0].weight.grad).sum(), torch.isnan(self.decoder_1[0].weight.grad).sum(), torch.isnan(self.decoder_mean_1[0].weight.grad).sum(), torch.isnan(self.decoder_disp_1[0].weight.grad).sum(), torch.isnan(self.decoder_pi_1[0].weight.grad).sum())
                optimizer.step()
                loss_val += loss.item() * len(x_1_batch); loss_val_recon_1 += recon_loss_1.item() * len(x_1_batch); loss_val_recon_2 += recon_loss_2.item() * len(x_2_batch)
                loss_val_kld += kld_loss.item() * len(x_1_batch) if epoch + 1 >= epoch_num * self.cutoff else 0
            loss_val = loss_val / num; loss_val_recon_1 = loss_val_recon_1 / num; loss_val_recon_2 = loss_val_recon_2 / num; loss_val_kld = loss_val_kld / num
            print('Pretrain epoch {}: Total loss={:.6f}, ZINB_loss_1={:.6f}, ZINB_loss_2={:.6f}, KLD_loss={:.6f}'.format(epoch + 1, loss_val, loss_val_recon_1, loss_val_recon_2, loss_val_kld)) if epoch % 10 == 0 else None
        torch.save({'model_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(self.save_dir, 'pretrain_checkpoint_{:d}.pth'.format(epoch+1)))

    def fit(self, X_1, X_1_raw, X_1_size_factor, X_2, X_2_raw, X_2_size_factor, B, y=None, lr=1.0, n_clusters = 4, batch_size=256, epoch_num=10, update_interval=1, tol=1e-3):
        print("Clustering stage for scMultiCluster.")
        print("Initializing cluster centers with kmeans.")
        X_1 = torch.tensor(X_1, dtype=torch.float32).to(self.device); X_1_raw = torch.tensor(X_1_raw, dtype=torch.float32).to(self.device); X_1_size_factor = torch.tensor(X_1_size_factor, dtype=torch.float32).to(self.device)
        X_2 = torch.tensor(X_2, dtype=torch.float32).to(self.device); X_2_raw = torch.tensor(X_2_raw, dtype=torch.float32).to(self.device); X_2_size_factor = torch.tensor(X_2_size_factor, dtype=torch.float32).to(self.device)
        B = torch.tensor(B).to(self.device)
        Z = self.encode(X_1, X_2, B, batch_size=batch_size)
        kmeans = KMeans(n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(Z.data.cpu().numpy()); y_pred_last = y_pred
        self.mu = nn.Parameter(torch.Tensor(n_clusters, self.z_dim), requires_grad=True).to(self.device) # add cluster centers as parameters
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_)) # initialize cluster centers
        if y is not None:
            ami = np.round(adjusted_mutual_info_score(y, y_pred), 5); nmi = np.round(normalized_mutual_info_score(y, y_pred), 5); ari = np.round(adjusted_rand_score(y, y_pred), 5)
            print('Initializing k-means: AMI={:.4f}, NMI={:.4f}, ARI={:.4f}'.format(ami, nmi, ari))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=0.95)
        self.train()
        num_sample = X_1.shape[0]; num_batch = int(math.ceil(1.0*X_1.shape[0]/batch_size))
        for epoch in range(epoch_num):
            if epoch % update_interval == 0:
                Z = self.encode(X_1, X_2, B, batch_size=batch_size)
                dist, _ = self.kmeans_loss(Z)
                y_pred = torch.argmin(dist, dim=1).data.cpu().numpy()
                if y is not None:
                    acc = np.round(cluster_acc(y, y_pred), 5); 
                    ami = np.round(adjusted_mutual_info_score(y, y_pred), 5); nmi = np.round(normalized_mutual_info_score(y, y_pred), 5); ari = np.round(adjusted_rand_score(y, y_pred), 5)
                    print('Clustering {:d}: ACC={:.4f}, AMI={:.4f}, NMI={:.4f}, ARI={:.4f}'.format(epoch, acc, ami, nmi, ari))
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num_sample # check convergence
                y_pred_last = y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label:', delta_label, '< tol:', tol, '. Reached tolerance threshold. Stopping training.')
                    # torch.save({'epoch': epoch+1, 'state_dict': self.state_dict(), 'mu': self.mu, 'y_pred': y_pred, 'y_pred_last': y_pred_last, 'y': y }, os.path.join(self.save_dir, 'finetune_checkpoint_{:d}.pth'.format(epoch+1)))
                    break
            loss_val = 0; loss_val_recon_1 = 0; loss_val_recon_2 = 0; loss_val_kld = 0; loss_val_cluster = 0
            for batch_idx in range(num_batch):
                x_1 = X_1[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]; x_2 = X_2[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
                x_1_raw = X_1_raw[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]; x_2_raw = X_2_raw[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
                x_1_size_factor = X_1_size_factor[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]; x_2_size_factor = X_2_size_factor[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
                b = B[batch_idx*batch_size:min((batch_idx+1)*batch_size, num_sample)]
                z_batch, q_batch, z_batch_matrix, z_batch_latent_q, mean_1_batch, mean_2_batch, disp_1_batch, disp_2_batch, pi_1_batch, pi_2_batch = self.forward(x_1, x_2, b, stage='finetune')
                _, cluster_loss = self.kmeans_loss(z_batch)
                recon_loss_1 = self.zinb_loss(x=x_1_raw, mean=mean_1_batch, r=disp_1_batch, pi=pi_1_batch, scale_factor=x_1_size_factor)
                recon_loss_2 = self.zinb_loss(x=x_2_raw, mean=mean_2_batch, r=disp_2_batch, pi=pi_2_batch, scale_factor=x_2_size_factor)
                latent_p_batch = self.target_distribution(z_batch_latent_q); latent_p_batch = latent_p_batch + torch.diag(torch.diag(z_batch_matrix)); latent_q_batch = z_batch_latent_q + torch.diag(torch.diag(z_batch_matrix))
                kld_loss = self.kld_loss(latent_p_batch, latent_q_batch) # latent_p_batch is the target distribution, latent_q_batch is the predicted distribution
                loss = recon_loss_1 + recon_loss_2 + kld_loss * self.phi_2 + cluster_loss * self.gamma
                optimizer.zero_grad()
                loss.backward(); 
                torch.nn.utils.clip_grad_norm_(self.mu, 1) # clip gradient
                optimizer.step()
                loss_val += loss.data * len(x_1); loss_val_cluster += cluster_loss.data * len(x_1); loss_val_recon_1 += recon_loss_1.data * len(x_1); loss_val_recon_2 += recon_loss_2.data * len(x_2); loss_val_kld += kld_loss.data * len(x_1);
            loss_val = loss_val / num_sample; loss_val_cluster = loss_val_cluster / num_sample; loss_val_recon_1 = loss_val_recon_1 / num_sample; loss_val_recon_2 = loss_val_recon_2 / num_sample; loss_val_kld = loss_val_kld / num_sample
            print('Epoch {:d}: total_loss={:.6f}, clustering_loss={:.6f}, ZINB_loss_1={:.6f}, ZINB_loss_2={:.6f}, KLD_loss={:.6f}'.format(epoch+1, loss_val, loss_val_cluster, loss_val_recon_1, loss_val_recon_2, loss_val_kld)) if epoch % 10 == 0 else None
        return y_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_file', default='./data/CITESeq_GSE128639_BMNC_anno.h5')
    parser.add_argument('--pretrain_weight_file', default='None', type=str)
    parser.add_argument('--save_dir', default='./result/')
    parser.add_argument('--run', default=1, type=int) # the number of runs
    parser.add_argument('--nbatch', default=2, type=int)
    parser.add_argument('-el','--encode_layer', nargs='+', default=[256,64,32,16])
    parser.add_argument('-dl1','--decode_layer_1', nargs='+', default=[16,64,256])
    parser.add_argument('-dl2','--decode_layer_2', nargs='+', default=[16,20])
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=400, type=int)
    parser.add_argument('--finetune_epochs', default=5000, type=int)
    parser.add_argument('--cutoff', default=0.5, type=float, help='Start to train combined layer after what ratio of epoch')
    parser.add_argument('--gamma', default=.1, type=float, help='coefficient of clustering loss')
    parser.add_argument('--tau', default=1., type=float, help='fuzziness of clustering loss')          
    parser.add_argument('--sigma_1', default=2.5, type=float)
    parser.add_argument('--sigma_2', default=1.5, type=float)          
    parser.add_argument('--phi_1', default=0.001, type=float, help='coefficient of KL loss in pretraining stage')
    parser.add_argument('--phi_2', default=0.001, type=float, help='coefficient of KL loss in clustering stage')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--n_clusters', default=27, type=int)
    parser.add_argument('--resolution', default=0.2, type=float)
    parser.add_argument('--n_neighbors', default=30, type=int)
    parser.add_argument('--no_labels', action='store_true', default=False)
    parser.add_argument('--filter_1', action='store_true', default=False, help='Do mRNA selection')
    parser.add_argument('--filter_2', action='store_true', default=False, help='Do ADT/ATAC selection')
    parser.add_argument('--filter_1_num', default=1000, type=float, help='Number of mRNA after feature selection')
    parser.add_argument('--filter_2_num', default=2000, type=float, help='Number of ADT/ATAC after feature selection')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1\ preprocessing scRNA-seq read counts matrix
    with h5py.File(args.data_file, 'r') as f:
        x1 = np.array(f['X1']); x2 = np.array(f['X2']); y = np.array(f['Y']) if not args.no_labels else None; b = np.array(f['Batch'])    
    enc = OneHotEncoder(); enc.fit(b.reshape(-1, 1)); B = enc.transform(b.reshape(-1, 1)).toarray() # shape: (n_obs, n_batch)
    x1 = x1[:, gene_selection(x1, n=args.f1, plot=False)] if args.filter1 else x1
    x2 = x2[:, gene_selection(x2, n=args.f2, plot=False)] if args.filter2 else x2 
    adata1 = sc.AnnData(x1); #adata1.obs['Group'] = y
    adata1 = read_dataset(adata1, transpose=False, test_split=False, copy=True)
    adata1 = normalize(adata1, size_factors=True, normalize_input=True, logtrans_input=True)
    adata2 = sc.AnnData(x2); #adata2.obs['Group'] = y
    adata2 = read_dataset(adata2, transpose=False, test_split=False, copy=True)
    adata2 = normalize(adata2, size_factors=True, normalize_input=True, logtrans_input=True)
    
    # 2\ Pretrain scMultiCluster
    model = scMultiClusterBatch(input_dim_1=adata1.n_vars, input_dim_2=adata2.n_vars, n_batch=args.nbatch, encode_layer=args.encode_layer, decode_layer_1=args.decode_layer_1, decode_layer_2=args.decode_layer_2,
                           activation='elu', tau=args.tau, sigma_1=args.sigma_1, sigma_2=args.sigma_2, gamma=args.gamma, phi_1=args.phi_1, phi_2=args.phi_2, cutoff = args.cutoff, save_dir=args.save_dir)
    if not os.path.exists(args.pretrain_weight_file):
        model.pretrain_autoencoder(X_1=adata1.X, X_1_raw=adata1.raw.X, X_1_size_factor=adata1.obs.size_factors.to_numpy(), X_2=adata2.X, X_2_raw=adata2.raw.X, X_2_size_factor=adata2.obs.size_factors.to_numpy(), B=B,
                                   batch_size=args.batch_size, epoch_num=args.pretrain_epochs)
    else:
        print("==> loading checkpoint '{}'".format(args.pretrain_weight_file))
        model.load_state_dict(torch.load(args.pretrain_weight_file)['state_dict'])
        
    # 3\ Fine-tune scMultiCluster according to the clustering loss that makes the latent representation cluster well (refine the representation and cluster results)
    latent = model.encodeBatch(torch.tensor(adata1.X, dtype=torch.float32).to(model.device), torch.tensor(adata2.X, dtype=torch.float32).to(model.device), torch.tensor(B, dtype=torch.float32).to(model.device)).cpu().numpy()
    n_clusters = get_cluster_number(latent, res=args.resolution, n=args.n_neighbors) if args.n_clusters == -1 else args.n_clusters      
    y_pred = model.fit(X_1=adata1.X, X_1_raw=adata1.raw.X, X_1_size_factor=adata1.obs.size_factors.to_numpy(), X_2=adata2.X, X_2_raw=adata2.raw.X, X_2_size_factor=adata2.obs.size_factors.to_numpy(), 
                       B=B, y=y, n_clusters=n_clusters, batch_size=args.batch_size, epoch_num=args.finetune_epochs, update_interval=args.update_interval, tol=args.tol, lr=args.lr)
    y_pred_best_map = best_map(y, y_pred) if not args.no_labels else y_pred.astype(int)
    np.savetxt(args.save_dir + "/" + str(args.run) + "_pred.csv", y_pred, delimiter=",")
    final_latent = model.encodeBatch(torch.tensor(adata1.X, dtype=torch.float32).to(model.device), torch.tensor(adata2.X, dtype=torch.float32).to(model.device), torch.tensor(B, dtype=torch.float32).to(model.device)).cpu().numpy()
    np.savetxt(args.save_dir + "/" + str(args.run) + "_embedding.csv", final_latent, delimiter=",")
    if not args.no_labels:
        ami = np.round(adjusted_mutual_info_score(y, y_pred), 5); nmi = np.round(normalized_mutual_info_score(y, y_pred), 5); ari = np.round(adjusted_rand_score(y, y_pred), 5)
        print('Final: AMI= %.4f, NMI= %.4f, ARI= %.4f' % (ami, nmi, ari))
