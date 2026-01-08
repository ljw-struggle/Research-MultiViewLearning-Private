import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import os
import random
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.backends import cudnn
from tqdm import tqdm
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.sparse.linalg import svds
from scipy.special import comb
np.random.seed(1)


def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                # print(S.shape, t, i)
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp


def post_proC(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5 * (C + C.T)
    r = min(d * K + 1, C.shape[0] - 1)
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize', random_state=66)
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def f1_score(gt_s, s):
    N = len(gt_s)
    num_t = 0
    num_h = 0
    num_i = 0
    for n in range(N - 1):
        tn = (gt_s[n] == gt_s[n + 1:]).astype('int')
        hn = (s[n] == s[n + 1:]).astype('int')
        num_t += np.sum(tn)
        num_h += np.sum(hn)
        num_i += np.sum(tn * hn)
    p = r = f = 1
    if num_h > 0:
        p = num_i / num_h
    if num_t > 0:
        r = num_i / num_t
    if p + r == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f


def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeConvBlock, self).__init__()
        self.de_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        dec = self.de_conv(inputs)
        dec = self.relu(dec)
        return dec


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        conv = self.conv(inputs)
        conv = self.relu(conv)
        return conv


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=input_dim, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=output_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class EncoderSingle(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EncoderSingle, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=input_dim, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)
        self.conv_block3 = ConvBlock(in_channels=64, out_channels=output_dim)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.de_conv1 = DeConvBlock(in_channels=input_dim, out_channels=64)
        self.de_conv2 = DeConvBlock(in_channels=64, out_channels=64)
        # self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
        # output_padding=1, padding=1)
        self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)

    def forward(self, x):
        x = self.de_conv1(x)
        x = self.de_conv2(x)
        x = self.de_conv3(x)
        return x


class DecoderSingle(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DecoderSingle, self).__init__()
        self.de_conv1 = DeConvBlock(in_channels=input_dim, out_channels=64)
        self.de_conv2 = DeConvBlock(in_channels=64, out_channels=64)
        self.de_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=output_dim, kernel_size=3, stride=2,
                                           output_padding=1, padding=1)

    def forward(self, x):
        x = self.de_conv1(x)
        x = self.de_conv2(x)
        x = self.de_conv3(x)
        return x


class SelfExpression(nn.Module):
    def __init__(self, n_samples):
        super(SelfExpression, self).__init__()
        self.cof = nn.Parameter(1.0e-8 * torch.ones(n_samples, n_samples, dtype=torch.float32), requires_grad=True)
        self.n_samples = n_samples

    def forward(self, x):
        y = torch.matmul(self.cof - torch.diag(torch.diag(self.cof)), x.view(self.n_samples, -1))
        y = y.view(x.size())
        # 返回自表示系数矩阵，以及重构的latent
        return self.cof, y


class AutoEncoderInit(nn.Module):
    def __init__(self, batch_size, ft=False):
        super(AutoEncoderInit, self).__init__()
        self.ft = ft
        # different view feature input
        self.batch_size = batch_size
        self.iter = 0

        self.encoder1 = Encoder(input_dim=3, output_dim=64)
        self.encoder2 = Encoder(input_dim=1, output_dim=64)
        self.encoder1_single = Encoder(input_dim=3, output_dim=64)
        self.encoder2_single = Encoder(input_dim=1, output_dim=64)

        self.decoder1 = Decoder(input_dim=64, output_dim=3)
        self.decoder2 = Decoder(input_dim=64, output_dim=1)
        self.decoder1_single = Decoder(input_dim=64, output_dim=3)
        self.decoder2_single = Decoder(input_dim=64, output_dim=1)
        self.self_express_view_1 = SelfExpression(batch_size)
        self.self_express_view_2 = SelfExpression(batch_size)
        self.self_express_view_common = torch.nn.Parameter\
            (1.0e-8 * torch.ones(self.batch_size, self.batch_size, dtype=torch.float32).cuda(), requires_grad=True)

    def forward(self, all_views_data):
        latent1 = self.encoder1(all_views_data[0])
        latent2 = self.encoder2(all_views_data[1])
        diversity_latent_1 = self.encoder1_single(all_views_data[0])
        diversity_latent_2 = self.encoder2_single(all_views_data[1])
        # if self.ft is True, we reconstruct data by using after self-expressive, or use without self-expressive
        if self.ft:
            # Self Expressive Layer Parts
            # Diversity Self Expressive, \|F_i^s - F_i^s * Z_i\|
            # latent1_diversity_se = torch.reshape(latent1_diversity_se, shape=diversity_latent_1.size())
            # latent2_diversity_se = torch.reshape(latent2_diversity_se, shape=diversity_latent_2.size())
            # Common Self Expressive, \|F_i^c - F_i^c * Z\|
            z1, latent1_diversity_se = self.self_express_view_1(diversity_latent_1)
            z2, latent2_diversity_se = self.self_express_view_2(diversity_latent_2)
            # Common Self Expressive Coef, \|F_i^c - F_i^c * Z_{common}\|
            z_common = self.self_express_view_common - torch.diag(torch.diag(self.self_express_view_common))
            latent1_se = torch.matmul(z_common, latent1.view(self.batch_size, -1))
            latent2_se = torch.matmul(z_common, latent2.view(self.batch_size, -1))
            latent1_se = torch.reshape(latent1_se, shape=latent1.size())
            latent2_se = torch.reshape(latent2_se, shape=latent2.size())

            view1_r = self.decoder1(latent1_se)
            view2_r = self.decoder2(latent2_se)
            view1_r_diversity = self.decoder1_single(latent1_diversity_se)
            view2_r_diversity = self.decoder2_single(latent2_diversity_se)
        else:
            view1_r = self.decoder1(latent1)
            view2_r = self.decoder2(latent2)
            view1_r_diversity = self.decoder1_single(diversity_latent_1)
            view2_r_diversity = self.decoder2_single(diversity_latent_2)

        # print(latent1.shape, view1_r.shape, view1_r_diversity.shape)
        # print(latent2.shape, view2_r.shape, view2_r_diversity.shape)

        if self.ft:
            return view1_r, view2_r, view1_r_diversity, view2_r_diversity, z1, z2, z_common, diversity_latent_1, \
                   diversity_latent_2, latent1, latent2, latent1_diversity_se, latent2_diversity_se, latent1_se, latent2_se
        else:
            return view1_r, view2_r, view1_r_diversity, view2_r_diversity


class MsDSCN():
    def __init__(self,
                 views_data,
                 n_samples,
                 label,
                 device=torch.device('cpu'),
                 learning_rate=1e-3,
                 weight_decay=0.00,
                 epochs=600,
                 ft=False,
                 random_seed=41,
                 alpha=1,
                 beta=0.1,
                 theta=0.1,
                 lamda=0.1,
                 loss_type='mse',
                 model_path=None,
                 show_res=10):
        self.views_data = views_data
        self.learning_rate = learning_rate
        self.device = device
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.ft = ft
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda = lamda
        self.loss_type = loss_type
        self.batch_size = n_samples
        self.model_path = model_path
        self.show_res = show_res
        self.label = label

        fix_seed(random_seed)

    def HSIC(self, c_v, c_w):
        N = c_v.shape[0]
        H = torch.ones((N, N)) * ((1 / N) * (-1)) + torch.eye(N)
        H = H.cuda()
        K_1 = torch.matmul(c_v, c_v.t()).cuda()
        K_2 = torch.matmul(c_w, c_w.t()).cuda()
        rst = torch.matmul(K_1, H).cuda()
        rst = torch.matmul(rst, K_2).cuda()
        rst = torch.matmul(rst, H).cuda()
        rst = torch.trace(rst).cuda()
        return rst

    def train(self):
        views_data = self.views_data
        views_data[0] = views_data[0].to(self.device)
        views_data[1] = views_data[1].to(self.device)

        model = AutoEncoderInit(batch_size=self.batch_size, ft=self.ft)
        model = model.to(self.device)

        if self.ft:  # load parameters from the init_pretrained_autoencoder
            print("============loading pretrained params============")
            pre_trained = torch.load(self.model_path, map_location=self.device)
            if not hasattr(pre_trained, "named_parameters"):
                model.load_state_dict(pre_trained, strict=False)
            else:
                parameters_initAE = dict(pre_trained.named_parameters())
                for name, param in model.named_parameters():
                    if name in parameters_initAE:
                        param.data.copy_(parameters_initAE[name].data)
        print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_values = []
        epoch_iter = tqdm(range(self.epochs))
        for epoch in epoch_iter:
            model.train()
            if self.ft:
                view1_out, view2_out, view1_r_out, view2_r_out, z1, z2, z_common, diversity_latent_1, \
                diversity_latent_2, latent1, latent2, latent1_diversity_se, latent2_diversity_se, latent1_se, \
                latent2_se = model(views_data)
                view1_rec_loss = torch.sum(torch.pow(view1_out - views_data[0], 2.0)) + torch.sum(torch.pow(view1_r_out - views_data[0], 2.0))
                view2_rec_loss = torch.sum(torch.pow(view2_out - views_data[1], 2.0)) + torch.sum(torch.pow(view2_r_out - views_data[1], 2.0))
                view_1_se_loss = torch.sum(torch.pow(latent1 - latent1_se, 2.0)) + torch.sum(torch.pow(diversity_latent_1 - latent1_diversity_se, 2.0))
                view_2_se_loss = torch.sum(torch.pow(latent2 - latent2_se, 2.0)) + torch.sum(torch.pow(diversity_latent_2 - latent2_diversity_se, 2.0))
                # loss of reconstruction
                reconstruct_loss = view1_rec_loss + view2_rec_loss
                # loss of self-expression
                expression_loss = view_1_se_loss + view_2_se_loss
                # cof regularization
                reg_loss = torch.sum(torch.pow(z1, 2.0)) + torch.sum(torch.pow(z2, 2.0)) + torch.sum(torch.pow(z_common, 2.0))
                # unify loss
                unify_loss = torch.sum(torch.abs(z_common - z1)) + torch.sum(torch.abs(z_common - z2))
                hsic_loss = self.HSIC(z1, z2)
                loss = reconstruct_loss + self.alpha * expression_loss + self.beta * reg_loss + 0.1 * unify_loss + 0.1 * hsic_loss
                if (epoch + 1) % self.show_res == 0:
                    alpha = max(0.4 - (self.label.shape[0] - 1) / 10 * 0.1, 0.1)
                    Coef = thrC(z_common.detach().cpu().numpy(), alpha)
                    y_hat, L = post_proC(Coef, self.label.max(), 3, 1)
                    missrate_x = err_rate(self.label, y_hat)
                    acc_x = 1 - missrate_x
                    nmi = normalized_mutual_info_score(self.label, y_hat)
                    f_measure = f1_score(self.label, y_hat)
                    ri = rand_index_score(self.label, y_hat)
                    ar = adjusted_rand_score(self.label, y_hat)
                    print("nmi: %.4f" % nmi, "accuracy: %.4f" % acc_x, "F-measure: %.4f" % f_measure, "RI: %.4f" % ri,
                          "AR: %.4f" % ar)

            else:
                view1_out, view2_out, view1_r_out, view2_r_out = model(views_data)
                # loss = F.mse_loss(data, out, reduction="none")
                # loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
                loss1 = 0.5 * torch.sum(torch.pow(torch.sub(view1_out, views_data[0]), 2.0))
                loss2 = 0.5 * torch.sum(torch.pow(torch.sub(view2_out, views_data[1]), 2.0))
                loss3 = 0.5 * torch.sum(torch.pow(torch.sub(view1_r_out, views_data[0]), 2.0))
                loss4 = 0.5 * torch.sum(torch.pow(torch.sub(view2_r_out, views_data[1]), 2.0))
                loss = loss1 + loss2 + loss3 + loss4
                # loss = loss1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.ft:
                epoch_iter.set_description(f"# Epoch {epoch}, train_loss: {loss.item():.4f}, "
                                           f"rec_loss: {reconstruct_loss.item():.4f}, "
                                           f"self_exp_loss: {expression_loss.item():.4f}, "
                                           f"reg_loss: {reg_loss.item():.4f}, "
                                           f"hisc-loss: {hsic_loss.item():.4f}")
            else:
                epoch_iter.set_description(
                    f"# Epoch {epoch}, train_loss: {loss.item():.4f}, loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}, loss3: {loss3.item():.4f},loss4: {loss4.item():.4f}")
            loss_values.append(loss.item())
        plt.plot(np.linspace(1, self.epochs, self.epochs).astype(int), loss_values)
        return model


def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == "__main__":
    origin_data = sio.loadmat('data/rgbd_mtv.mat')
    label = origin_data['gt'][:, 0]
    all_features = origin_data['X']

    view_shape = []
    views = []
    for v in all_features[0]:
        view_shape.append(v.shape[1])
        views.append(v)

    labelSubjects = np.array(label[0: 500])
    labelSubjects = labelSubjects - labelSubjects.min() + 1
    labelSubjects = np.squeeze(labelSubjects)

    # 我们先从single_view 开始，以第一个视图为例
    single_view = views[0]
    num_classes = np.unique(label).shape[0]

    reg1 = 1.0
    reg2 = 1.0
    alpha = max(0.4 - (num_classes - 1) / 10 * 0.1, 0.1)
    lr = 1e-3
    views[0] = np.transpose(views[0], [0, 3, 1, 2])
    views[1] = np.transpose(views[1], [0, 3, 1, 2])
    del views[2]

    tensors = [torch.from_numpy(arr) for arr in (views)]

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    init_model = MsDSCN(views_data=tensors, n_samples=label.shape[0], device=device, learning_rate=1e-3, epochs=10000, ft=False, label=labelSubjects)
    model = init_model.train()

    tensors[0] = tensors[0].to('cuda')
    tensors[1] = tensors[1].to('cuda')
    view1_out, view2_out, view1_r_out, view2_r_out = model(tensors)

    rec_data = view1_out.detach().cpu().numpy()
    plt.imshow(np.transpose(np.clip(rec_data[100], 0, 1), [1, 2, 0]))
    plt.imshow(np.transpose(np.clip(views[0][100], 0, 1), [1, 2, 0]))
    torch.save(model.state_dict(), 'result/mvc_pretrained_auto_enc.pt')
    # model = torch.load('pre_train_models/mvc_pretrained_auto_enc.pt')

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self_exp_ae = MsDSCN(views_data=tensors, n_samples=label.shape[0], device=device, learning_rate=1e-3, epochs=100, ft=True, model_path='result/mvc_pretrained_auto_enc.pt' ,label=labelSubjects, show_res=10)
    self_exp_model = self_exp_ae.train()

    view1_out, view2_out, view1_r_out, view2_r_out, z1, z2, z_common, diversity_latent_1, diversity_latent_2, latent1, latent2, latent1_diversity_se, latent2_diversity_se, latent1_se, latent2_se = self_exp_model(tensors)

    rec_data = view1_out.detach().cpu().numpy()
    plt.imshow(np.transpose(np.clip(rec_data[100], 0, 1), [1, 2, 0]))

    fig = plt.figure(figsize=(10,8))
    ax = plt.gca()
    cax = plt.imshow(z_common.detach().cpu().numpy())
    # set up colorbar
    cbar = plt.colorbar(cax, extend='both', drawedges = False)
    cbar.set_label('Intensity',size=36, weight='bold')
    cbar.ax.tick_params(labelsize=18)
    plt.show()

    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    # show clustering results
    from metric import thrC, post_proC, err_rate, f1_score, rand_index_score

    alpha = max(0.4 - (num_classes - 1) / 10 * 0.1, 0.1)
    Coef = thrC(z_common.detach().cpu().numpy(), alpha)
    sio.savemat('./result/rgbd_coef.mat', dict([('coef', Coef)]))
    y_hat, L = post_proC(Coef, labelSubjects.max(), 3, 1)
    missrate_x = err_rate(labelSubjects, y_hat)
    acc_x = 1 - missrate_x
    nmi = normalized_mutual_info_score(labelSubjects, y_hat)
    f_measure = f1_score(labelSubjects, y_hat)
    ri = rand_index_score(labelSubjects, y_hat)
    ar = adjusted_rand_score(labelSubjects, y_hat)
    print("nmi: %.4f" % nmi, "accuracy: %.4f" % acc_x, "F-measure: %.4f" % f_measure, "RI: %.4f" % ri, "AR: %.4f" % ar)
