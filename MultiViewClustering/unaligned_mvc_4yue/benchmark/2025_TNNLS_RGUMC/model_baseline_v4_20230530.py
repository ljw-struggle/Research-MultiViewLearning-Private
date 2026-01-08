import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
import evaluation
from util import next_batch
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))

                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent

class Clustering(nn.Module):
    def __init__(self, K,d):
        super(Clustering, self).__init__()
        # input_A = 784  input_B = 784
        #self.commonz = input1
        self.weights = nn.Parameter(torch.randn(K,d).cuda(), requires_grad=True)
#        self.layer1 = nn.Linear(d, K, bias = False)

    def forward(self, comz):
        q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(comz, 1) - self.weights, 2), 2)))
        q = torch.t(torch.t(q1) / torch.sum(q1))
        loss_q = torch.log(q)
        return loss_q, q

class Completer_my_20220605():
    """COMPLETER module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])

        # Dual predictions.
        # To illustrate easily, we use "img" and "txt" to denote two different views.
        self.img2txt = Prediction(self._dims_view1)
        self.txt2img = Prediction(self._dims_view2)

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.img2txt.to(device)
        self.txt2img.to(device)




    def KL_divergence(self, cov1, cov2, mean_1, mean_2):  # cov_s, cov_t, source, target
        """
        =========没看懂=========
        https://zhuanlan.zhihu.com/p/438129018
        就是当作多元高斯分布求的KL散度
        :param cov1: cov_s, 源域32×32协方差
        :param cov2: cov_t, 目标域32×32协方差
        :param mean_1: source, 源域1×32维均值
        :param mean_2: target, 目标域1×32维均值
        :return: KL散度值
        """
        trace_ = torch.trace(cov1.inverse() @ cov2)  # torch.trace算矩阵的迹，对角线元素之和, 协方差矩阵的最大值
        logs_ = torch.logdet(cov1) - torch.logdet(cov2)  # torch.logdet计算一个方阵或几批方阵的对数行列式
        mean_sub = mean_1 - mean_2
        manhanobis = (mean_sub @ cov1.inverse() @ mean_sub.t()).squeeze()
        res = (trace_ + logs_ + manhanobis) / 2
        return res


    def normalize_rows(self, array):
        row_sums = array.sum(axis=1)  # 计算每行的和
        normalized_array = array / row_sums[:, np.newaxis]  # 将每个元素除以对应行的和
        return normalized_array

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def train(self, config, logger, x_train, Y_list, mask, optimizer, device, tmp_idx):
    # def train(self, config, logger, x_train, Y_list, mask, optimizer, device, tmp_idx, X_train_raw, Y_list_raw, a):# #测试时用未shuffle的数据

        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """

        # Get complete data for training
        view_num = len(x_train)
        for v in range(view_num):
            x_train[v] = torch.as_tensor(x_train[v]).to(device)
        train_view = x_train.copy()
        Y_lable_1 = Y_list[:len(train_view[0])]
        Y_lable_2 = Y_list[len(train_view[0]):]

        # Get the unpaired data for training
        n_clusters = np.unique(Y_list)
        K = len(n_clusters)
        d_subspace = 128#128#512#256#
        n_sample = len(Y_list)



        # training
        '''show loss and show measure'''
        loss_show, loss_show_silhouette, loss_show_clustering, loss_show_rec1, loss_show_rec2 = [], [], [], [], []
        loss_show_z_norm, loss_show_structure = [], []
        loss_show_fea_structure, loss_show_withY_structure, loss_show_KL_divergence = [], [], []
        acc_show, nmi_show, precision_show, F_measure_show  = [], [], [], []
        recall_show, ARI_show, AMI_show = [], [], []
        result_single_view1_all, result_single_view2_all = [], []
        loss_show_centeriod_orth = []

        p_y1, log_unreliable1 = [], []
        for epoch in range(config['training']['epoch']):
            # 数据训练时shuffle
            shuffle_idx = np.arange(len(train_view[0]))
            X1, X2, shuffle_idx = shuffle(train_view[0], train_view[1], shuffle_idx)
            '''中间值'''
            loss_all, loss_rec1, loss_rec2, loss_z_norm1, loss_structure1, loss_fea_structure1, loss_withY_structure1, \
            loss_KL_divergence \
                = 0, 0, 0, 0, 0, 0, 0, 0
            loss_centeriod_orth1, loss_KL_divergence1 = 0, 0


            for batch_x1, batch_x2, batch_No in next_batch(X1, X2, config['training']['batch_size']):
                structure_loss, structure_loss_fea, structure_loss_withY = 0, 0, 0
                # loss_centriod_sample_distance, loss_centriod_centriod_distance = 0, 0
                KL_divergence_loss, structure_loss_centeriod_orth = 0, 0
                # encoder
                z_before = []
                z_before.append(self.autoencoder1.encoder(batch_x1))
                z_before.append(self.autoencoder2.encoder(batch_x2))
                z = []

                z.append(torch.as_tensor(z_before[0]).clone())
                z.append(torch.as_tensor(z_before[1]).clone())
                commonZ_term = []
                commonZ_term.append(torch.as_tensor(z_before[0]).clone().tolist)
                commonZ_term.append(torch.as_tensor(z_before[1]).clone().tolist)


                 # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_before[0]), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_before[1]), batch_x2)
                reconstruction_loss = recon1 + recon2
                # print('reconstruction_loss', reconstruction_loss)


                '''20220521, # L2正则化，L2_norm'''
                loss_z_norm = torch.norm(
                        z_before[0].t() @ z_before[0] - torch.as_tensor(np.eye(d_subspace).astype(np.float32)).to(
                            device), p=2) + \
                    torch.norm(
                        z_before[1].t() @ z_before[1] - torch.as_tensor(np.eye(d_subspace).astype(np.float32)).to(
                            device), p=2)



                '''20230312  silhouette coefficient: choose the anchor sample '''
                # step1: k-means 得到cluster_labels
                # step2: compute silhouette of each sample
                # step3: choose the silhouette score eq to 1, (weight)
                commonZ_term_before = []
                for v in range(view_num):
                    commonZ_term_before.append(torch.as_tensor(z_before[v]).clone())

                silhouette_score_avg, silhouette_score_sample, centroids_views, label_views, label_views_matrix =[], [], [], [], []
                sil_idx, centroids_views1, weight = [], [], []

                commonZ_term_before_anchor1 = []
                for v in range(view_num):
                    estimator = KMeans(K).fit(commonZ_term_before[v].cpu().detach().numpy())
                    centroids_views.append(estimator.cluster_centers_)
                    label_pred = estimator.labels_
                    label_views.append([label_pred])
                    label_views_matrix.append(self.label_to_matrix(label_pred).astype(np.float32))

                    # label_views_matrix.append(np.diag(label_pred))
                    '''20230312  silhouette coefficient: choose the most relaible view'''
                    silhouette_score_avg1 = silhouette_score(commonZ_term_before[v].cpu().detach().numpy(), label_pred)
                    silhouette_score_avg.append(silhouette_score_avg1)
                    silhouette_score_sample1 = silhouette_samples(commonZ_term_before[v].cpu().detach().numpy(), label_pred)  # 轮廓系数
                    '''compute the number of silhouette_score > 0'''  '''compute the number of silhouette_score > silhouette_score_avg1'''


                    '''找到 轮廓系数 >0 元素的下标，作为anchor样本'''
                    sil_idx = (np.nonzero(silhouette_score_sample1 > silhouette_score_avg1))
                    # print(np.nonzero(silhouette_score_sample1 > 0))
                    silhouette_score_sample.append(silhouette_score_sample1)
                    # choose the sample with silhouette_score > 0
                    commonZ_term_before_anchor = commonZ_term_before[v][sil_idx, :]
                    commonZ_term_before_anchor = torch.squeeze(commonZ_term_before_anchor).to(device)
                    commonZ_term_before_anchor1.append(commonZ_term_before_anchor)
                    centroids_views1.append(torch.as_tensor(centroids_views[v]).clone())

                    '''增加权重系数，轮廓系数[-1,1]->[0-1], -> [0.5, 1]'''
                    # aa = (silhouette_score_sample1 + 1) / 2
                    # a.append(aa)
                    weight.append(np.eye(len(commonZ_term_before[v])).astype(np.float32))
                    # weight.append(np.diag( (silhouette_score_sample1 + 1) / 2))  # wieght 用轮廓系数表示[0,1]
                    # weight.append(np.diag(0.5 + (silhouette_score_sample1 + 1) / 4))# wieght 用轮廓系数表示[0.5, 1]
                    # weight.append(np.diag(0.75 + (silhouette_score_sample1 + 1) / 8))  # wieght 用轮廓系数表示[0.75, 1]
                    # weight.append(np.diag(0.8 + (silhouette_score_sample1 + 1) / 10))  # wieght 用轮廓系数表示[0.8, 1]# 测试最有效！
                    # weight.append(np.diag(0.875 + (silhouette_score_sample1 + 1) / 16))  # wieght 用轮廓系数表示[0.875, 1]
                    # weight.append(np.diag(0.75 + ((silhouette_score_sample1 + 1) / 4)*((silhouette_score_sample1 + 1) / 4)))# wieght 用轮廓系数表示[0.75, 1]

                weight = torch.as_tensor(weight).to(device)



                '''KL散度使得向更可靠视图靠近'''
                '''-----------------style1:  KL divergence, 用 reliable view 指导 unreliable views, global--------------------'''
                ''' https://blog.csdn.net/Answer3664/article/details/106265132/ '''
                '''record the most reliable view'''
                reliable_view_idx = np.where(
                    silhouette_score_avg == np.max(silhouette_score_avg))  # 记录最可靠的聚类视图，其他视图向其靠近
                for v in range(view_num):
                    p_y = F.softmax(commonZ_term_before[reliable_view_idx[0][0]], dim=1)
                    if v != reliable_view_idx:
                        '''利用离散方式'''
                        log_unreliable = F.log_softmax(weight[v] @ commonZ_term_before[v], dim=1)
                        KL_divergence_loss += F.kl_div(log_unreliable, p_y, reduction='sum')



                '''style2: 多个视图之间互相指导'''
                " [v1, v2,v3,v4, v5], 自动选择若干个指导的视图，每个视图的权重为1/sum(sil)"
                '''sigmoid 函数y取值为0-1之间。且x取值在0-1范围内有较好的小值变大，大值变小的性质'''

                # for v in range(view_num):
                #     log_unreliable = F.log_softmax(commonZ_term_before[v], dim=1)  # for unreliable view
                #     reliable_view_idx = np.where(silhouette_score_avg >= silhouette_score_avg[v])
                #     reliable_view_idx_weight, reliable_view_idx_weight_tmp = [], []
                #     for i in range(len(reliable_view_idx[0])):
                #         reliable_view_idx_weight_tmp.append(self.sigmoid(silhouette_score_avg[reliable_view_idx[0][i]]))
                #     # 归一化数组
                #     reliable_view_idx_weight = self.normalize_rows(np.array([reliable_view_idx_weight_tmp]))
                #     for v_t in range(len(reliable_view_idx[0])):
                #         p_y = F.softmax(commonZ_term_before[reliable_view_idx[0][v_t]], dim=-1)  # # reliable view
                #         KL_divergence_loss += reliable_view_idx_weight[0][v_t] * F.kl_div(log_unreliable, p_y,
                #                                                                       reduction='sum')


                #1: 检查不同聚类簇中样本的个数；
                '''Compactness, 每个聚类簇中的样本点到聚类中心的平均距离'''
                for v in range(view_num):
                    for c in range(K):#label_pred# 预测的样本分配 centroids_views[v] #聚类中心 commonZ_term_before[0]# 样本
                        c_idx = np.array(np.where(label_pred == c))
                        prototypes_tmp = np.tile(centroids_views[v][c], (len(c_idx[0]), 1))
                        x_tmp = commonZ_term_before[v][c_idx]
                        prototypes_tmp = torch.as_tensor(prototypes_tmp).to(device)
                        dist = pow(torch.norm(torch.as_tensor(x_tmp - prototypes_tmp).to(device), p=2), 2)
                        structure_loss_fea += dist


                KL_divergence_loss = KL_divergence_loss / (view_num*view_num)

                loss = reconstruction_loss + loss_z_norm * config['training']['lambda1'] + KL_divergence_loss * \
                       config['training']['lambda2'] + structure_loss_fea * config['training']['lambda3']


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_z_norm1 += loss_z_norm.item() * config['training']['lambda1']
                loss_KL_divergence1 += KL_divergence_loss.item() * config['training']['lambda2']
                loss_fea_structure1 += structure_loss_fea.item() * config['training']['lambda3']

            # 每个epoch结束计算测试样本的acc和nmi并记录
            '''用学得的autoencoder参数进行整体样本的评估，学得整体的子空间表示Z，然后用k-means得到Y_pre'''
            # 测试也用shuffle后数据
            scores = self.evaluation_my_v3(config, logger, mask, x_train, Y_list, device, tmp_idx)# Y_list 已经提前设置为[Y_v1; Y_v2]
            acc_baseline, nmi_baseline, Precision_baseline, F_measure_baseline, recall_baseline, ARI_baseline, AMI_baseline \
                    = scores[0]['kmeans']['accuracy'], scores[0]['kmeans']['NMI'], \
                      scores[0]['kmeans']['precision'], scores[0]['kmeans']['f_measure'], \
                      scores[0]['kmeans']['recall'], scores[0]['kmeans']['ARI'], scores[0]['kmeans']['AMI']
            result_baseline1 = np.array(
                [acc_baseline, nmi_baseline, Precision_baseline, F_measure_baseline, recall_baseline,
                 ARI_baseline, AMI_baseline]) * 100

            acc_show.extend([acc_baseline])
            nmi_show.extend([nmi_baseline])
            precision_show.extend([Precision_baseline])
            F_measure_show.extend([F_measure_baseline])
            recall_show.extend([recall_baseline])
            ARI_show.extend([ARI_baseline])
            AMI_show.extend([AMI_baseline])



        return scores[0]['kmeans']['accuracy'], scores[0]['kmeans']['NMI'], \
                      scores[0]['kmeans']['precision'], scores[0]['kmeans']['f_measure'], \
                      scores[0]['kmeans']['recall'], scores[0]['kmeans']['ARI'], scores[0]['kmeans']['AMI'], acc_show, \
               nmi_show, precision_show, F_measure_show, recall_show, ARI_show, AMI_show, result_single_view1_all, result_single_view2_all




    def evaluation_my_v3(self, config, logger, mask, x_train, Y_list, device, tmp_idx):
        # 正确！
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            self.img2txt.eval(), self.txt2img.eval()
            view_num = len(x_train)
            train_view = x_train.copy()
            z = []
            z.append(self.autoencoder1.encoder(train_view[0]))
            z.append(self.autoencoder2.encoder(train_view[1]))
            latent_fusion = torch.cat([z[0], z[1]], dim=0).cpu().numpy()  # 20220509

            scores = evaluation.clustering([latent_fusion], Y_list)
            logger.info("\033[2;29m" + 'view_concat ' + str(scores) + "\033[0m") #信息展示，时间+info+view_conncat+scores

            self.autoencoder1.train(), self.autoencoder2.train()  # 观测样本
            self.img2txt.train(), self.txt2img.train()  # 缺失样本
        return scores, latent_fusion, z, Y_list

