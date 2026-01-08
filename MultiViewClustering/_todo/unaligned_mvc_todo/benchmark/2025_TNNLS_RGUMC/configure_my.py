def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(# [48,40,254,1984,512] [2358]
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                seed=4,
                missing_rate=1,#0.5,
                start_dual_prediction=100,
                batch_size=128,#256,# 2048,#
                epoch=50,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=0.9,# cross-view and inter-view loss
                lambda1=1e-1,  # 1e-2,# Z-norm
                lambda2=1e3,  # sample structure loss: ||(Z^1)(Z^1)^T- (Z^2)(Z^2)^T||_2^2: N*N # KL-divergence
                lambda3=1e-5,  # feature structure loss: # ||(Z^1)^T(Z^2)||_2^2: D*D    # loss_centriod_sample_distance
                lambda4=0,  # feature structure loss with Y # ||(Z^v-Y^v C^v)||_2^2 # loss_centriod_centriod_distance
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=1e3,  # 1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 128],
                arch2=[59, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#4485,#256,#2048,
                epoch=50,#500,
                lr=1.0e-4,
                alpha=10,#The entropy parameter α
                lambda1=1e-1,  # 1e-2,# Z-norm
                lambda2=1e3,  # sample structure loss: ||(Z^1)(Z^1)^T- (Z^2)(Z^2)^T||_2^2: N*N # KL-divergence
                lambda3=1e-2,  # feature structure loss: # ||(Z^1)^T(Z^2)||_2^2: D*D    # loss_centriod_sample_distance
                lambda4=0,  # feature structure loss with Y # ||(Z^v-Y^v C^v)||_2^2 # loss_centriod_centriod_distance
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=0,  # 1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]


            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[20, 1024, 1024, 1024, 64],# 20；59；40
                arch2=[40, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=0,
                seed=3,
                start_dual_prediction=100,
                epoch=50,#500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=1e-1,  # 1e-2,# Z-norm
                lambda2=1e3,  # sample structure loss: ||(Z^1)(Z^1)^T- (Z^2)(Z^2)^T||_2^2: N*N # KL-divergence
                lambda3=1e-2,  # feature structure loss: # ||(Z^1)^T(Z^2)||_2^2: D*D    # loss_centriod_sample_distance
                lambda4=0,  # 1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=1,  # 1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]
            ),
        )

    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[784, 1024, 1024, 1024, 64],
                arch2=[784, 1024, 1024, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5
                seed=0,
                start_dual_prediction=100,
                epoch=50,#500,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
            ),
        )



    elif data_name in ['digit_5view']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[76, 1024, 1024, 1024, 128],
                arch2=[216, 1024, 1024, 1024, 128],
                arch3=[64, 1024, 1024, 1024, 128],
                arch4=[240, 1024, 1024, 1024, 128],
                arch5=[47, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#1024,#1024,#512,#256,#4485,#256,#2048,
                epoch=50,
                lr=1.0e-4,
                alpha=0.9,#10,# The entropy parameter α
                # 关于lambda1
                # distance_loss: model_my_0617
                # the trade-off hyper-parameters λ1(clustering loss): model_my_0613
                lambda1=0,#1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=0,#1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=0,#1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=1,#1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,# clustering loss
                lambda6=0,#1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,#1e3,#0,# loss_var
                lambda8=0,#1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,#1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,#1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,#1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )


    elif data_name in ['Caltech101-20-5views']:
        """The default configs.""" # #[48,40,254,1984,512] [2358]
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[48, 1024, 1024, 1024, 128],
                arch2=[40, 1024, 1024, 1024, 128],
                arch3=[254, 1024, 1024, 1024, 128],
                arch4=[1984, 1024, 1024, 1024, 128],
                arch5=[512, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#1024,#1024,#512,#256,#4485,#256,#2048,
                epoch=50,
                lr=1.0e-4,
                alpha=10,#10,# The entropy parameter α
                # 关于lambda1
                # distance_loss: model_my_0617
                # the trade-off hyper-parameters λ1(clustering loss): model_my_0613
                lambda1=0,#1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=1,#1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-3,#1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,#1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,# clustering loss
                lambda6=0,#1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,#1e3,#0,# loss_var
                lambda8=0,#1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,#1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,#1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,#1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )


    elif data_name in ['flower17_5views']:
        """The default configs.""" # #[48,40,254,1984,512] [2358]
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1360, 1024, 1024, 1024, 128],
                arch2=[1360, 1024, 1024, 1024, 128],
                arch3=[1360, 1024, 1024, 1024, 128],
                arch4=[1360, 1024, 1024, 1024, 128],
                arch5=[1360, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#1024,#1024,#512,#256,#4485,#256,#2048,
                epoch=50,
                lr=1.0e-4,
                alpha=10,#10,# The entropy parameter α
                # 关于lambda1
                # distance_loss: model_my_0617
                # the trade-off hyper-parameters λ1(clustering loss): model_my_0613
                lambda1=0,#1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=1,#1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-3,#1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,#1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,# clustering loss
                lambda6=0,#1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,#1e3,#0,# loss_var
                lambda8=0,#1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,#1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,#1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,#1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )

    elif data_name in ['reuters']:
        """The default configs."""# [21526,24892,34121,15487, 11539] [2358]
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[21526, 1024, 1024, 1024, 128],
                arch2=[24892, 1024, 1024, 1024, 128],
                arch3=[34121, 1024, 1024, 1024, 128],
                arch4=[15487, 1024, 1024, 1024, 128],
                arch5=[11539, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#1024,#1024,#512,#256,#4485,#256,#2048,
                epoch=20,
                lr=1.0e-4,
                alpha=10,#10,# The entropy parameter α
                # 关于lambda1
                # distance_loss: model_my_0617
                # the trade-off hyper-parameters λ1(clustering loss): model_my_0613
                lambda1=0,#1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=0,#1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-3,#1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,#1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,# clustering loss
                lambda6=0,#1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,#1e3,#0,# loss_var
                lambda8=0,#1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,#1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,#1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,#1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )

    elif data_name in ['digit_2view']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[76, 1024, 1024, 1024, 128],#128
                arch2=[216, 1024, 1024, 1024, 128],#128
                # arch1=[76, 1024, 1024, 1024, 216],
                # arch2=[216, 1024, 1024, 1024, 216],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#1024,#1024,#512,#256,#4485,#256,#2048,
                epoch=50, #50,#50,
                lr=1.0e-4,
                alpha=0.5,#10,# The entropy parameter α

                ## KL-UMC 最优超参数   (单个视图指导)
                lambda1=1e-1,  # 1e-2,# Z-norm
                lambda2=1e3,  # sample structure loss: ||(Z^1)(Z^1)^T- (Z^2)(Z^2)^T||_2^2: N*N # KL-divergence
                lambda3=1e-2,  # feature structure loss: # ||(Z^1)^T(Z^2)||_2^2: D*D    # loss_centriod_sample_distance

                ## WKL-UMC 最优超参数  （多个视图指导）
                # lambda1=1,  # 1e-2,# Z-norm
                # lambda2=1e4,  # sample structure loss: ||(Z^1)(Z^1)^T- (Z^2)(Z^2)^T||_2^2: N*N # KL-divergence
                # lambda3=1e-1,  # feature structure loss: # ||(Z^1)^T(Z^2)||_2^2: D*D    # loss_centriod_sample_distance

                lambda4=0,  # feature structure loss with Y # ||(Z^v-Y^v C^v)||_2^2 # loss_centriod_centriod_distance
                lambda5=0,  # 聚类中心C的正交约束
                k_means_max_iter=1e1,#1e1
                tolerate=1e-3,
                tau_cross=1e-1,# 匈牙利算法计算聚类中心相似度，温度系数
            ),
        )

    elif data_name in ['flower17_2views']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1360, 1024, 1024, 1024, 128],
                arch2=[1360, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#4485,#256,#2048,
                epoch=50,#500,
                lr=1.0e-4,
                alpha=0.01,#The entropy parameter α
                lambda1=1e2,#1e1,  # 1e-2,# Z-norm
                lambda2=1e4,  # sample structure loss: ||(Z^1)(Z^1)^T- (Z^2)(Z^2)^T||_2^2: N*N # KL-divergence
                lambda3=1e-2,  # feature structure loss: # ||(Z^1)^T(Z^2)||_2^2: D*D    # loss_centriod_sample_distance
                lambda4=0,  # feature structure loss with Y # ||(Z^v-Y^v C^v)||_2^2 # loss_centriod_centriod_distance
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=0,  #  completer 方法中的对比方法 只要互信息
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )

    elif data_name in ['reuters_2views']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[21531, 1024, 1024, 1024, 128],
                arch2=[24892, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#4485,#256,#2048,
                epoch=50,#500,
                lr=1.0e-4,
                alpha=10,#The entropy parameter α
                lambda1=1e-1,  # 1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=1e3,  # 1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-2,
                # 1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,  # 1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=0,  # 1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )


    elif data_name in ['AWA_2views']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[2688, 1024, 1024, 1024, 128],
                arch2=[2000, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=2048,#4485,#256,#2048,
                epoch=50,#500,
                lr=1.0e-4,
                alpha=9,#The entropy parameter α
                lambda1=1e-1,  # 1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=1e3,#1e2,  # 1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-2,#1e-2,
                # 1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,#1,  # 1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=0,  # 1e-3,#  The entropy hyper-parameter  , match to alpha,
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本
            ),
        )


    elif data_name in ['YouTuBe']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[2000, 1024, 1024, 1024, 128],
                arch2=[1000, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,  # 0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,  # 4485,#256,#2048,
                epoch=50,  # 500,
                lr=1.0e-4,
                alpha=9,  # The entropy parameter α
                lambda1=1e-1,  # 1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=1e3,  # 1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-2,
                # 1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,  # 1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=0,  # 1e-3,#  The entropy hyper-parameter  , match to alpha,
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本
            ),
        )


    elif data_name in ['reuters_views_18758']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[21531, 1024, 1024, 1024, 128],
                arch2=[24892, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,  # 0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,  # 4485,#256,#2048,
                epoch=50,  # 500,
                lr=1.0e-4,
                alpha=9,  # The entropy parameter α, between inter-view contrastive and cross-view contrastive loss
                # RG-UMC
                lambda1=1e1,  # 1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=1e3,#1e2,  # 1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-2,#1e-2,

                # 1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,  # 1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,#1,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=0,  # 1e-3,#  The entropy hyper-parameter  , match to alpha,
                tau_inter=1e-1,  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本
            ),
        )




    elif data_name in ['reuters_views_111740']:
        """The default configs."""# [21526,24892,34121,15487, 11539] [2358]
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
                arch3=[128, 256, 128],
                arch4=[128, 256, 128],
                arch5=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[21526, 1024, 1024, 1024, 128],
                arch2=[24892, 1024, 1024, 1024, 128],
                arch3=[34121, 1024, 1024, 1024, 128],
                arch4=[15487, 1024, 1024, 1024, 128],
                arch5=[11539, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                activations3='relu',
                activations4='relu',
                activations5='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=256,#1024,#1024,#512,#256,#4485,#256,#2048,
                epoch=50,
                lr=1.0e-4,
                alpha=10,#10,# The entropy parameter α
                # 关于lambda1
                # distance_loss: model_my_0617
                # the trade-off hyper-parameters λ1(clustering loss): model_my_0613
                lambda1=0,#1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=0,#1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-3,#1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,#1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,# clustering loss
                lambda6=0,#1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,#1e3,#0,# loss_var
                lambda8=0,#1e-3,#  The entropy hyper-parameter  , match to alpha,# completer 方法中的对比方法
                tau_inter=1e-1,#1e-1,#1e4,#loss_inter_contrastive
                tau_cross=1e-1,#1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,#1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )




    elif data_name in ['office31_2views']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[2048, 1024, 1024, 1024, 128],
                arch2=[2048, 1024, 1024, 1024, 128],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                missing_rate=1,#0.5,
                seed=8,
                start_dual_prediction=100,
                batch_size=498,#256,#4485,#256,#2048,
                epoch=50,#500,
                lr=1.0e-4,
                alpha=0.01,#The entropy parameter α
                lambda1=1e-1,  # 1e-3,#1e-3,#1,#1e1,#1e-3,#0,#1e-3,# 0.1,# loss_Hungary_alignment
                lambda2=1e3,  # 1e3,#1e3,#1e2,#1e3,# 0.1,# and λ2(contrastive loss):loss_inter_contrastive
                lambda3=1e-2,
                # 1e1,#1e1,#1e3,#1e1,#1e1,#1e-1,#1e-1,# 0.1,# and λ2(contrastive loss):loss_cross_contrastive
                lambda4=0,  # 1,#1e-5,#0,#1e-3,# loss_z_norm
                lambda5=0,  # clustering loss
                lambda6=0,  # 1e-3,#1e-3,#loss_cross_entrypy_alignment## 从分布角度考虑, 约束单视图聚类中心与多视图聚类中心对齐；
                lambda7=0,  # 1e3,#0,# loss_var
                lambda8=0,  #  completer 方法中的对比方法 只要互信息
                tau_inter=[1e-1],  # 1e-1,#1e4,#loss_inter_contrastive
                tau_cross=[1e-1],  # 1,#1e-1,#loss_inter_contrastive
                k_means_max_iter=1e1,  # 1e1
                tolerate=1e-3,
                epsilon=0.5,  # 选择置信样本  [0.5-1]

            ),
        )


    else:
        raise Exception('Undefined data_name')
