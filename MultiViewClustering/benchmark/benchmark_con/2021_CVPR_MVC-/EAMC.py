"""
EAMC (End-to-End Adversarial Multi-View Clustering) 模型
包含模型定义、训练函数和配置
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from _utils import (
    load_dataset, create_dataloader, calc_metrics, to_numpy,
    he_init_weights, cdist, vector_kernel, d_cs, triu
)


# ==================== 基础模块 ====================

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, layers, activation='relu', use_bn=False, use_bias=True):
        """
        layers: list of int, 每层的神经元数量 [input_dim, hidden1, hidden2, ..., output_dim]
        activation: 'relu', 'sigmoid', 'tanh', None
        use_bn: 是否使用BatchNorm
        use_bias: 是否使用偏置
        """
        super().__init__()
        self.layers_list = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1], bias=use_bias))
            if use_bn:
                self.layers_list.append(nn.BatchNorm1d(layers[i+1]))
            if activation == 'relu':
                self.layers_list.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.layers_list.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.layers_list.append(nn.Tanh())
    
    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        for layer in self.layers_list:
            x = layer(x)
        return x


class CNN(nn.Module):
    """简单的CNN"""
    def __init__(self, input_channels, conv_layers, fc_layers, activation='relu'):
        """
        conv_layers: list of (out_channels, kernel_size, stride, padding)
        fc_layers: list of int, 全连接层神经元数量
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        
        in_channels = input_channels
        for out_channels, kernel_size, stride, padding in conv_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            if activation == 'relu':
                self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
        
        # 全连接层
        if fc_layers:
            self.fc_layers.append(nn.Flatten())
            in_features = self._get_conv_output_size()
            for out_features in fc_layers:
                self.fc_layers.append(nn.Linear(in_features, out_features))
                if activation == 'relu':
                    self.fc_layers.append(nn.ReLU())
                in_features = out_features
    
    def _get_conv_output_size(self):
        # 简化版本，假设输入是28x28
        return 64 * 7 * 7  # 需要根据实际输入调整
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.fc_layers:
            x = layer(x)
        return x


class DDC(nn.Module):
    """深度判别聚类模块"""
    def __init__(self, input_dim, n_hidden, n_clusters, use_bn=True):
        super().__init__()
        self.n_clusters = n_clusters
        hidden_layers = [nn.Linear(input_dim, n_hidden), nn.ReLU()]
        if use_bn:
            hidden_layers.append(nn.BatchNorm1d(n_hidden))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(n_hidden, n_clusters), nn.Softmax(dim=1))
    
    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden


class Discriminator(nn.Module):
    """EAMC判别器"""
    def __init__(self, mlp, output_layer):
        super().__init__()
        self.mlp = mlp
        self.output_layer = output_layer
        self.d0 = None
        self.dv = None
    
    def forward(self, x0, xv):
        """接受两个输入x0和xv，返回[d0, dv]"""
        self.d0 = self.output_layer(self.mlp(x0))
        self.dv = self.output_layer(self.mlp(xv))
        return [self.d0, self.dv]


# ==================== 模型定义 ====================

class EAMC(nn.Module):
    """端到端对抗注意力多视图聚类模型"""
    def __init__(self, backbone_configs, n_hidden=128, n_clusters=10, use_attention=True,
                 use_discriminator=True, attention_tau=10.0, gamma=10.0, use_bn=True,
                 lr_backbones=1e-5, lr_clustering=1e-5, lr_att=1e-4, lr_disc=1e-3,
                 t=1, t_disc=1, clip_norm=0.5, device='cuda', discriminator_layers=None):
        """
        use_attention: 是否使用注意力机制
        use_discriminator: 是否使用判别器
        attention_tau: 注意力温度参数
        gamma: 生成器损失权重
        t: 生成器训练步数
        t_disc: 判别器训练步数
        """
        super().__init__()
        self.device = device
        self.n_views = len(backbone_configs)
        self.gamma = gamma
        self.t = t
        self.t_disc = t_disc
        self.clip_norm = clip_norm
        # 保存discriminator_layers供后续使用
        self._discriminator_layers = discriminator_layers if discriminator_layers is not None else [256, 256, 128]
        
        # 创建backbones
        self.backbones = nn.ModuleList()
        for cfg in backbone_configs:
            if cfg['type'] == 'mlp':
                bb = MLP(cfg['layers'], cfg.get('activation', 'relu'), 
                        cfg.get('use_bn', False), cfg.get('use_bias', True))
            elif cfg['type'] == 'cnn':
                bb = CNN(cfg['input_channels'], cfg['conv_layers'], 
                        cfg.get('fc_layers', []), cfg.get('activation', 'relu'))
            else:
                raise ValueError(f"Unknown backbone type: {cfg['type']}")
            self.backbones.append(bb)
        
        # 获取backbone输出维度（在模型移到device之前，所以使用CPU）
        with th.no_grad():
            dummy_input = th.randn(1, backbone_configs[0]['layers'][0] if backbone_configs[0]['type'] == 'mlp' else (1, 28, 28))
            dummy_output = self.backbones[0](dummy_input)  # 在CPU上计算
            hidden_dim = dummy_output.shape[1]
        
        # 注意力模块
        if use_attention:
            # 与原版一致：MLP layers=(100, 50), activation=None
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * self.n_views, 100),
                nn.Linear(100, 50),
                nn.Linear(50, self.n_views)
            )
            self.attention_tau = attention_tau
            self.weights = None
        else:
            self.attention = None
            self.weights = th.ones(self.n_views, device=device) / self.n_views
        
        # 判别器
        if use_discriminator:
            self.discriminators = nn.ModuleList()
            # 使用保存的discriminator_layers
            disc_layers = self._discriminator_layers
            for _ in range(self.n_views - 1):
                # 与原版一致：MLP layers，activation="leaky_relu:0.2"
                mlp_layers = []
                mlp_layers.append(nn.Linear(hidden_dim, disc_layers[0], bias=True))
                mlp_layers.append(nn.LeakyReLU(0.2))
                for i in range(len(disc_layers) - 1):
                    mlp_layers.append(nn.Linear(disc_layers[i], disc_layers[i+1], bias=True))
                    if i < len(disc_layers) - 2:  # 最后一层不加激活
                        mlp_layers.append(nn.LeakyReLU(0.2))
                mlp = nn.Sequential(*mlp_layers)
                output_layer = nn.Sequential(
                    nn.Linear(disc_layers[-1], 1, bias=True),
                    nn.Sigmoid()
                )
                # 包装成可以接受两个输入的判别器
                discriminator = Discriminator(mlp, output_layer)
                self.discriminators.append(discriminator)
        else:
            self.discriminators = None
        
        # 聚类模块
        self.ddc = DDC(hidden_dim, n_hidden, n_clusters, use_bn)
        
        # 优化器（与原版一致：使用不同的beta参数）
        clustering_optimizer_spec = [
            {"params": self.backbones.parameters(), "lr": lr_backbones, "betas": (0.95, 0.999)},
            {"params": self.ddc.parameters(), "lr": lr_clustering, "betas": (0.95, 0.999)}
        ]
        if use_attention:
            clustering_optimizer_spec.append(
                {"params": self.attention.parameters(), "lr": lr_att, "betas": (0.95, 0.999)}
            )
        self.clustering_optimizer = th.optim.Adam(clustering_optimizer_spec)
        
        if use_discriminator:
            self.discriminator_optimizer = th.optim.Adam(
                [{"params": [p for d in self.discriminators for p in d.parameters()], 
                  "lr": lr_disc, "betas": (0.5, 0.999)}]
            )
        else:
            self.discriminator_optimizer = None
        
        # 初始化权重
        self.apply(he_init_weights)
        
        self.backbone_outputs = None
        self.discriminator_outputs = None
        self.fused = None
        self.hidden = None
        self.output = None
    
    def forward(self, views):
        self.backbone_outputs = [bb(v) for bb, v in zip(self.backbones, views)]
        
        # 判别器输出
        if self.discriminators is not None:
            self.discriminator_outputs = [
                self.discriminators[i](self.backbone_outputs[0], self.backbone_outputs[i+1])
                for i in range(len(self.backbone_outputs) - 1)
            ]
        
        # 融合
        if self.attention is not None:
            h = th.cat(self.backbone_outputs, dim=1)
            act = self.attention(h)
            e = F.softmax(F.sigmoid(act) / self.attention_tau, dim=1)
            self.weights = th.mean(e, dim=0)
        else:
            self.weights = th.ones(self.n_views, device=self.device) / self.n_views
        
        self.fused = th.sum(self.weights[None, None, :] * th.stack(self.backbone_outputs, dim=-1), dim=-1)
        self.output, self.hidden = self.ddc(self.fused)
        return self.output
    
    def calc_losses(self, ignore_in_total=tuple(), rel_sigma=0.15):
        """计算损失"""
        losses = {}
        
        # DDC损失（使用与原版一致的命名：ddc_1, ddc_2_flipped, ddc_3）
        hidden_kernel = vector_kernel(self.hidden, rel_sigma)
        losses['ddc_1'] = d_cs(self.output, hidden_kernel, self.ddc.n_clusters)
        n = self.output.size(0)
        losses['ddc_2_flipped'] = 2 / (self.ddc.n_clusters * (self.ddc.n_clusters - 1)) * triu(self.output.t() @ self.output)
        eye = th.eye(self.ddc.n_clusters, device=self.device)
        m = th.exp(-th.cdist(self.output, eye))
        losses['ddc_3'] = d_cs(m, hidden_kernel, self.ddc.n_clusters)
        
        # 注意力损失
        if self.attention is not None and 'att' not in ignore_in_total:
            backbone_kernels = [vector_kernel(h, rel_sigma) for h in self.backbone_outputs]
            fusion_kernel = vector_kernel(self.fused, rel_sigma)
            kc = th.sum(self.weights[None, None, :] * th.stack(backbone_kernels, dim=-1), dim=-1)
            dif = (fusion_kernel - kc)
            losses['att'] = th.trace(dif @ dif.t())
        
        # 生成器损失
        if self.discriminators is not None and 'gen' not in ignore_in_total:
            gen_loss = th.tensor(0.0, device=self.device)
            target = th.ones(self.output.size(0), device=self.device)
            for d0, dv in self.discriminator_outputs:
                # 生成器希望判别器认为view_i+1也是真实的
                gen_loss += F.binary_cross_entropy(dv.squeeze(), target)
            losses['gen'] = self.gamma * gen_loss
        
        # 判别器损失
        if self.discriminators is not None and 'disc' not in ignore_in_total:
            disc_loss = th.tensor(0.0, device=self.device)
            real_target = th.ones(self.output.size(0), device=self.device)
            fake_target = th.zeros(self.output.size(0), device=self.device)
            for d0, dv in self.discriminator_outputs:
                # 判别器希望正确区分view0（真实）和view_i+1（虚假）
                disc_loss += F.binary_cross_entropy(d0.squeeze(), real_target) + \
                            F.binary_cross_entropy(dv.squeeze(), fake_target)
            losses['disc'] = disc_loss
        
        # 总损失
        losses['tot'] = sum([losses[k] for k in losses.keys() if k not in ignore_in_total and k != 'tot'])
        
        return losses
    
    def train_step(self, batch, epoch, it, n_batches):
        """训练一步"""
        # 决定训练模式
        if self.discriminators is None:
            train_mode = "gen"
        else:
            train_mode = "gen" if (it % (self.t + self.t_disc) < self.t) else "disc"
        
        if train_mode == "disc":
            # 训练判别器
            opt = self.discriminator_optimizer
            loss_key = "disc"
            ignore_in_total = ("ddc_1", "ddc_2_flipped", "ddc_3", "att", "gen")
        else:
            # 训练生成器（backbones + clustering）
            opt = self.clustering_optimizer
            loss_key = "tot"
            ignore_in_total = ("disc",)
        
        opt.zero_grad()
        _ = self(batch)
        losses = self.calc_losses(ignore_in_total=ignore_in_total)
        losses[loss_key].backward()
        
        # 梯度裁剪
        if train_mode == "gen" and self.clip_norm is not None:
            th.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        
        opt.step()
        return losses


# ==================== 配置函数 ====================

# MLP默认配置
MLP_DEFAULT_LAYERS = (512, 512, 256)

# DDC默认配置
DDC_DEFAULT_N_HIDDEN = 100
DDC_DEFAULT_USE_BN = True


def get_eamc_config(dataset_name):
    """
    获取EAMC模型的配置（从src/config/eamc/experiments.py中提取）
    
    Args:
        dataset_name: 数据集名称 ('blobs_overlap', 'blobs_overlap_5', 'mnist_mv', 'fmnist', 'coil', 'rgbd')
    
    Returns:
        dict: 包含所有配置参数的字典
    """
    # 从src中提取的配置
    configs = {
        'blobs_overlap': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'discriminator_layers': [32, 32, 32],  # discriminator MLP layers
            'n_clusters': 3,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'use_attention': True,
            'use_discriminator': True,
            'attention_tau': 10.0,
            'gamma': 10.0,
            'lr_backbones': 2e-4,
            'lr_clustering': 1e-5,  # 使用默认值
            'lr_att': 1e-4,
            'lr_disc': 1e-5,
            't': 1,
            't_disc': 1,
            'clip_norm': 0.5,
        },
        'blobs_overlap_5': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'discriminator_layers': [32, 32, 32],
            'n_clusters': 5,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'use_attention': True,
            'use_discriminator': True,
            'attention_tau': 10.0,
            'gamma': 10.0,
            'lr_backbones': 2e-4,
            'lr_clustering': 1e-5,
            'lr_att': 1e-4,
            'lr_disc': 1e-5,
            't': 1,
            't_disc': 1,
            'clip_norm': 0.5,
        },
        'mnist_mv': {
            'backbone_configs': [
                {'type': 'cnn', 'input_channels': 1, 
                 'conv_layers': [(32, 5, 1, 2), (64, 5, 1, 2)],  # 简化：实际需要根据CNN_LAYERS解析
                 'fc_layers': [500], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1,
                 'conv_layers': [(32, 5, 1, 2), (64, 5, 1, 2)],
                 'fc_layers': [500], 'activation': 'relu'},
            ],
            'discriminator_layers': [256, 256, 128],  # 默认discriminator
            'n_clusters': 10,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'use_attention': True,
            'use_discriminator': True,
            'attention_tau': 10.0,
            'gamma': 10.0,
            'lr_backbones': 1e-5,  # 默认值
            'lr_clustering': 1e-5,
            'lr_att': 1e-4,
            'lr_disc': 1e-3,
            't': 1,
            't_disc': 1,
            'clip_norm': 0.5,
        },
        'fmnist': {
            'backbone_configs': [
                {'type': 'cnn', 'input_channels': 1,
                 'conv_layers': [(32, 5, 1, 2), (64, 5, 1, 2)],
                 'fc_layers': [500], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1,
                 'conv_layers': [(32, 5, 1, 2), (64, 5, 1, 2)],
                 'fc_layers': [500], 'activation': 'relu'},
            ],
            'discriminator_layers': [256, 256, 128],
            'n_clusters': 10,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'use_attention': True,
            'use_discriminator': True,
            'attention_tau': 10.0,
            'gamma': 10.0,
            'lr_backbones': 1e-5,
            'lr_clustering': 1e-5,
            'lr_att': 1e-4,
            'lr_disc': 1e-3,
            't': 1,
            't_disc': 1,
            'clip_norm': 0.5,
        },
        'coil': {
            'backbone_configs': [
                {'type': 'cnn', 'input_channels': 1,
                 'conv_layers': [(32, 5, 1, 2), (64, 5, 1, 2)],
                 'fc_layers': [500], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1,
                 'conv_layers': [(32, 5, 1, 2), (64, 5, 1, 2)],
                 'fc_layers': [500], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1,
                 'conv_layers': [(32, 5, 1, 2), (64, 5, 1, 2)],
                 'fc_layers': [500], 'activation': 'relu'},
            ],
            'discriminator_layers': [256, 256, 128],
            'n_clusters': 20,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'use_attention': True,
            'use_discriminator': True,
            'attention_tau': 10.0,
            'gamma': 10.0,
            'lr_backbones': 1e-5,
            'lr_clustering': 1e-5,
            'lr_att': 1e-4,
            'lr_disc': 1e-3,
            't': 1,
            't_disc': 1,
            'clip_norm': 0.5,
        },
        'rgbd': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [2048, 200, 200, 500], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [300, 200, 200, 500], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'discriminator_layers': [256, 256, 128],
            'n_clusters': 13,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'use_attention': True,
            'use_discriminator': True,
            'attention_tau': 10.0,
            'gamma': 10.0,
            'lr_backbones': 6e-5,
            'lr_clustering': 1e-5,
            'lr_att': 1e-4,
            'lr_disc': 2e-5,
            't': 1,
            't_disc': 1,
            'clip_norm': 0.5,
        },
    }
    
    if dataset_name not in configs:
        raise ValueError(f"未知的数据集: {dataset_name}. 支持的数据集: {list(configs.keys())}")
    
    return configs[dataset_name].copy()


# ==================== 训练函数 ====================

def train_eamc(dataset_name='voc', n_clusters=10, n_epochs=100, batch_size=128,
                lr_backbones=1e-5, lr_clustering=1e-5, lr_att=1e-4, lr_disc=1e-3,
                use_attention=True, use_discriminator=True, device='cuda', n_samples=None,
                use_config=True):
    """
    训练EAMC模型
    
    Args:
        dataset_name: 数据集名称
        n_clusters: 聚类数量
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr_backbones: backbone学习率
        lr_clustering: 聚类模块学习率
        lr_att: 注意力模块学习率
        lr_disc: 判别器学习率
        use_attention: 是否使用注意力机制
        use_discriminator: 是否使用判别器
        device: 设备
        n_samples: 采样数量（None表示使用全部数据）
        use_config: 如果True，使用配置文件中的配置；如果False，使用自定义参数
    """
    print("=" * 50)
    print("训练 EAMC 模型")
    print("=" * 50)
    
    # 加载数据
    dataset = load_dataset(dataset_name, n_samples=n_samples, device=device)
    loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
    
    # 获取数据信息
    *views, labels = dataset.tensors
    n_views = len(views)
    input_dims = [v.shape[1] if len(v.shape) == 2 else v.shape[1:] for v in views]
    
    print(f"数据集: {dataset_name}")
    print(f"视图数量: {n_views}")
    print(f"输入维度: {input_dims}")
    print(f"样本数量: {len(labels)}")
    
    # 获取配置
    if use_config:
        try:
            config = get_eamc_config(dataset_name)
            backbone_configs = config['backbone_configs']
            discriminator_layers = config.get('discriminator_layers', [256, 256, 128])
            n_clusters = config['n_clusters'] if n_clusters is None else n_clusters
            n_hidden = config['n_hidden']
            use_bn = config['use_bn']
            use_attention = config.get('use_attention', use_attention)
            use_discriminator = config.get('use_discriminator', use_discriminator)
            attention_tau = config.get('attention_tau', 10.0)
            gamma = config.get('gamma', 10.0)
            lr_backbones = config.get('lr_backbones', lr_backbones)
            lr_clustering = config.get('lr_clustering', lr_clustering)
            lr_att = config.get('lr_att', lr_att)
            lr_disc = config.get('lr_disc', lr_disc)
            t = config.get('t', 1)
            t_disc = config.get('t_disc', 1)
            clip_norm = config.get('clip_norm', 0.5)
            print(f"使用配置: n_clusters={n_clusters}, n_hidden={n_hidden}, lr_backbones={lr_backbones}, lr_disc={lr_disc}")
        except (ValueError, KeyError) as e:
            print(f"警告: 无法加载配置 {e}，使用自定义参数")
            use_config = False
    
    if not use_config:
        # 使用自定义配置
        backbone_configs = [
            {'type': 'mlp', 'layers': [dim if isinstance(dim, int) else np.prod(dim), 128, 64], 
             'activation': 'relu', 'use_bn': False, 'use_bias': True}
            for dim in input_dims
        ]
        discriminator_layers = [256, 256, 128]
        n_hidden = 128
        use_bn = True
        attention_tau = 10.0  # 与原版一致
        gamma = 10.0  # 与原版一致
        t = 1
        t_disc = 1
        clip_norm = 0.5
    
    print(f"类别数量: {n_clusters}")
    
    # 创建模型
    model = EAMC(
        backbone_configs=backbone_configs,
        n_hidden=n_hidden,
        n_clusters=n_clusters,
        use_attention=use_attention,
        use_discriminator=use_discriminator,
        attention_tau=attention_tau,
        gamma=gamma,
        use_bn=use_bn,
        lr_backbones=lr_backbones,
        lr_clustering=lr_clustering,
        lr_att=lr_att,
        lr_disc=lr_disc,
        t=t,
        t_disc=t_disc,
        clip_norm=clip_norm,
        device=device,
        discriminator_layers=discriminator_layers if use_config else None
    ).to(device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print("\n开始训练...\n")
    
    # 训练
    best_acc = 0.0
    n_batches = len(loader)
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []
        
        for batch_idx, (*batch, _) in enumerate(loader):
            losses = model.train_step(batch, epoch=epoch-1, it=batch_idx, n_batches=n_batches)
            epoch_losses.append(to_numpy(losses))
        
        # 评估
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with th.no_grad():
                all_preds = []
                all_labels = []
                for *batch, labels_batch in loader:
                    output = model(batch)
                    preds = output.argmax(dim=1)
                    all_preds.append(to_numpy(preds))
                    all_labels.append(to_numpy(labels_batch))
                
                all_preds = np.concatenate(all_preds)
                all_labels = np.concatenate(all_labels)
                metrics = calc_metrics(all_labels, all_preds)
                
                print(f"Epoch {epoch:3d} | Loss: {np.mean([l['tot'] for l in epoch_losses]):.4f} | "
                      f"ACC: {metrics['acc']:.4f} | NMI: {metrics['nmi']:.4f}")
                
                if metrics['acc'] > best_acc:
                    best_acc = metrics['acc']
    
    print(f"\n训练完成！最佳准确率: {best_acc:.4f}")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='训练EAMC模型')
    parser.add_argument('--dataset', type=str, default='voc', help='数据集名称')
    parser.add_argument('--n_clusters', type=int, default=None, help='聚类数量')
    parser.add_argument('--n_epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=100, help='批次大小')
    parser.add_argument('--lr_backbones', type=float, default=1e-5, help='backbone学习率')
    parser.add_argument('--lr_clustering', type=float, default=1e-5, help='聚类模块学习率')
    parser.add_argument('--lr_att', type=float, default=1e-4, help='注意力模块学习率')
    parser.add_argument('--lr_disc', type=float, default=1e-3, help='判别器学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--n_samples', type=int, default=None, help='采样数量')
    parser.add_argument('--no_attention', action='store_true', help='不使用注意力机制')
    parser.add_argument('--no_discriminator', action='store_true', help='不使用判别器')
    parser.add_argument('--no_config', action='store_true', help='不使用配置文件')
    
    args = parser.parse_args()
    
    train_eamc(
        dataset_name=args.dataset,
        n_clusters=args.n_clusters,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr_backbones=args.lr_backbones,
        lr_clustering=args.lr_clustering,
        lr_att=args.lr_att,
        lr_disc=args.lr_disc,
        use_attention=not args.no_attention,
        use_discriminator=not args.no_discriminator,
        device=args.device,
        n_samples=args.n_samples,
        use_config=not args.no_config
    )
