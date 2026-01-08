"""
CoMVC (Contrastive Multi-View Clustering) 模型
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


class MeanFusion(nn.Module):
    """平均融合"""
    def forward(self, inputs):
        return th.mean(th.stack(inputs, -1), dim=-1)


class WeightedMeanFusion(nn.Module):
    """加权平均融合"""
    def __init__(self, n_views):
        super().__init__()
        self.weights = nn.Parameter(th.ones(n_views) / n_views)
    
    def forward(self, inputs):
        weights = F.softmax(self.weights, dim=0)
        return th.sum(weights[None, None, :] * th.stack(inputs, dim=-1), dim=-1)


# ==================== 模型定义 ====================

class CoMVC(nn.Module):
    """对比多视图聚类模型"""
    def __init__(self, backbone_configs, fusion_type='mean', n_hidden=128, n_clusters=10,
                 projector_layers=None, use_bn=True, lr=1e-3, tau=0.1, delta=1.0, 
                 negative_samples_ratio=-1, adaptive_contrastive_weight=False, device='cuda'):
        """
        projector_layers: list of int, 投影层的维度，如果为None则不使用投影层
        tau: 温度参数
        delta: 对比损失权重
        negative_samples_ratio: 负采样比例，-1表示不使用负采样（仅支持2视图），>0表示使用负采样（支持多视图）
        adaptive_contrastive_weight: 是否使用自适应对比损失权重
        """
        super().__init__()
        self.device = device
        self.n_views = len(backbone_configs)
        self.tau = tau
        self.delta = delta
        self.negative_samples_ratio = negative_samples_ratio
        self.adaptive_contrastive_weight = adaptive_contrastive_weight
        self.large_num = 1e9
        
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
        
        # 融合模块
        if fusion_type == 'mean':
            self.fusion = MeanFusion()
        else:
            self.fusion = WeightedMeanFusion(self.n_views)
        
        # 投影层
        if projector_layers is None:
            self.projector = nn.Identity()
            proj_dim = hidden_dim
        else:
            self.projector = MLP([hidden_dim] + projector_layers, 'relu', use_bn, True)
            proj_dim = projector_layers[-1]
        
        # 聚类模块
        self.ddc = DDC(hidden_dim, n_hidden, n_clusters, use_bn)
        
        # 优化器
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr)
        
        # 初始化权重
        self.apply(he_init_weights)
        
        self.backbone_outputs = None
        self.fused = None
        self.hidden = None
        self.output = None
        self.projections = None
    
    def forward(self, views):
        self.backbone_outputs = [bb(v) for bb, v in zip(self.backbones, views)]
        self.fused = self.fusion(self.backbone_outputs)
        self.projections = self.projector(th.cat(self.backbone_outputs, dim=0))
        self.output, self.hidden = self.ddc(self.fused)
        return self.output
    
    def calc_losses(self, rel_sigma=0.15):
        """计算损失：DDC损失 + 对比损失"""
        # DDC损失
        hidden_kernel = vector_kernel(self.hidden, rel_sigma)
        ddc1 = d_cs(self.output, hidden_kernel, self.ddc.n_clusters)
        n = self.output.size(0)
        ddc2 = 2 / (n * (n - 1)) * triu(self.output @ self.output.t())
        eye = th.eye(self.ddc.n_clusters, device=self.device)
        m = th.exp(-cdist(self.output, eye))
        ddc3 = d_cs(m, hidden_kernel, self.ddc.n_clusters)
        
        # 对比损失（支持多视图）
        if self.negative_samples_ratio == -1:
            # 无负采样版本，仅支持2个视图
            if self.n_views == 2:
                contrast_loss = self._contrastive_loss_without_negative_sampling()
            else:
                contrast_loss = th.tensor(0.0, device=self.device)
        else:
            # 有负采样版本，支持多视图
            contrast_loss = self._contrastive_loss_with_negative_sampling()
        
        return {
            'ddc1': ddc1,
            'ddc2': ddc2,
            'ddc3': ddc3,
            'contrast': contrast_loss,
            'tot': ddc1 + ddc2 + ddc3 + self.delta * contrast_loss
        }
    
    def _normalized_projections(self):
        """获取归一化的投影向量（仅用于2视图无负采样版本）"""
        n = self.projections.size(0) // 2
        h1, h2 = self.projections[:n], self.projections[n:]
        h1 = F.normalize(h1, p=2, dim=1)
        h2 = F.normalize(h2, p=2, dim=1)
        return n, h1, h2
    
    @staticmethod
    def _get_positive_samples(logits, v, n):
        """
        获取正样本对
        
        :param logits: 相似度矩阵
        :param v: 视图数量
        :param n: 每个视图的样本数（batch size）
        :return: 正样本对的相似度和索引
        """
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = th.diagonal(logits, offset=diagonal_offset)
            _lower = th.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = th.arange(0, diag_length, device=logits.device)
            _lower_inds = th.arange(i * n, v * n, device=logits.device)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]
        
        pos = th.cat(diagonals, dim=0)
        pos_inds = th.cat(inds, dim=0)
        return pos, pos_inds
    
    def _draw_negative_samples(self, pos_indices):
        """
        构造负样本集
        
        :param pos_indices: 正样本在拼接相似度矩阵中的行索引
        :return: 负样本索引
        """
        eye = th.eye(self.ddc.n_clusters, device=self.device)
        cat = self.output.detach().argmax(dim=1)
        cat = th.cat(self.n_views * [cat], dim=0)
        
        # 使用双重索引以保持与 arch/mvc 一致
        weights = (1 - eye[cat])[:, cat[pos_indices]].T
        n_negative_samples = int(self.negative_samples_ratio * cat.size(0))
        negative_sample_indices = th.multinomial(weights, n_negative_samples, replacement=True)
        return negative_sample_indices
    
    def _contrastive_loss_without_negative_sampling(self):
        """
        无负采样的对比损失（仅支持2个视图）
        参考: https://github.com/google-research/simclr/blob/master/objective.py
        """
        assert self.n_views == 2, "Contrastive loss without negative sampling only supports 2 views."
        n, h1, h2 = self._normalized_projections()
        
        labels = th.arange(0, n, device=self.device, dtype=th.long)
        masks = th.eye(n, device=self.device)
        
        logits_aa = ((h1 @ h1.t()) / self.tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / self.tau) - masks * self.large_num
        logits_ab = (h1 @ h2.t()) / self.tau
        logits_ba = (h2 @ h1.t()) / self.tau
        
        loss_a = F.cross_entropy(th.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = F.cross_entropy(th.cat((logits_ba, logits_bb), dim=1), labels)
        
        loss = loss_a + loss_b
        
        if self.adaptive_contrastive_weight:
            if hasattr(self.fusion, 'weights'):
                w = th.min(F.softmax(self.fusion.weights.detach(), dim=0))
                loss *= w
        
        return loss
    
    def _contrastive_loss_with_negative_sampling(self):
        """
        有负采样的对比损失（支持多视图）
        """
        n = self.output.size(0)
        v = self.n_views
        
        # 计算相似度矩阵（使用余弦相似度）
        h = F.normalize(self.projections, p=2, dim=1)
        logits = (h @ h.t()) / self.tau
        
        # 获取正样本对
        pos, pos_inds = self._get_positive_samples(logits, v, n)
        
        # 获取负样本
        neg_inds = self._draw_negative_samples(pos_inds)
        neg = logits[pos_inds.view(-1, 1), neg_inds]
        
        # 计算损失
        inputs = th.cat((pos.view(-1, 1), neg), dim=1)
        labels = th.zeros(v * (v - 1) * n, device=self.device, dtype=th.long)
        loss = F.cross_entropy(inputs, labels)
        
        if self.adaptive_contrastive_weight:
            if hasattr(self.fusion, 'weights'):
                w = th.min(F.softmax(self.fusion.weights.detach(), dim=0))
                loss *= w
        
        return loss
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        _ = self(batch)
        losses = self.calc_losses()
        losses['tot'].backward()
        self.optimizer.step()
        return losses


# ==================== 配置函数 ====================

# MLP默认配置
MLP_DEFAULT_LAYERS = (512, 512, 256)

# DDC默认配置
DDC_DEFAULT_N_HIDDEN = 100
DDC_DEFAULT_USE_BN = True


def get_comvc_config(dataset_name):
    """
    获取CoMVC模型的配置
    
    Args:
        dataset_name: 数据集名称 ('voc', 'rgbd', 'mnist_mv', 'fmnist', 'coil', 'ccv', 'blobs_overlap', 'blobs_overlap_5')
    
    Returns:
        dict: 包含所有配置参数的字典
    """
    configs = {
        'voc': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [512] + list(MLP_DEFAULT_LAYERS), 'input_size': 512, 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [399] + list(MLP_DEFAULT_LAYERS), 'input_size': 399, 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 20,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 0.1,
        },
        'rgbd': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [2048, 512, 512, 256], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [300, 512, 512, 256], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 13,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 0.1,
        },
        'mnist_mv': {
            'backbone_configs': [
                {'type': 'cnn', 'input_channels': 1, 'conv_layers': [(32, 5, 1, 2), (32, 5, 1, 2), (32, 3, 1, 1), (32, 3, 1, 1)], 'fc_layers': [], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1, 'conv_layers': [(32, 5, 1, 2), (32, 5, 1, 2), (32, 3, 1, 1), (32, 3, 1, 1)], 'fc_layers': [], 'activation': 'relu'},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 10,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 0.1,
        },
        'fmnist': {
            'backbone_configs': [
                {'type': 'cnn', 'input_channels': 1, 'conv_layers': [(32, 5, 1, 2), (32, 5, 1, 2), (32, 3, 1, 1), (32, 3, 1, 1)], 'fc_layers': [], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1, 'conv_layers': [(32, 5, 1, 2), (32, 5, 1, 2), (32, 3, 1, 1), (32, 3, 1, 1)], 'fc_layers': [], 'activation': 'relu'},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 10,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 0.1,
        },
        'coil': {
            'backbone_configs': [
                {'type': 'cnn', 'input_channels': 1, 'conv_layers': [(32, 5, 1, 2), (32, 5, 1, 2), (32, 3, 1, 1), (32, 3, 1, 1)], 'fc_layers': [], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1, 'conv_layers': [(32, 5, 1, 2), (32, 5, 1, 2), (32, 3, 1, 1), (32, 3, 1, 1)], 'fc_layers': [], 'activation': 'relu'},
                {'type': 'cnn', 'input_channels': 1, 'conv_layers': [(32, 5, 1, 2), (32, 5, 1, 2), (32, 3, 1, 1), (32, 3, 1, 1)], 'fc_layers': [], 'activation': 'relu'},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 20,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 20.0,
        },
        'ccv': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [5000, 512, 512, 256], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [5000, 512, 512, 256], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [4000, 512, 512, 256], 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 20,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 20.0,
        },
        'blobs_overlap': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'input_size': 2, 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'input_size': 2, 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 3,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 0.1,
        },
        'blobs_overlap_5': {
            'backbone_configs': [
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'input_size': 2, 'activation': 'relu', 'use_bn': False, 'use_bias': True},
                {'type': 'mlp', 'layers': [2, 32, 32, 32], 'input_size': 2, 'activation': 'relu', 'use_bn': False, 'use_bias': True},
            ],
            'fusion_type': 'weighted_mean',
            'n_clusters': 5,
            'n_hidden': DDC_DEFAULT_N_HIDDEN,
            'use_bn': DDC_DEFAULT_USE_BN,
            'projector_layers': None,
            'lr': 1e-3,
            'tau': 0.1,
            'delta': 0.1,
        },
    }
    
    if dataset_name not in configs:
        raise ValueError(f"未知的数据集: {dataset_name}. 支持的数据集: {list(configs.keys())}")
    
    return configs[dataset_name].copy()


# ==================== 训练函数 ====================

def train_comvc(dataset_name='voc', n_clusters=None, n_epochs=100, batch_size=100,
                lr=None, tau=None, delta=None, device='cuda', n_samples=None, use_config=True):
    """
    训练CoMVC模型
    
    Args:
        dataset_name: 数据集名称
        n_clusters: 聚类数量（None表示使用配置中的默认值）
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率（None表示使用配置中的默认值）
        tau: 温度参数（None表示使用配置中的默认值）
        delta: 对比损失权重（None表示使用配置中的默认值）
        device: 设备
        n_samples: 采样数量（None表示使用全部数据）
        use_config: 如果True，使用配置文件中的配置；如果False，使用自定义参数
    """
    print("=" * 50)
    print("训练 CoMVC 模型")
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
            config = get_comvc_config(dataset_name)
            backbone_configs = config['backbone_configs']
            fusion_type = config['fusion_type']
            n_clusters = config['n_clusters'] if n_clusters is None else n_clusters
            n_hidden = config['n_hidden']
            use_bn = config['use_bn']
            projector_layers = config.get('projector_layers', None)
            lr = config['lr'] if lr is None else lr
            tau = config.get('tau', 0.1) if tau is None else tau
            delta = config.get('delta', 0.1) if delta is None else delta
            print(f"使用原版配置: n_clusters={n_clusters}, lr={lr}, tau={tau}, delta={delta}")
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
        fusion_type = 'weighted_mean'
        n_clusters = n_clusters if n_clusters is None else 10
        n_hidden = 128
        use_bn = True
        projector_layers = [128, 64]
        lr = lr if lr is None else 1e-3
        tau = tau if tau is None else 0.1
        delta = delta if delta is None else 1.0
    
    print(f"类别数量: {n_clusters}")
    
    # 创建模型
    model = CoMVC(
        backbone_configs=backbone_configs,
        fusion_type=fusion_type,
        n_hidden=n_hidden,
        n_clusters=n_clusters,
        projector_layers=projector_layers,
        use_bn=use_bn,
        lr=lr,
        tau=tau,
        delta=delta,
        device=device
    ).to(device)
    
    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print("\n开始训练...\n")
    
    # 训练
    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []
        
        for batch_idx, (*batch, _) in enumerate(loader):
            losses = model.train_step(batch)
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
    parser = argparse.ArgumentParser(description='训练CoMVC模型')
    parser.add_argument('--dataset', type=str, default='voc', help='数据集名称')
    parser.add_argument('--n_clusters', type=int, default=None, help='聚类数量')
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数（默认100，与原版Experiment一致）')
    parser.add_argument('--batch_size', type=int, default=100, help='批次大小（默认100，与原版Experiment一致）')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--tau', type=float, default=None, help='温度参数')
    parser.add_argument('--delta', type=float, default=None, help='对比损失权重')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--n_samples', type=int, default=None, help='采样数量')
    parser.add_argument('--no_config', action='store_true', help='不使用配置文件')
    
    args = parser.parse_args()
    
    train_comvc(
        dataset_name=args.dataset,
        n_clusters=args.n_clusters,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        delta=args.delta,
        device=args.device,
        n_samples=args.n_samples,
        use_config=not args.no_config
    )
