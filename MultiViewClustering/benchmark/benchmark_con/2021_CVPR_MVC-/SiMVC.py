"""
SiMVC (Simple Multi-View Clustering) 模型
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

class SiMVC(nn.Module):
    """简单多视图聚类模型"""
    def __init__(self, backbone_configs, fusion_type='mean', n_hidden=128, n_clusters=10, 
                 use_bn=True, lr=1e-3, device='cuda'):
        """
        backbone_configs: list of dict, 每个视图的backbone配置
            - type: 'mlp' or 'cnn'
            - 对于MLP: {'type': 'mlp', 'layers': [input_dim, hidden1, hidden2, ...], 'activation': 'relu'}
            - 对于CNN: {'type': 'cnn', 'input_channels': 1, 'conv_layers': [...], 'fc_layers': [...]}
        fusion_type: 'mean' or 'weighted_mean'
        """
        super().__init__()
        self.device = device
        self.n_views = len(backbone_configs)
        self.backbones = nn.ModuleList()
        for cfg in backbone_configs:
            bb = MLP(cfg['layers'], cfg.get('activation', 'relu'), 
                    cfg.get('use_bn', False), cfg.get('use_bias', True))
            self.backbones.append(bb)
        
        # 获取backbone输出维度（假设所有backbone输出维度相同）
        with th.no_grad():
            dummy_input = th.randn(1, backbone_configs[0]['layers'][0] if backbone_configs[0]['type'] == 'mlp' else (1, 28, 28))
            dummy_output = self.backbones[0](dummy_input)  # 在CPU上计算
            hidden_dim = dummy_output.shape[1]
        
        # 融合模块
        if fusion_type == 'mean':
            self.fusion = MeanFusion()
        else:
            self.fusion = WeightedMeanFusion(self.n_views)
        
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
    
    def forward(self, views):
        self.backbone_outputs = [bb(v) for bb, v in zip(self.backbones, views)]
        self.fused = self.fusion(self.backbone_outputs)
        self.output, self.hidden = self.ddc(self.fused)
        return self.output
    
    def calc_losses(self, rel_sigma=0.15):
        """计算DDC损失"""
        # DDC1: Cauchy-Schwarz散度
        hidden_kernel = vector_kernel(self.hidden, rel_sigma)
        ddc1 = d_cs(self.output, hidden_kernel, self.ddc.n_clusters)
        
        # DDC2: 正交性约束
        n = self.output.size(0)
        ddc2 = 2 / (n * (n - 1)) * triu(self.output @ self.output.t())
        
        # DDC3: 聚类中心约束
        eye = th.eye(self.ddc.n_clusters, device=self.device)
        m = th.exp(-cdist(self.output, eye))
        ddc3 = d_cs(m, hidden_kernel, self.ddc.n_clusters)
        
        return {
            'ddc1': ddc1,
            'ddc2': ddc2,
            'ddc3': ddc3,
            'tot': ddc1 + ddc2 + ddc3
        }
    
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


def get_simvc_config(dataset_name):
    """
    获取SiMVC模型的配置
    
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
            'lr': 1e-3,
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
            'lr': 1e-3,
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
            'lr': 1e-3,
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
            'lr': 1e-3,
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
            'lr': 1e-3,
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
            'lr': 1e-3,
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
            'lr': 1e-3,
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
            'lr': 1e-3,
        },
    }
    
    if dataset_name not in configs:
        raise ValueError(f"未知的数据集: {dataset_name}. 支持的数据集: {list(configs.keys())}")
    
    return configs[dataset_name].copy()


# ==================== 训练函数 ====================

def train_simvc(dataset_name='voc', n_clusters=None, n_epochs=100, batch_size=100, 
                lr=None, device='cuda', n_samples=None, use_config=True):
    """
    训练SiMVC模型
    
    Args:
        dataset_name: 数据集名称
        n_clusters: 聚类数量（None表示使用配置中的默认值）
        n_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率（None表示使用配置中的默认值）
        device: 设备
        n_samples: 采样数量（None表示使用全部数据）
        use_config: 如果True，使用配置文件中的配置；如果False，使用自定义参数
    """
    print("=" * 50)
    print("训练 SiMVC 模型")
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
            config = get_simvc_config(dataset_name)
            backbone_configs = config['backbone_configs']
            fusion_type = config['fusion_type']
            n_clusters = config['n_clusters'] if n_clusters is None else n_clusters
            n_hidden = config['n_hidden']
            use_bn = config['use_bn']
            lr = config['lr'] if lr is None else lr
            print(f"使用原版配置: n_clusters={n_clusters}, lr={lr}, n_hidden={n_hidden}")
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
        lr = lr if lr is None else 1e-3
    
    print(f"类别数量: {n_clusters}")
    
    # 创建模型
    model = SiMVC(
        backbone_configs=backbone_configs,
        fusion_type=fusion_type,
        n_hidden=n_hidden,
        n_clusters=n_clusters,
        use_bn=use_bn,
        lr=lr,
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
    
    parser = argparse.ArgumentParser(description='训练SiMVC模型')
    parser.add_argument('--dataset', type=str, default='voc', help='数据集名称')
    parser.add_argument('--n_clusters', type=int, default=None, help='聚类数量')
    parser.add_argument('--n_epochs', type=int, default=100, help='训练轮数（默认100，与原版Experiment一致）')
    parser.add_argument('--batch_size', type=int, default=100, help='批次大小（默认100，与原版Experiment一致）')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--n_samples', type=int, default=None, help='采样数量')
    parser.add_argument('--no_config', action='store_true', help='不使用配置文件')
    
    args = parser.parse_args()
    
    train_simvc(
        dataset_name=args.dataset,
        n_clusters=args.n_clusters,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        n_samples=args.n_samples,
        use_config=not args.no_config
    )
