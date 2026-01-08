"""
工具函数模块 - 包含数据加载、指标计算和其他共享工具函数
"""
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


# ==================== 数据加载函数 ====================

def load_dataset(name, data_dir='data/processed', n_samples=None, select_views=None, 
                 select_labels=None, to_dataset=True, device='cuda'):
    """
    加载数据集
    
    Args:
        name: 数据集名称（如 'voc', 'rgbd', 'blobs_overlap'）
        data_dir: 数据目录
        n_samples: 采样数量（None表示使用全部）
        select_views: 选择的视图索引（None表示使用全部）
        select_labels: 选择的标签（None表示使用全部）
        to_dataset: 是否返回Dataset对象
        device: 设备
    
    Returns:
        Dataset对象或(views, labels)元组
    """
    import os
    from pathlib import Path
    
    data_path = Path(data_dir) / f"{name}.npz"
    if not data_path.exists():
        # 尝试从项目根目录查找
        data_path = Path(__file__).parent.parent / "arch" / "mvc" / "data" / "processed" / f"{name}.npz"
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {data_path}")
    
    npz = np.load(data_path)
    labels = npz["labels"]
    views = [npz[f"view_{i}"] for i in range(npz["n_views"])]
    
    # 选择标签
    if select_labels is not None:
        mask = np.isin(labels, select_labels)
        labels = labels[mask]
        views = [v[mask] for v in views]
        # 重新映射标签为0, 1, 2, ...
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels])
    
    # 采样
    if n_samples is not None:
        n = min(labels.shape[0], int(n_samples))
        idx = np.random.choice(labels.shape[0], size=n, replace=False)
        labels = labels[idx]
        views = [v[idx] for v in views]
    
    # 选择视图
    if select_views is not None:
        if not isinstance(select_views, (list, tuple)):
            select_views = [select_views]
        views = [views[i] for i in select_views]
    
    # 转换为float32
    views = [v.astype(np.float32) for v in views]
    
    if to_dataset:
        dataset = data.TensorDataset(
            *[th.tensor(v).to(device) for v in views],
            th.tensor(labels).to(device)
        )
        return dataset
    else:
        return views, labels


def create_dataloader(dataset, batch_size=128, shuffle=True):
    """创建DataLoader"""
    return data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0,
        drop_last=True,
        pin_memory=False
    )


# ==================== 指标计算函数 ====================

def ordered_cmat(labels, pred):
    """
    计算混淆矩阵和准确率（使用最佳聚类-类别分配）
    
    Args:
        labels: 真实标签
        pred: 预测标签
    
    Returns:
        (accuracy, confusion_matrix)
    """
    cmat = confusion_matrix(labels, pred)
    ri, ci = linear_sum_assignment(-cmat)
    ordered = cmat[np.ix_(ri, ci)]
    acc = np.sum(np.diag(ordered)) / np.sum(ordered)
    return acc, ordered


def calc_metrics(labels, pred):
    """
    计算评估指标
    
    Args:
        labels: 真实标签
        pred: 预测标签
    
    Returns:
        dict包含 acc, nmi, cmat
    """
    acc, cmat = ordered_cmat(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred, average_method="geometric")
    return {
        "acc": acc,
        "nmi": nmi,
        "cmat": cmat
    }


def to_numpy(tensor):
    """将tensor转换为numpy数组"""
    if isinstance(tensor, (list, tuple)):
        return [to_numpy(t) for t in tensor]
    elif isinstance(tensor, dict):
        return {k: to_numpy(v) for k, v in tensor.items()}
    else:
        return tensor.cpu().detach().numpy() if isinstance(tensor, th.Tensor) else tensor


# ==================== 共享工具函数 ====================

def he_init_weights(module):
    """He初始化权重"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def cdist(X, Y):
    """计算X和Y之间的成对距离矩阵"""
    xyT = X @ Y.t()
    x2 = th.sum(X**2, dim=1, keepdim=True)
    y2 = th.sum(Y**2, dim=1, keepdim=True)
    d = x2 - 2 * xyT + y2.t()
    return d


def vector_kernel(x, rel_sigma=0.15):
    """计算向量之间的核矩阵（与原版一致）"""
    # 计算距离矩阵
    dist = cdist(x, x)
    dist = F.relu(dist)  # 确保非负（处理浮点误差）
    
    # 计算sigma
    sigma2 = rel_sigma * th.median(dist)
    sigma2 = sigma2.detach()  # 不计算梯度
    sigma2 = th.where(sigma2 < 1e-9, th.tensor(1e-9, device=sigma2.device, dtype=sigma2.dtype), sigma2)
    
    # 高斯核
    k = th.exp(-dist / (2 * sigma2))
    return k


def d_cs(A, K, n_clusters):
    """Cauchy-Schwarz散度"""
    nom = A.t() @ K @ A
    dnom_squared = th.unsqueeze(th.diagonal(nom), -1) @ th.unsqueeze(th.diagonal(nom), 0)
    
    nom = th.clamp(nom, min=1e-9)
    dnom_squared = th.clamp(dnom_squared, min=1e-18)
    
    triu_sum = th.sum(th.triu(nom / th.sqrt(dnom_squared), diagonal=1))
    d = 2 / (n_clusters * (n_clusters - 1)) * triu_sum
    return d


def triu(X):
    """上三角部分的和"""
    return th.sum(th.triu(X, diagonal=1))
