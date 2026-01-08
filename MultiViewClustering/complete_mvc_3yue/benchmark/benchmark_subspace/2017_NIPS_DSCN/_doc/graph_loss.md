# Graph Regularization Loss

## 代码

```python
def graph_loss(Z, S):
    Z = torch.reshape(Z, (Z.shape[0], -1))  # (N, d)
    S = 0.5 * (S.T + S)                     # (N, N) 对称化
    D = torch.diag(torch.sum(S, dim=1))      # (N, N) 度矩阵
    L = D - S                                # (N, N) 图拉普拉斯矩阵
    return 2 * torch.trace(Z.T @ L @ Z)      # scalar
```

## 符号定义

| 符号 | 含义 | 维度 |
|------|------|------|
| $Z$ | 编码器输出的特征矩阵，第 $i$ 行 $z_i$ 是样本 $i$ 的特征 | $N \times d$ |
| $S$ | 自表示系数矩阵 | $N \times N$ |
| $D$ | 度矩阵，$D_{ii} = \sum_j S_{ij}$ | $N \times N$ |
| $L$ | 图拉普拉斯矩阵，$L = D - S$ | $N \times N$ |

## 损失函数

$$\mathcal{L}_{\text{graph}} = 2 \cdot \text{tr}(Z^\top L Z)$$

## 推导：$\text{tr}(Z^\top L Z) = \frac{1}{2}\sum_{i,j} S_{ij} \|z_i - z_j\|^2$

### 第一步：分解为单维情况

设 $z_k \in \mathbb{R}^N$ 是 $Z$ 的第 $k$ 列（所有样本在第 $k$ 个特征维度上的值），则：

$$\text{tr}(Z^\top L Z) = \sum_{k=1}^{d} (Z^\top L Z)_{kk} = \sum_{k=1}^{d} z_k^\top L \, z_k$$

只需证明单维情况 $z^\top L z = \frac{1}{2}\sum_{i,j} S_{ij}(z_i - z_j)^2$，然后对 $d$ 个维度求和即得。

### 第二步：展开 $z^\top L z$

$$z^\top L z = z^\top (D - S) z = z^\top D z - z^\top S z$$

分别展开两项：

$$z^\top D z = \sum_i D_{ii} z_i^2 = \sum_i \left(\sum_j S_{ij}\right) z_i^2 = \sum_{i,j} S_{ij} z_i^2$$

$$z^\top S z = \sum_{i,j} S_{ij} z_i z_j$$

因此：

$$z^\top L z = \sum_{i,j} S_{ij} z_i^2 - \sum_{i,j} S_{ij} z_i z_j$$

### 第三步：利用对称性凑完全平方

由于 $S$ 已被对称化（$S_{ij} = S_{ji}$），可以交换 $i, j$ 下标：

$$\sum_{i,j} S_{ij} z_i^2 = \sum_{i,j} S_{ji} z_j^2 = \sum_{i,j} S_{ij} z_j^2$$

所以：

$$\sum_{i,j} S_{ij} z_i^2 = \frac{1}{2}\sum_{i,j} S_{ij} z_i^2 + \frac{1}{2}\sum_{i,j} S_{ij} z_j^2$$

代入：

$$z^\top L z = \frac{1}{2}\sum_{i,j} S_{ij} z_i^2 + \frac{1}{2}\sum_{i,j} S_{ij} z_j^2 - \sum_{i,j} S_{ij} z_i z_j$$

$$= \frac{1}{2}\sum_{i,j} S_{ij} \left(z_i^2 - 2z_i z_j + z_j^2\right)$$

$$= \frac{1}{2}\sum_{i,j} S_{ij} (z_i - z_j)^2$$

### 第四步：推广到多维

$$\text{tr}(Z^\top L Z) = \sum_{k=1}^{d} \frac{1}{2}\sum_{i,j} S_{ij}(z_{ik} - z_{jk})^2 = \frac{1}{2}\sum_{i,j} S_{ij} \|z_i - z_j\|^2$$

因此最终损失为：

$$\boxed{\mathcal{L}_{\text{graph}} = 2 \cdot \text{tr}(Z^\top L Z) = \sum_{i,j} S_{ij} \|z_i - z_j\|^2}$$

## 直觉

这是一个**以自表示系数为权重的、样本对之间特征距离的加权和**：

- $S_{ij}$ 大（样本 $i$ 和 $j$ 在子空间中相似）→ $\|z_i - z_j\|^2$ 被赋予大权重 → 优化器将 $z_i$ 和 $z_j$ 拉近
- $S_{ij} \approx 0$（不相关）→ 该样本对不贡献损失 → 远近无所谓

最小化此损失的效果：**让编码器学出的特征表示保持自表示系数矩阵所描述的邻域结构**。

## L 矩阵的性质

| 性质 | 说明 |
|------|------|
| 半正定 | 所有特征值 $\geq 0$，因此 $z^\top L z \geq 0$ |
| 行和为零 | $L \mathbf{1} = 0$，即常数向量是特征值 0 对应的特征向量 |
| 对角元素 | $L_{ii} = D_{ii} - S_{ii} = \sum_{j} S_{ij} - S_{ii}$ |
| 非对角元素 | $L_{ij} = -S_{ij}$，越负代表 $i, j$ 越相似 |
| 零特征值个数 | 等于图的连通分量个数 |
