# DeepCluster v1 与 DeepCluster v2 总结

## 1. 总体定位

DeepCluster 系列方法本质上都不是传统意义上“只输出一次聚类结果”的普通聚类算法，而是：

- **基于聚类的无监督表示学习方法**
- 或者说 **clustering-based self-supervised / unsupervised representation learning**

它们的共同思想是：

1. 先用当前网络提取样本特征
2. 对特征做聚类
3. 利用聚类结果构造监督信号
4. 再反过来训练网络
5. 交替迭代，使表示学习与聚类结果相互促进

---

## 2. DeepCluster v1 的核心思想

### 核心一句话
**先聚类生成伪标签，再把聚类标签当作分类监督信号训练网络。**

### 基本流程

给定无标签数据集 \(\{x_i\}_{i=1}^N\)，设特征提取网络为 \(f_\theta(\cdot)\)。

#### Step 1：提取特征
\[
\mathbf{h}_i = f_\theta(x_i)
\]

#### Step 2：对特征聚类
通常对 \(\mathbf{h}_i\) 做 \(k\)-means，得到簇标签：
\[
z_i \in \{1,\dots,K\}
\]

#### Step 3：把簇标签当伪标签训练网络
将 \(z_i\) 当成监督信号，训练网络预测 cluster id：
\[
\mathcal{L} = - \sum_{i=1}^N \log p(z_i \mid x_i)
\]

#### Step 4：重新提特征并重复
网络更新后重新提特征，再聚类，再训练。

---

## 3. DeepCluster v1 的本质

DeepCluster v1 更像：

- **pseudo-label learning**
- **self-training**
- **clustering-based pseudo-supervised learning**

因为它的监督信号本质上是：

- 聚类算法产生的 **离散簇编号**
- 再把这些编号当作“伪类别标签”来训练分类器

所以可以把 v1 理解成：

> **聚类驱动的伪标签学习**

---

## 4. DeepCluster v1 的主要问题

### 4.1 伪标签不稳定
每次重新聚类后，cluster id 可能发生重排。例如：

- 上一轮的 cluster 1
- 下一轮未必还是 cluster 1

也就是说，**簇编号在不同轮次之间没有天然一致性**。

### 4.2 分类头不稳定
由于每轮 cluster id 都可能变化，上一轮学到的分类头往往不能直接延续到下一轮，训练过程容易不稳定。

### 4.3 容易受伪标签噪声影响
聚类得到的标签并不是真实标签，如果早期特征质量较差，则伪标签噪声较大，容易影响训练。

---

## 5. DeepCluster v2 的核心思想

### 核心一句话
**保留“聚类生成监督信号”的框架，但不再主要学习预测离散簇编号，而是让样本特征直接与聚类中心（prototypes / centroids）进行匹配。**

---

## 6. DeepCluster v2 的主要改进

### 6.1 从“预测簇编号”改为“匹配聚类中心”
v1 的方式是：

- 生成 cluster id
- 用分类头预测这个 id

v2 的方式是：

- 聚类得到 centroids
- 让样本特征直接与这些 centroids 做比较和匹配

因此，v2 不再那么像普通的伪标签分类，而更像：

- **prototype learning**
- **centroid-based representation learning**
- **clustering-based prototype learning**

### 6.2 监督信号从离散标签变成几何匹配
v1 更关注：

- “这个样本属于第几类？”

v2 更关注：

- “这个样本的特征更接近哪个中心？”

所以：

- **v1 是 label-style supervision**
- **v2 是 prototype-style supervision**

### 6.3 训练更稳定
因为 centroid 是特征空间里的连续几何对象，而 cluster id 只是离散编号，相比“直接预测编号”，**与中心进行匹配通常更稳定**。

---

## 7. DeepCluster v2 的本质

DeepCluster v2 更适合被理解为：

- **prototype learning**
- **prototype-based self-supervised learning**
- **clustering-based prototype matching**

它仍然依赖聚类 assignment，但训练目标不再主要表现为“分类伪标签”，而是“样本与原型中心之间的对齐”。

所以可以概括为：

> **v1 是伪标签驱动的分类式训练；v2 是聚类原型驱动的匹配式训练。**

---

## 8. v1 与 v2 的核心区别

| 方面 | DeepCluster v1 | DeepCluster v2 |
|---|---|---|
| 核心监督信号 | 聚类得到的离散簇标签 | 聚类得到的中心 / prototypes |
| 训练形式 | 预测 cluster id 的分类训练 | 特征与 centroid 的显式匹配 |
| 方法风格 | pseudo-label learning | prototype learning |
| 主要问题 | 簇编号跨轮不稳定 | 相对更稳定 |
| 本质 | 聚类驱动的伪标签学习 | 聚类驱动的原型匹配学习 |

---

## 9. 如何理解两者关系

可以把它们理解成同一条方法路线上的两个阶段：

### DeepCluster v1
先把聚类结果离散化成“伪类别标签”，再用普通分类损失去训练网络。

### DeepCluster v2
不再过分依赖离散簇编号，而是更直接地利用聚类中心来约束特征空间。

因此，v2 可以看作对 v1 的一次“从伪标签分类到原型匹配”的升级。

---

## 10. 从机器学习类型上看

### DeepCluster v1 更像
- self-training
- pseudo-label learning
- clustering-based unsupervised learning

### DeepCluster v2 更像
- prototype learning
- prototype-based SSL
- clustering-based representation learning

---

## 11. 评估时使用什么指标

这取决于你把 DeepCluster 当成什么来用。

### 如果关注聚类质量
那么可用：

- ACC
- NMI
- ARI
- AMI
- Silhouette Score

也就是 **聚类指标**。

### 如果关注表示学习质量 / 预训练效果
那么可用：

- Accuracy
- F1
- AUROC
- linear evaluation accuracy
- fine-tuning performance

也就是 **分类或下游任务指标**。

---

## 12. 方法定位总结

更准确地说，DeepCluster 系列方法：

- **不是单纯的聚类算法**
- **也不只是传统监督分类方法**
- 而是 **由聚类驱动的无监督表示学习框架**

其中：

- **v1** 更偏向伪标签学习
- **v2** 更偏向原型学习

---

## 13. 一句话总结

### DeepCluster v1
> 利用聚类生成的离散簇标签作为伪标签来训练分类网络，是一种典型的聚类驱动伪标签学习方法。

### DeepCluster v2
> 保留聚类生成监督信号的基本框架，但将离散伪标签分类替换为特征与聚类中心之间的显式匹配，更适合被理解为聚类驱动的原型学习方法。

---

## 14. 超简版对比

```text
DeepCluster v1:
聚类 -> 得到 cluster id -> 把 cluster id 当伪标签训练分类器

DeepCluster v2:
聚类 -> 得到 centroids / prototypes -> 让特征去匹配这些中心