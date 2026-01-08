# np.linalg.svd vs np.linalg.eigh

## 一、基本定义

### SVD（奇异值分解）

将**任意**矩阵 $A \in \mathbb{R}^{m \times n}$ 分解为：

$$A = U \Sigma V^T$$

| 成分 | 形状 | 含义 |
|------|------|------|
| $U$ | $m \times m$（或 $m \times k$） | 左奇异向量（正交矩阵） |
| $\Sigma$ | $m \times n$（或 $k \times k$） | 对角矩阵，奇异值 $\sigma_i \geq 0$ |
| $V^T$ | $n \times n$（或 $k \times n$） | 右奇异向量（正交矩阵） |

> $k = \min(m, n)$，取决于 `full_matrices` 参数。

### Eigh（对称矩阵特征分解）

将**实对称矩阵** $A \in \mathbb{R}^{n \times n}$（$A = A^T$）分解为：

$$A = Q \Lambda Q^T$$

| 成分 | 形状 | 含义 |
|------|------|------|
| $Q$ | $n \times n$ | 特征向量矩阵（正交矩阵） |
| $\Lambda$ | $n \times n$ | 对角矩阵，特征值 $\lambda_i$（可正可负可零） |

---

## 二、API 对比

### np.linalg.svd

```python
U, s, Vt = np.linalg.svd(A, full_matrices=True)
```

| 参数 | 说明 |
|------|------|
| `A` | 任意形状的矩阵 |
| `full_matrices=True` | 返回完整 $U$（$m \times m$）和 $V^T$（$n \times n$） |
| `full_matrices=False` | 经济分解：$U$（$m \times k$），$V^T$（$k \times n$） |

返回值：
- `U`：左奇异向量
- `s`：一维数组，奇异值（**降序排列**，$\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$）
- `Vt`：右奇异向量的转置 $V^T$

### np.linalg.eigh

```python
vals, vecs = np.linalg.eigh(A)
```

| 参数 | 说明 |
|------|------|
| `A` | 必须是实对称（或 Hermitian）矩阵 |

返回值：
- `vals`：一维数组，特征值（**升序排列**，$\lambda_1 \leq \lambda_2 \leq \cdots$）
- `vecs`：特征向量矩阵，第 $i$ 列对应 $\lambda_i$

---

## 三、核心区别

| | `np.linalg.svd` | `np.linalg.eigh` |
|--|-----------------|-------------------|
| **输入要求** | 任意矩阵 | 必须对称 |
| **分解形式** | $A = U \Sigma V^T$（三个矩阵） | $A = Q \Lambda Q^T$（两个矩阵） |
| **值的范围** | $\sigma_i \geq 0$ | $\lambda_i$ 可正可负可零 |
| **默认排序** | **降序**（最大在前） | **升序**（最小在前） |
| **速度** | 较慢 | 更快（利用对称性优化） |
| **底层调用** | LAPACK `dgesdd` | LAPACK `dsyevd` |
| **复杂度** | $O(mn^2)$，$m \geq n$ | $O(n^3)$（但常数更小） |

---

## 四、对称矩阵上两者的关系

当 $A$ 是实对称矩阵时：

$$\sigma_i = |\lambda_i|$$

- 奇异值 = 特征值的**绝对值**
- 如果 $A$ 是半正定的（$\lambda_i \geq 0$），则 $\sigma_i = \lambda_i$，两者数值完全一致
- SVD 的 $U$ 和 $V$ 都等于特征向量矩阵 $Q$（可能差一个符号）

---

## 五、内部算法流程

### SVD 内部流程

1. **双对角化（Bidiagonalization）**
   - 通过 Householder 变换将 $A$ 化为双对角矩阵 $B$：$A = U_1 B V_1^T$
   - 复杂度：$O(mn^2)$

2. **求奇异值（Divide-and-Conquer）**
   - 将双对角矩阵 $B$ 递归分裂为更小的子问题
   - 自底向上合并，通过 secular equation 求解奇异值
   - 复杂度：$O(n^2)$ ~ $O(n^3)$

3. **回代，组装 $U$ 和 $V^T$**
   - 将各阶段的变换矩阵组合
   - 复杂度：$O(mn^2)$

### Eigh 内部流程

1. **三对角化（Tridiagonalization）**
   - 通过 Householder 变换将对称矩阵 $A$ 化为三对角矩阵 $T$：$A = Q_1 T Q_1^T$
   - 复杂度：$O(n^3)$

2. **求特征值（Divide-and-Conquer）**
   - 对三对角矩阵 $T$ 用分治法求特征值和特征向量
   - 复杂度：$O(n^2)$ ~ $O(n^3)$

3. **回代，组装 $Q$**
   - 合并变换矩阵
   - 复杂度：$O(n^3)$

---

## 六、使用场景建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 谱聚类（拉普拉斯矩阵） | `eigh` | $L$ 是对称半正定，`eigh` 更快且结果自动升序 |
| PCA（协方差矩阵） | `eigh` 或 `svd` | 协方差矩阵对称，两者均可；直接对数据矩阵做 SVD 更数值稳定 |
| 矩阵近似 / 压缩 | `svd` | 需要低秩近似，SVD 的截断形式最优（Eckart-Young 定理） |
| 非方阵 / 非对称矩阵 | `svd` | `eigh` 不适用 |
| Grassmann 距离 | `svd` | 需要对非对称的 $A^T B$ 做分解 |

---

## 七、代码示例

```python
import numpy as np

# 构造一个对称半正定矩阵（拉普拉斯矩阵）
W = np.array([[0, 1, 0.5],
              [1, 0, 0.8],
              [0.5, 0.8, 0]])
D = np.diag(W.sum(axis=1))
L = D - W

# ========== eigh ==========
vals, vecs = np.linalg.eigh(L)
print("eigh 特征值（升序）:", vals)
# 输出: [≈0, λ₂, λ₃]

# ========== svd ==========
U, s, Vt = np.linalg.svd(L)
print("svd 奇异值（降序）:", s)
# 输出: [λ₃, λ₂, ≈0]  （和 eigh 顺序相反）

# 对于半正定对称矩阵，两者数值一致（顺序相反）
print("一致性验证:", np.allclose(np.sort(vals), np.sort(s)))  # True
```
