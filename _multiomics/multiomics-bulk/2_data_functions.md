# 2_data_functions.R 代码解析

## 概述
这个R脚本包含了生物信息学数据预处理的关键函数，主要用于TCGA多组学数据的预处理、特征选择和差异表达分析。

## 依赖包说明

```r
library(MethylPipeR)  # DNA甲基化数据处理
library(glmnet)       # 正则化线性模型（Lasso/Ridge）
library(igraph)       # 图论和网络分析
library(DESeq2)       # RNA-seq差异表达分析
library(dplyr)        # 数据处理和操作
library(ggplot2)      # 数据可视化
library(edgeR)        # RNA-seq数据分析
library(WGCNA)        # 加权基因共表达网络分析
```

## 函数详细解析

### 1. removeTraitNAs函数

#### 功能
移除指定trait中含有缺失值(NA)的样本，并对其他相关数据框进行同步筛选。

#### 参数
- `traitDF`: 包含trait信息的数据框
- `otherDFs`: 其他需要同步筛选的数据框列表
- `trait`: 要检查缺失值的trait名称

#### 实现逻辑
```r
removeTraitNAs <- function(traitDF, otherDFs, trait) {
  # 1. 找到trait中非缺失值的行
  rowsToKeep <- !is.na(traitDF[[trait]])
  
  # 2. 筛选trait数据框
  traitDF <- traitDF[rowsToKeep, ]
  
  # 3. 对其他数据框进行同步筛选
  otherDFs <- lapply(otherDFs, function(df) {
    if (is.data.frame(df) || is.matrix(df)) {
      df[rowsToKeep, ]          # 数据框/矩阵：保留对应行
    } else if (is.null(df)) {
      df                        # NULL值：保持不变
    } else {
      df[rowsToKeep]           # 向量：保留对应元素
    }
  })
  
  # 4. 返回处理后的数据
  list(traitDF = traitDF, otherDFs = otherDFs)
}
```

#### 设计原因
- **数据一致性**: 确保所有数据框的样本顺序保持一致
- **灵活性**: 可以处理不同类型的数据结构（数据框、矩阵、向量）
- **预处理必要性**: 机器学习模型通常无法处理缺失值

### 2. cvTrait函数

#### 功能
使用交叉验证的Lasso回归预测特定trait，主要用于DNA甲基化数据的特征选择。

#### 参数
- `trainMethyl`: 训练用的甲基化数据矩阵
- `trainPhenotypes`: 训练用的表型数据
- `trait`: 要预测的trait名称
- `nFolds`: 交叉验证的折数

#### 实现逻辑
```r
cvTrait <- function(trainMethyl, trainPhenotypes, trait, nFolds) {
  # 1. 数据清洗：移除缺失值
  print(paste0('Removing rows with missing ', trait, ' from training data.'))
  trainRemoveNAResult <- removeTraitNAs(trainPhenotypes, list(trainMethyl = trainMethyl), trait)
  trainPhenotypes <- trainRemoveNAResult$traitDF
  trainMethyl <- trainRemoveNAResult$otherDFs$trainMethyl
  
  # 2. 拟合Lasso回归模型
  print('Fitting lasso model')
  methylModel <- cv.glmnet(
    x = trainMethyl,                           # 特征矩阵（甲基化数据）
    y = as.factor(trainPhenotypes[[trait]]),   # 标签（转换为因子）
    seed = 42,                                 # 随机种子
    family = 'multinomial',                    # 多分类问题
    type.measure = "class",                    # 评估指标：分类错误率
    alpha = 1,                                 # L1正则化（Lasso）
    nFolds = nFolds,                          # 交叉验证折数
    parallel = TRUE,                          # 并行计算
    trace.it = 1                              # 显示训练过程
  )
  
  # 3. 返回模型结果
  print(methylModel)
  list(trait = trait, model = methylModel)
}
```

#### 设计原因
- **特征选择**: Lasso回归能够自动选择重要的甲基化位点
- **过拟合防护**: 交叉验证帮助选择最优的正则化参数
- **多分类支持**: 适用于多种疾病亚型的分类问题
- **并行计算**: 提高大规模数据的处理效率

### 3. diff_expr函数

#### 功能
进行差异表达分析的完整流程，包括数据预处理、异常值检测、标准化和差异表达分析。

#### 参数
- `count_mtx`: 原始计数矩阵
- `datMeta`: 样本元数据
- `trait`: 分组变量
- `n_genes`: 每个对比中选择的top基因数量
- `modality`: 数据类型（'miRNA'或其他）

#### 实现步骤

##### 步骤1: 低表达基因过滤
```r
# 移除表达量为0的基因
to_keep = rowSums(count_mtx) > 0
count_mtx <- count_mtx[to_keep,]

# 对非miRNA数据进行进一步过滤
if (modality != 'miRNA') {
  to_keep = filterByExpr(datExpr , group = datMeta[[trait]])
  count_mtx = count_mtx[to_keep,]
}
```
**原因**: 
- 移除无信息的基因，减少多重检验负担
- miRNA数据通常基因数量较少，过滤标准更宽松

##### 步骤2: 异常值检测
```r
# 计算双相关系数矩阵
absadj = count_mtx %>% bicor %>% abs
netsummary = fundamentalNetworkConcepts(absadj)
ku = netsummary$Connectivity

# 计算连接度的Z分数
z.ku = (ku-mean(ku))/sqrt(var(ku))

# 移除连接度过低的样本（Z分数 < -2）
to_keep = z.ku > -2
count_mtx <- count_mtx[,to_keep]
datMeta <- datMeta[to_keep,]
```
**原因**:
- 使用基于连接度的方法检测异常样本
- 双相关系数对异常值更加鲁棒
- Z分数阈值-2是统计学上的标准

##### 步骤3: DESeq2标准化
```r
# 创建DESeq2对象
datMeta[[trait]] <- as.factor(datMeta[[trait]])
dds = DESeqDataSetFromMatrix(countData = count_mtx, colData = datMeta, 
                            design = formula(paste("~ 0 +",trait)))

# 进行差异表达分析
dds = DESeq(dds)
```
**原因**:
- DESeq2是RNA-seq数据分析的金标准
- 使用负二项分布模型更适合计数数据
- 内置的标准化方法考虑了测序深度和组成偏差

##### 步骤4: 方差稳定转换(VST)
```r
# 根据基因数量选择转换参数
nsub_check = sum( rowMeans( counts(dds, normalized=TRUE)) > 5 )
if (nsub_check < 1000) {
  vsd = vst(dds , nsub= nsub_check)
} else {
  vsd = vst(dds)
}

datExpr_vst = assay(vsd)
datMeta_vst = colData(vsd)
```
**原因**:
- VST转换使方差在不同表达水平上保持稳定
- 对于后续的相关分析和聚类分析更适合
- 自适应参数选择提高转换效果

##### 步骤5: 差异表达分析和top基因选择
```r
# 进行所有可能的两两对比
subtypes <- levels(as.factor(datMeta[[trait]]))
top_genes = c()
for (subtype1 in subtypes[1:length(subtypes)-1]) {
  subtypes = subtypes[subtypes != subtype1]
  for (subtype2 in subtypes)  {
    if (subtype1 != subtype2) {
      res <- results(dds , contrast = c(trait , subtype1 , subtype2))
    }
    top_genes = unique(c(top_genes , head(order(res$padj) , n_genes) ))
  }
}
```
**原因**:
- 全面对比所有亚型组合
- 基于校正p值选择最显著的基因
- 去重确保基因列表的唯一性

#### 返回值
```r
list(dds = dds,                    # DESeq2对象
     datExpr = datExpr_vst,        # VST转换后的表达数据
     datMeta = datMeta_vst,        # 对应的元数据
     top_genes = top_genes)        # 差异表达的top基因
```

## 代码设计特点

### 1. 模块化设计
- 每个函数职责明确
- 函数间通过标准接口通信
- 便于维护和扩展

### 2. 鲁棒性考虑
- 异常值检测和处理
- 多种数据类型的兼容性
- 参数自适应选择

### 3. 生物学意义
- 遵循生物信息学标准流程
- 考虑了数据的统计分布特性
- 结合了领域最佳实践

### 4. 可视化支持
- 内置质量控制图表
- 便于结果解释和验证

## 使用建议

1. **数据预处理顺序**: 先运行`removeTraitNAs`清理数据，再进行分析
2. **参数选择**: 根据数据规模调整`nFolds`和`n_genes`
3. **质量控制**: 注意观察生成的图表，确保数据质量
4. **计算资源**: 大规模数据建议使用并行计算选项

## 注意事项

1. **内存需求**: 大规模数据可能需要大量内存
2. **计算时间**: 某些步骤（如交叉验证）可能耗时较长
3. **参数调优**: 可能需要根据具体数据调整阈值参数
4. **结果解释**: 需要结合生物学背景解释分析结果