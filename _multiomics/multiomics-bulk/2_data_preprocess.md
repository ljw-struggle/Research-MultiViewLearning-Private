# 2_data_preprocess.R 完整代码解析

## 概述
这个R脚本是一个完整的TCGA多组学数据预处理管道，用于处理mRNA、miRNA、DNA甲基化、蛋白质组学(RPPA)、拷贝数变异(CNV)和简单核苷酸变异(SNV)六种组学数据。脚本实现了从原始数据到分析就绪数据的完整转换流程。

```bash
# 在终端运行
nohup R --no-save --no-restore < 2_data_preprocess.R > brca_process_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 依赖包和初始化设置

```r
library(reshape2)              # 数据重塑和转换
library(kableExtra)            # 表格美化
library(plotly)                # 交互式图表
library(vsn)                   # 方差稳定标准化
library(tibble)                # 现代数据框操作
library(pheatmap)              # 热图绘制
library(SummarizedExperiment)  # 基因组数据容器
source('./2_data_functions.R') # 加载自定义函数
setwd('~/Bioinfor/Bioinfor-MMBEMB-Private/data')
```

### 项目配置
```r
project <- 'BRCA'                        # 癌症类型：乳腺癌
dataset <- 'TCGA'                        # 数据集：TCGA
trait <- 'paper_BRCA_Subtype_PAM50'      # 研究特征：PAM50乳腺癌亚型
```

**设计原因**：
- 参数化配置便于切换不同癌症类型
- PAM50是乳腺癌的标准分子分型系统
- 模块化设计支持不同研究需求

## 第一部分：元数据文件生成

### 功能说明
从mRNA数据中提取和整理样本的临床信息，创建统一的元数据文件。

### 代码解析
```r
# 加载mRNA数据获取元数据
load(paste0('./data/TCGA-',project,'/mRNA/mRNA.rda'))

# 提取关键的临床信息
coldata <- colData(data)
datMeta <- as.data.frame(coldata[,c('patient','race' , 'gender' , 'sample_type' , trait)])

# 数据清洗步骤
datMeta <- datMeta[!(is.na(datMeta[[trait]])) , ]           # 移除trait缺失的样本
datMeta <- datMeta[!(duplicated(datMeta[ , c('patient' , trait)])) , ]  # 移除重复样本
datMeta[[trait]] <- factor(datMeta[[trait]])                # 转换为因子类型
rownames(datMeta) <- datMeta$patient                        # 设置行名为患者ID

# 保存元数据文件
write.csv(datMeta , file = paste0('./data/TCGA-',project,'/datMeta.csv'))
```

**设计原因**：
1. **数据一致性**：确保所有组学数据使用相同的样本集
2. **质量控制**：移除无效和重复样本
3. **标准化**：统一的元数据格式便于后续分析
4. **可追溯性**：保留患者ID作为唯一标识符

## 第二部分：mRNA预处理

### 功能说明
对mRNA表达数据进行完整的预处理，包括数据清洗、标准化和差异表达分析。

### 代码解析
```r
# 1. 数据加载和初始处理
load(paste0('./data/TCGA-',project,'/mRNA/mRNA.rda'))
count_mtx <- assay(data)
colnames(count_mtx) <- substr(colnames(count_mtx) , 1, 12)    # 截取TCGA样本ID
count_mtx <- count_mtx[, !(duplicated(colnames(count_mtx)))]   # 移除重复样本

# 2. 数据匹配
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)
common_idx <- intersect(colnames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[ , common_idx]
datMeta <- datMeta[common_idx , ]

# 3. 差异表达分析
diff_expr_res <- diff_expr(count_mtx , datMeta , trait , 500 , 'mRNA')

# 4. 保存结果
datExpr <- diff_expr_res$datExpr    # VST转换后的表达数据
datMeta <- diff_expr_res$datMeta    # 过滤后的元数据
dds <- diff_expr_res$dds           # DESeq2对象
top_genes <- diff_expr_res$top_genes  # 差异表达基因
save(datExpr, datMeta, dds, top_genes, file=paste0('./data/',dataset,'/',project,'/mRNA_processed.RData'))
```

**关键步骤说明**：
1. **样本ID标准化**：TCGA样本ID格式复杂，截取前12位保证一致性
2. **数据匹配**：确保表达数据和元数据样本完全对应
3. **差异表达分析**：使用自定义`diff_expr`函数进行完整分析
4. **特征选择**：选择top 500个差异表达基因

## 第三部分：miRNA预处理

### 功能说明
处理miRNA测序数据，提取计数矩阵并进行差异表达分析。

### 代码解析
```r
# 1. 数据加载和结构解析
load(paste0('./data/TCGA-',project,'/miRNA/miRNA.rda'))

# 2. 提取计数矩阵（每3列为一组，第2列是计数）
read_count <- data.frame(row.names = data$miRNA_ID)
read_per_million <- data.frame(row.names = data$miRNA_ID)

for (i in 2:dim(data)[2]) {
  if (i%%3 == 2) {                    # 每3列中的第2列：原始计数
    read_count <- cbind(read_count , data[ , i] )
  }
  if (i%%3 == 0) {                    # 每3列中的第3列：RPM标准化
    read_per_million <- cbind(read_per_million , data[ , i])
  }
}

# 3. 提取样本名称
colname_read_count <- c()
for (i in 2:dim(data)[2]) {
  if (i%%3 == 2) {
    colname_read_count <- c(colname_read_count , 
                           substr(strsplit(colnames(data)[i] , '_')[[1]][3], 1, 12))
  }
}
colnames(read_count) <- colname_read_count

# 4. 后续处理步骤与mRNA类似
# ... (数据匹配、差异表达分析、保存结果)
```

**miRNA特殊处理**：
1. **数据格式复杂**：TCGA miRNA数据每个样本有3列（计数、RPM、cross-mapped）
2. **列名解析**：需要从复杂的列名中提取样本ID
3. **特征数量**：选择top 200个miRNA（相比mRNA数量更少）

## 第四部分：DNA甲基化预处理

### 功能说明
处理DNA甲基化数据，进行质量控制和特征选择。

### 代码解析
```r
# 1. 数据加载和质量控制
load(paste0('./data/TCGA-',project,'/DNAm/DNAm.rda'))
count_mtx <- assay(data)

# 2. 移除缺失值过多的CpG位点
to_keep = complete.cases(count_mtx)
count_mtx <- t(count_mtx[to_keep,])     # 转置：样本为行，CpG为列
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)

# 3. 基于方差的特征选择
cpg_variances <- colVars(count_mtx)     # 计算每个CpG的方差
sorted_indices <- order(cpg_variances, decreasing = TRUE)  # 按方差排序
num_top_cpg <- 200000                   # 选择top 200,000个CpG位点
top_cpg_indices <- sorted_indices[1:num_top_cpg]
count_mtx <- count_mtx[ , top_cpg_indices]

# 4. 使用Lasso回归进行特征选择
phenotypes <- datMeta[,c('patient' , trait )]
traits <- c(trait)

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, traits, nFolds = 10)
})

# 5. 提取重要的CpG位点
cpg_sites <- c()
for (res in traitResults) { 
  trait_coefs <- coef(res$model , s = "lambda.min")
  cpg_sites_tmp <- c()
  for (coefs in trait_coefs) {
    class_coefs <- rownames(coefs)[which(coefs != 0)]
    class_coefs <- class_coefs[2:length(class_coefs)]  # 移除截距项
    cpg_sites_tmp <- unique(c(cpg_sites_tmp ,class_coefs ))
  }
  cpg_sites[[res$trait]] <- cpg_sites_tmp
}
```

**甲基化数据特点**：
1. **高维数据**：约48万个CpG位点，需要降维
2. **β值分布**：0-1之间的连续值，代表甲基化程度
3. **二阶段特征选择**：方差过滤 + Lasso回归
4. **生物学意义**：选择与表型相关的甲基化位点

## 第五部分：蛋白质组学(RPPA)预处理

### 功能说明
处理反向蛋白芯片(RPPA)数据，识别与表型相关的蛋白质。

### 代码解析
```r
# 1. 数据加载和转换
load(paste0('./data/TCGA-',project,'/RPPA/RPPA.rda'))
count_mtx <- t(data[  , 6:ncol(data) ])      # 转置，前5列为注释信息
colnames(count_mtx) <- data$peptide_target   # 蛋白质名称
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)

# 2. 处理缺失值
count_mtx <- count_mtx[,colSums(is.na(count_mtx))<0.5*nrow(count_mtx)] # 移除50%以上缺失的蛋白质

# 3. 缺失值填充
for(i in 1:ncol(count_mtx)){
  count_mtx[is.na(count_mtx[,i]), i] <- mean(count_mtx[,i], na.rm = TRUE)
}

# 4. 使用Lasso回归选择重要蛋白质
phenotypes <- datMeta[,c('patient' , trait , 'race' , 'gender')]
traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, trait, nFolds = 10)
})
```

**RPPA数据特点**：
1. **数据稀疏**：蛋白质种类有限（约200个）
2. **缺失值问题**：实验技术限制导致缺失值较多
3. **标准化程度高**：已经过厂家标准化处理
4. **功能聚焦**：主要是癌症相关的信号通路蛋白

## 第六部分：拷贝数变异(CNV)预处理

### 功能说明
处理拷贝数变异数据，识别与表型相关的基因拷贝数变化。

### 代码解析
```r
# 1. 数据加载和格式处理
load(paste0('./data/TCGA-',project,'/CNV/CNV.rda'))
count_mtx <- t(assay(data))

# 2. 样本名称处理
rownames_mtx <- c()
for (name in strsplit(rownames(count_mtx) , ',')) {
  rownames_mtx <- c(rownames_mtx , substr(name[1] ,1, 12))
}
rownames(count_mtx) <- rownames_mtx

# 3. 质量控制
count_mtx <- count_mtx[,colSums(is.na(count_mtx))<0.5*nrow(count_mtx)]
count_mtx[is.na(count_mtx)] <- 0    # 缺失值填充为0（正常拷贝数）

# 4. 对数转换
count_mtx_log <- log(count_mtx)     # 对数转换使数据更接近正态分布

# 5. 使用Lasso回归识别重要CNV
traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx_log, phenotypes, traits, nFolds = 10)
})
```

**CNV数据特点**：
1. **数据范围**：拷贝数通常在0-4之间
2. **偏态分布**：需要对数转换
3. **基因水平**：已经汇总到基因水平的拷贝数
4. **缺失值含义**：通常代表正常拷贝数（2个拷贝）

## 第七部分：简单核苷酸变异(SNV)预处理

### 功能说明
处理体细胞突变数据，将MAF格式的突变事件转换为样本-基因突变矩阵，并识别与表型相关的显著突变基因。

### 代码解析
```r
# 1. 数据加载和结构检查
load(paste0('./data/TCGA-',project,'/SNV/SNV.rda'))
print("SNV data structure:")
print(str(data))

# 2. 样本ID提取和标准化
if("Tumor_Sample_Barcode" %in% colnames(data)) {
  snv_samples <- substr(data$Tumor_Sample_Barcode, 1, 12)
  data$patient_id <- snv_samples
}

# 3. 构建突变矩阵（样本 x 基因）
mutated_genes <- unique(data_filtered$Hugo_Symbol)
mutation_matrix <- matrix(0, 
                         nrow = length(common_idx), 
                         ncol = length(mutated_genes),
                         dimnames = list(common_idx, mutated_genes))

# 4. 填充突变矩阵（1=突变，0=野生型）
for(i in 1:nrow(data_filtered)) {
  sample_id <- data_filtered$patient_id[i]
  gene <- data_filtered$Hugo_Symbol[i]
  mutation_matrix[sample_id, gene] <- 1
}

# 5. 突变频率分析和过滤
gene_mutation_freq <- colSums(mutation_matrix) / nrow(mutation_matrix)
min_mutation_freq <- 0.05  # 保留突变频率≥5%的基因
frequent_genes <- names(gene_mutation_freq)[gene_mutation_freq >= min_mutation_freq]

# 6. Fisher精确检验识别显著差异突变
for(gene in frequent_genes) {
  contingency_table <- matrix(c(
    sum(gene_mutations[group1_samples]),    # 亚型1中的突变
    sum(gene_mutations[group2_samples]),    # 亚型2中的突变
    sum(!gene_mutations[group1_samples]),   # 亚型1中的野生型
    sum(!gene_mutations[group2_samples])    # 亚型2中的野生型
  ), nrow = 2, byrow = TRUE)
  
  fisher_test <- fisher.test(contingency_table)
}

# 7. 多重检验校正和基因选择
p_values_adj <- p.adjust(p_values, method = "fdr")
significant_genes <- names(p_values_adj)[p_values_adj < 0.05]
```

**SNV数据特点**：
1. **数据格式**：MAF (Mutation Annotation Format)，每行一个突变事件
2. **数据转换**：从事件数据转换为二进制矩阵（突变/野生型）
3. **稀疏性高**：大多数基因在大多数样本中未突变
4. **统计挑战**：需要处理稀疏数据的统计检验

**技术创新**：
1. **智能特征选择**：优先选择显著差异基因，备选高频突变基因
2. **突变负荷计算**：反映肿瘤基因组不稳定性
3. **精确统计检验**：Fisher精确检验适合稀疏数据
4. **多重检验校正**：FDR控制假阳性率

**生物学意义**：
- **驱动基因识别**：发现与癌症亚型相关的关键突变
- **肿瘤异质性**：比较不同亚型的突变模式
- **精准医疗**：指导靶向治疗和个体化方案
- **预后标志物**：突变负荷作为预后指标

## 数据预处理策略比较

| 数据类型 | 特征选择策略 | 数据转换 | 特征数量 | 主要挑战 |
|----------|------------|----------|----------|----------|
| mRNA | 差异表达分析 | VST | 500 | 高维度、表达水平差异大 |
| miRNA | 差异表达分析 | VST | 200 | 特征数量少、表达水平低 |
| DNAm | 方差过滤+Lasso | 无 | 动态 | 超高维度、连续值 |
| RPPA | Lasso回归 | 无 | 动态 | 缺失值多、特征数量少 |
| CNV | Lasso回归 | 对数转换 | 动态 | 偏态分布、稀疏数据 |
| SNV | 频率过滤+Fisher检验 | 事件→矩阵 | 动态 | 稀疏数据、统计功效低 |

### SNV数据预处理的特殊考虑

SNV数据与其他组学数据类型具有根本性差异，需要特殊的处理策略：

#### 1. **数据结构转换**
```r
# 从事件表转换为样本-基因矩阵
# 输入：每行一个突变事件 (MAF格式)
# 输出：样本×基因的二进制矩阵
```

#### 2. **稀疏性处理**
- **问题**：多数基因在多数样本中未突变（矩阵稀疏度>95%）
- **解决**：频率过滤（仅保留突变频率≥5%的基因）
- **优势**：提高统计检验的功效

#### 3. **统计方法选择**
- **问题**：连续型数据的方法（如Lasso）不适用
- **解决**：使用Fisher精确检验处理分类数据
- **优势**：适合小样本和稀疏数据，无分布假设

#### 4. **生物学意义整合**
- **突变负荷**：反映基因组不稳定性
- **驱动基因**：识别癌症相关的关键突变
- **亚型特异性**：发现不同亚型的突变模式

#### 5. **质量控制策略**
```r
# 多层次质量控制
print(paste0("Total mutations: ", nrow(data_filtered)))
print(paste0("Mutated genes: ", length(mutated_genes)))  
print(paste0("Frequent genes (≥5%): ", length(frequent_genes)))
print(paste0("Significant genes (FDR<0.05): ", length(significant_genes)))
```

| 步骤    | 数据类型   | 样本对齐方式      | 特征选择方法         | 保存内容                     |
| ----- | ------ | ----------- | -------------- | ------------------------ |
| Meta  | 临床信息   | colData     | NA             | `datMeta.csv`            |
| mRNA  | 基因表达   | 样本 ID 交集    | 差异表达分析         | top500 genes             |
| miRNA | 小RNA表达 | 样本 ID 交集    | 差异表达分析         | top200 miRNAs            |
| DNAm  | CpG 位点 | 去NA + 变异排序  | Lasso（cvTrait） | top CpGs per subtype     |
| RPPA  | 蛋白质    | 样本对齐 + 缺失处理 | Lasso（cvTrait） | top proteins per subtype |
| CNV   | 拷贝数变异  | 对齐 + log 转换 | Lasso（cvTrait） | top CNVs per subtype     |
| SNV   | 体细胞突变  | MAF格式转矩阵   | Fisher精确检验     | significant mutated genes |

| 层级        | 组学名称            | 典型技术      |
| --------- | --------------- | --------- |
| mRNA 表达   | transcriptomics | RNA-seq   |
| miRNA 表达  | **miRNomics**   | miRNA-seq |
| DNA 甲基化   | epigenomics     | 450k/850k |
| CNV 拷贝数   | genomics        | SNP-array/WES |
| SNV 突变    | **mutomics**    | WES/WGS   |
| 蛋白质表达     | proteomics      | RPPA      |
| 代谢物水平     | metabolomics    | MS/LC     |

## 设计优势

### 1. 统一的处理框架
- 所有数据类型使用相同的样本集
- 统一的元数据格式
- 一致的保存和命名规范

### 2. 质量控制策略
- 缺失值处理
- 异常值检测
- 重复样本去除

### 3. 特征选择方法
- 根据数据特点选择合适的方法
- 考虑生物学意义和统计显著性
- 平衡特征数量和信息量

### 4. 可重现性
- 固定随机种子
- 参数化配置
- 详细的中间结果保存

## 使用建议

### 1. 内存管理
```r
# 及时清理大型对象
rm(count_mtx_log)
gc()  # 强制垃圾回收
```

### 2. 参数调整
- 根据样本数量调整特征选择数量
- 根据计算资源调整交叉验证折数
- 根据数据质量调整缺失值阈值

### 3. 质量检查
- 检查每步的样本数量变化
- 验证特征选择的合理性
- 确认数据分布的正态性

### 4. 并行化优化
```r
# 对于大规模数据，考虑并行处理
library(parallel)
options(mc.cores = detectCores())
```

### 5. SNV数据特殊建议
```r
# 调整突变频率阈值
min_mutation_freq <- 0.03  # 降低到3%获得更多基因（适合稀有癌症）
min_mutation_freq <- 0.10  # 提高到10%获得更可靠的基因（适合常见癌症）

# 验证突变注释质量
table(data$Variant_Classification)  # 检查突变类型分布
table(data$Variant_Type)           # 检查变异类型分布

# 考虑突变类型过滤
# 仅保留功能性突变（如错义突变、无义突变等）
functional_mutations <- data[data$Variant_Classification %in% 
  c("Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", 
    "Frame_Shift_Ins", "Splice_Site"), ]
```

## 潜在问题和解决方案

### 1. 内存不足
- 分批处理数据
- 使用更高效的数据结构
- 及时清理临时对象

### 2. 样本不匹配
- 严格的ID匹配检查
- 详细的样本过滤日志
- 交叉验证匹配结果

### 3. 特征选择偏差
- 使用交叉验证避免过拟合
- 考虑多种特征选择方法
- 验证生物学意义

### 4. 数据分布问题
- 适当的数据转换
- 分布检查和可视化
- 鲁棒性方法的使用

### 5. SNV数据特有问题
- **突变注释不一致**: 使用标准化的突变分类系统
- **假阳性突变**: 实施严格的质量过滤标准
- **样本纯度问题**: 考虑肿瘤纯度对突变检测的影响
- **统计功效低**: 合理设置频率阈值和选择适当的统计方法

```r
# SNV质量控制示例
# 过滤低质量突变
high_quality_mutations <- data[data$FILTER == "PASS", ]

# 检查突变分布
hist(gene_mutation_freq, 
     main="Gene Mutation Frequency Distribution",
     xlab="Mutation Frequency", 
     ylab="Number of Genes")
```

## 总结

这个预处理脚本实现了TCGA多组学数据的完整处理流程，具有以下特点：

1. **完整性**：涵盖六种主要组学数据类型
2. **标准化**：统一的处理框架和质量控制
3. **灵活性**：参数化配置支持不同研究需求
4. **可重现性**：详细记录和标准化流程
5. **效率**：针对不同数据类型的优化策略
6. **创新性**：SNV突变数据的智能处理和显著性分析

通过这个脚本，研究人员可以获得高质量、分析就绪的多组学数据，为后续的整合分析奠定坚实基础。现在包含的六种数据类型（转录组、表观基因组、基因组变异、蛋白质组）为全面理解癌症的分子机制提供了完整的数据基础。