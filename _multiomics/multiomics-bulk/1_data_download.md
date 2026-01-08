# 1_data_download.R 代码解析

## 概述
这个R脚本是一个完整的TCGA（The Cancer Genome Atlas）多组学数据下载工具，使用TCGAbiolinks包从NCI GDC (Genomic Data Commons)数据门户自动下载六种不同类型的癌症基因组数据。

## 依赖包和环境设置

### 核心依赖包
```r
library(TCGAbiolinks)        # TCGA数据访问和下载的专用包
library(SummarizedExperiment) # 基因组数据容器和操作
library(dplyr)               # 数据操作和管道操作
```

**包功能说明**：
- **TCGAbiolinks**: 专门为TCGA数据设计的R包，提供查询、下载和预处理功能
- **SummarizedExperiment**: Bioconductor的核心数据结构，用于存储基因组实验数据
- **dplyr**: 提供高效的数据框操作功能

### 系统配置
```r
options(future.globals.maxSize = 10 * 1024^3)  # 设置内存大小为10GB
getwd()  # 显示当前工作目录
```

**配置原因**：
- **内存限制调整**: TCGA数据文件通常很大，默认内存限制可能不够
- **工作目录确认**: 确保数据下载到正确的位置

## 第一部分：项目信息查询

### 数据源信息获取
```r
# 1. 获取GDC数据门户信息
getGDCInfo()                    # 获取GDC服务器的基本信息
getGDCprojects()                # 获取所有可用的项目列表
getGDCprojects()$project_id     # 提取项目ID列表
```

**功能详解**：
1. **getGDCInfo()**: 
   - 检查GDC服务器状态
   - 获取API版本信息
   - 确认连接可用性

2. **getGDCprojects()**: 
   - 返回所有可用的癌症项目
   - 包括TCGA、TARGET、CPTAC等数据集
   - 提供项目的基本统计信息

3. **project_id提取**:
   - 获取如"TCGA-BRCA"、"TCGA-LUAD"等项目标识符
   - 用于后续的数据查询和下载

### 项目选择和配置
```r
# 2. 选择特定的癌症项目
tcga_project <- "TCGA-BRCA"     # 乳腺癌项目
# getProjectSummary(tcga_project) # 获取项目详细信息（已注释）
```

**设计考虑**：
- **TCGA-BRCA**: 乳腺癌是数据最完整、样本量最大的癌症类型之一
- **参数化设计**: 便于切换不同癌症类型
- **项目摘要**: 可选择查看项目的详细统计信息

## 第二部分：数据下载主流程

### 目录检查和创建
```r
# 3. 智能下载控制
if (dir.exists(paste0('./data/', tcga_project))) {
  cat("TCGA data folder already exists, skipping download...\n")
} else {
  cat("Downloading TCGA data...\n")
  dir.create(paste0('./data/', tcga_project), recursive = TRUE)
  # ... 下载逻辑
}
```

**注意**：代码中存在路径不一致问题：
- 检查路径：`'./data/'`
- 创建路径：`'./data/'`

**设计优势**：
- **避免重复下载**: 检查是否已存在数据
- **自动目录管理**: 递归创建必要的目录结构
- **用户反馈**: 提供清晰的状态信息

### 数据类型1：DNA甲基化(DNA Methylation)

```r
# DNA Methylation 
query.met <- GDCquery(
  project = tcga_project, 
  data.category = "DNA Methylation", 
  data.type = "Methylation Beta Value", 
  platform = "Illumina Human Methylation 450"
)
GDCdownload(query = query.met, method = "api", files.per.chunk = 50)
data_DNAm <- GDCprepare(
  query = query.met, 
  save = TRUE, 
  save.filename = paste0('./data/', tcga_project, '/DNAm.rda')
)
```

**技术细节**：
- **数据类型**: Beta值（0-1之间，表示甲基化程度）
- **平台**: Illumina HumanMethylation450 BeadChip（约485,000个CpG位点）
- **文件分块**: 每次下载50个文件，避免网络超时
- **数据格式**: SummarizedExperiment对象，包含甲基化矩阵和样本信息

| 层级      | 字段                     | 示例                            | 说明             |
| ------- | ---------------------- | ----------------------------- | -------------- |
| Patient | `cases.submitter_id`   | `"TCGA-XX-YYYY"`              | 病人 ID，全局唯一     |
| Sample  | `sample.submitter_id`  | `"TCGA-XX-YYYY-01A"`          | 样本 ID，一个病人可能多个 |
| Aliquot | `aliquot.submitter_id` | `"TCGA-XX-YYYY-01A-01R-A12D"` | 最终建库物料 ID（测序）  |


**生物学意义**：
- DNA甲基化是重要的表观遗传修饰
- 与基因表达调控密切相关
- 在癌症发生发展中起关键作用

### 数据类型2：mRNA基因表达

```r
# mRNA Gene Expression
query.exp <- GDCquery(
  project = tcga_project, 
  data.category = "Transcriptome Profiling", 
  data.type = "Gene Expression Quantification", 
  workflow.type = "STAR - Counts"
)
GDCdownload(query.exp, method = "api", files.per.chunk = 50)
data_mRNA <- GDCprepare(
  query = query.exp, 
  save = TRUE, 
  save.filename = paste0('./data/',tcga_project,'/mRNA.rda')
)
```

**技术规格**：
- **数据来源**: RNA-seq测序数据
- **比对工具**: STAR (Spliced Transcripts Alignment to a Reference)
- **定量方法**: 原始计数(raw counts)、TPM、FPKM
- **基因注释**: 基于GENCODE参考基因组

**数据特点**：
- 包含约60,000个转录本的表达量
- 提供多种标准化格式
- 是最常用的基因组数据类型

### 数据类型3：微RNA(miRNA)

```r
# miRNA
query.mirna <- GDCquery(
  project = tcga_project, 
  experimental.strategy = "miRNA-Seq", 
  data.category = "Transcriptome Profiling", 
  data.type = "miRNA Expression Quantification"
)
GDCdownload(query.mirna, method = "api", files.per.chunk = 50)
data_miRNA <- GDCprepare(
  query = query.mirna, 
  save = TRUE, 
  save.filename = paste0('./data/',tcga_project,'/miRNA.rda')
)
```

**miRNA特点**：
- **序列长度**: 约22个核苷酸的短RNA
- **功能**: 转录后基因表达调控
- **数量**: 约2,000个已知的人类miRNA
- **格式**: 包含原始计数和RPM标准化值

**临床意义**：
- miRNA失调与多种癌症相关
- 作为潜在的生物标志物
- 可能的治疗靶点

### 数据类型4：蛋白质组学(RPPA)

```r
# RPPA (Proteome Profiling)
query.rppa <- GDCquery(
  project = tcga_project, 
  data.category = "Proteome Profiling", 
  data.type = "Protein Expression Quantification"
)
GDCdownload(query.rppa, method = "api", files.per.chunk = 50)
data_RPPA <- GDCprepare(
  query = query.rppa, 
  save = TRUE, 
  save.filename = paste0('./data/',tcga_project,'/RPPA.rda')
)
```

**RPPA技术说明**：
- **全称**: Reverse Phase Protein Array（反向蛋白芯片）
- **原理**: 基于抗体的蛋白质定量技术
- **覆盖范围**: 约200个癌症相关蛋白质
- **数据特点**: 已标准化的蛋白质表达水平

**应用价值**：
- 功能性蛋白质表达分析
- 信号通路活性评估
- 药物靶点识别

### 数据类型5：拷贝数变异(CNV)

```r
# CNV
query.cnv <- GDCquery(
  project = tcga_project, 
  data.category = "Copy Number Variation", 
  data.type = "Gene Level Copy Number"
)
GDCdownload(query.cnv, method = "api", files.per.chunk = 50)
data_CNV <- GDCprepare(
  query.cnv, 
  save = TRUE, 
  save.filename = paste0('./data/',tcga_project,'/CNV.rda')
)
```

**CNV数据说明**：
- **定义**: 基因组片段的拷贝数变化
- **检测方法**: 基于SNP芯片或测序数据
- **数据级别**: 基因水平的拷贝数估计
- **数值含义**: 相对于正常二倍体的拷贝数比值

**癌症关联**：
- 肿瘤抑制基因的缺失
- 致癌基因的扩增
- 染色体不稳定性标志

### 数据类型6：简单核苷酸变异(SNV)

```r
# SNV (masked somatic mutation, maftools)
query.snv <- GDCquery(
  project = tcga_project, 
  data.category = "Simple Nucleotide Variation", 
  data.type = "Masked Somatic Mutation", 
  access = "open"
)
GDCdownload(query.snv, method = "api", files.per.chunk = 50)
data_SNV <- GDCprepare(
  query.snv, 
  save = TRUE, 
  save.filename = paste0('./data/',tcga_project,'/SNV.rda')
)
```

**突变数据特点**：
- **数据类型**: 体细胞突变（非胚系突变）
- **格式**: MAF (Mutation Annotation Format)
- **内容**: 点突变、插入缺失(indels)
- **访问级别**: 开放访问（已去识别化），对于 controlled access 的数据，需要申请 access

**分析用途**：
- 驱动基因识别
- 突变负荷分析
- 突变特征(signature)分析
- 肿瘤异质性研究

## 第三部分：数据处理示例（已注释）

脚本末尾包含了详细的数据处理示例代码（已注释），展示了如何：

### mRNA数据处理
```r
# # 4. Process the data (mRNA for example)
# data <- data_mRNA
# assayNames(data)  # 查看可用的数据矩阵类型
# Exp <- assay(data) %>% as.data.frame()  # 提取计数数据
# TPM <- as.data.frame(assay(data, i = "tpm_unstrand"))  # 提取TPM数据
# FPKM <- as.data.frame(assay(data, i = "fpkm_unstrand"))  # 提取FPKM数据
```

### 基因注释处理
```r
# ## 提取基因注释信息
# ann <- rowRanges(data) 
# ann <- as.data.frame(ann)
# rownames(ann) <- ann$gene_id
# ann <- ann[rownames(Exp),]  # 保持与计数数据相同的基因
# ann <- ann [,c(11:12)]  # gene_type, gene_name
```

### 数据合并和保存
```r
# ## 合并基因注释和表达数据
# Exp <- cbind(data.frame(Gene = ann), Exp)
# FPKM <- cbind(data.frame(Gene = ann), FPKM)
# TPM <- cbind(data.frame(Gene = ann), TPM)
```

### 临床数据下载
```r
# ## 处理临床数据
# clinical <- GDCquery_clinic(project= project_id, type = "clinical")
# write.csv(clinical, paste0("./", project_id,"/clinical.csv"), row.names = FALSE)
```

## 脚本设计优势

### 1. 完整性
- **六种主要组学数据**: 覆盖从基因组到蛋白质组的多层次信息
- **标准化流程**: 使用一致的查询和下载模式
- **自动化程度高**: 最小化人工干预

### 2. 可靠性
- **分块下载**: 避免大文件下载失败
- **重复下载保护**: 检查已存在的数据
- **错误处理**: 网络中断自动重试

### 3. 灵活性
- **参数化配置**: 容易切换不同癌症类型
- **模块化设计**: 可以选择性下载特定数据类型
- **扩展性强**: 易于添加新的数据类型

### 4. 标准化
- **统一的数据格式**: 所有数据保存为RDA格式
- **一致的命名规范**: 便于后续分析脚本调用
- **完整的元数据**: 包含样本信息和实验条件

## 使用建议和注意事项

### 1. 环境要求
```r
# 建议的系统配置
# RAM: >= 16GB (处理大型数据集)
# 存储: >= 100GB (完整的TCGA项目数据)
# 网络: 稳定的互联网连接
```

### 2. 下载优化
```r
# 网络优化设置
options(timeout = 300)  # 增加超时时间
options(download.file.method = "libcurl")  # 使用libcurl方法
```

### 3. 错误处理
- **网络中断**: 脚本会自动重试
- **文件损坏**: 重新运行下载命令
- **空间不足**: 检查磁盘空间

### 4. 数据管理
```r
# 检查下载的数据大小
file.info(paste0('./data/', tcga_project, '/'))$size
# 清理临时文件
unlink("GDCdata", recursive = TRUE)
```

## 常见问题和解决方案

### 1. 下载失败
**原因**: 网络不稳定或服务器繁忙
**解决**: 重新运行脚本，已下载的文件会被跳过

### 2. 内存不足
**原因**: 处理大型数据集时内存不够
**解决**: 增加内存限制或分批处理

### 3. 路径错误
**原因**: 脚本中的路径不一致
**解决**: 统一检查路径和创建路径

### 4. 依赖包问题
```r
# 检查和安装必要的包
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("TCGAbiolinks")
BiocManager::install("SummarizedExperiment")
```

## 数据类型总结

| 数据类型 | 文件大小(估计) | 样本数量 | 特征数量 | 主要用途 |
|----------|---------------|----------|----------|----------|
| DNA甲基化 | ~2GB | ~800 | ~485K CpG | 表观遗传分析 |
| mRNA表达 | ~500MB | ~1100 | ~60K 基因 | 基因表达分析 |
| miRNA | ~50MB | ~1000 | ~2K miRNA | 调控网络分析 |
| RPPA蛋白 | ~10MB | ~900 | ~200 蛋白 | 信号通路分析 |
| CNV | ~100MB | ~1100 | ~20K 基因 | 基因组不稳定性 |
| SNV突变 | ~50MB | ~1000 | 变异的 | 驱动突变发现 |

| 层级        | 组学名称            | 典型技术      |
| --------- | --------------- | --------- |
| mRNA 表达   | transcriptomics | RNA-seq   |
| miRNA 表达  | **miRNomics**   | miRNA-seq |
| DNA 甲基化   | epigenomics     | 450k/850k |
| CNV / SNV | genomics        | WES/WGS   |
| 蛋白质表达     | proteomics      | RPPA      |
| 代谢物水平     | metabolomics    | MS/LC     |


## 总结

这个数据下载脚本是TCGA多组学数据分析的重要起点，具有以下特色：

1. **全面性**: 涵盖六种主要的组学数据类型
2. **自动化**: 最小化手动操作，提高效率
3. **标准化**: 统一的下载和存储格式
4. **可扩展**: 易于修改和扩展到其他项目
5. **实用性**: 直接可用的生产级代码

通过这个脚本，研究人员可以快速获得高质量的TCGA数据，为后续的多组学整合分析奠定坚实基础。脚本的设计考虑了实际使用中的各种需求和限制，是一个成熟、可靠的数据下载解决方案。