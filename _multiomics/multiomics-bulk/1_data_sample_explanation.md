# TCGA样本重复检测详解

## 问题背景
在下载TCGA CNV数据时遇到错误：
```
Error: There are samples duplicated. We will not be able to prepare it
Warning: There are more than one file for the same case.
```

## TCGA样本条形码结构

TCGA使用标准化的样本条形码（Sample Barcode）系统，结构如下：

```
TCGA-XX-XXXX-XXX-XXX-XXXX-XX
 |   |   |     |   |   |    |
 |   |   |     |   |   |    |- 分析批次 (Analyte Replicate)
 |   |   |     |   |   |- 分析物类型 (Analyte Type)
 |   |   |     |   |- 小管号 (Vial)
 |   |   |     |- 样本类型 (Sample Type)
 |   |   |- 参与者 (Participant)
 |   |- 组织来源站点 (Tissue Source Site)
 |- 项目代码 (Project)
```

### 关键字段解释

#### 1. **Patient ID (前12位)**
```
TCGA-XX-XXXX
```
- 唯一标识一个患者
- 用于跨数据类型匹配同一患者的不同样本

#### 2. **Sample Type (第14-15位)**
```
01 - Primary Solid Tumor (原发肿瘤)
02 - Recurrent Solid Tumor (复发肿瘤)  
03 - Primary Blood Derived Cancer - Peripheral Blood
06 - Metastatic (转移性肿瘤)
10 - Blood Derived Normal (正常血液)
11 - Solid Tissue Normal (正常组织)
```

#### 3. **完整示例**
```
TCGA-A2-A04P-01A-11R-A089-07
 |------------|  |  |  |    |
 Patient ID   |  |  |  |    分析批次
             样本|  |  |
             类型|  |  分析物类型  
                |  小管号
                分析物
```

## 重复样本的定义和类型

### 1. **真正的重复 (True Duplicates)**
- **定义**: 同一患者、同一样本类型的多个文件
- **原因**: 
  - 技术重复 (Technical Replicates)
  - 不同批次处理的同一样本
  - 数据重新分析产生的多个版本

**示例**:
```
TCGA-A2-A04P-01A-11R-A089-07  # 同一患者的原发肿瘤
TCGA-A2-A04P-01A-11R-A089-07  # 相同条形码的重复文件
```

### 2. **生物学重复 (Biological Replicates)**
- **定义**: 同一患者的不同样本类型
- **特点**: 这些通常不被认为是"重复"，而是有意义的多样本

**示例**:
```
TCGA-A2-A04P-01A-11R-A089-07  # 原发肿瘤
TCGA-A2-A04P-11A-11R-A089-07  # 正常组织
```

### 3. **技术重复 (Technical Replicates)**
- **定义**: 同一样本的多次技术处理
- **区别**: Vial、Analyte或Analyte Replicate不同

**示例**:
```
TCGA-A2-A04P-01A-11R-A089-07  # Vial 11
TCGA-A2-A04P-01A-21R-A089-07  # Vial 21
```

## GDCprepare的重复检测逻辑

### 1. **检测方法**
```r
# GDCprepare内部检测逻辑 (简化版)
check_duplicates <- function(query_results) {
  # 提取样本标识符
  sample_ids <- query_results$cases.submitter_id
  
  # 检查是否有重复的样本ID
  if (any(duplicated(sample_ids))) {
    stop("There are samples duplicated. We will not be able to prepare it")
  }
}
```

### 2. **检测标准**
- **主要依据**: `cases.submitter_id` 字段
- **检测粒度**: 通常基于患者ID + 样本类型
- **严格程度**: 即使是技术重复也可能被标记为重复

### 3. **为什么会出现重复**

#### CNV数据的特殊性
1. **多平台数据**: SNP array + WES/WGS
2. **不同算法**: 多种拷贝数推断算法
3. **数据更新**: TCGA数据的迭代更新

#### 常见重复场景
```r
# 示例：CNV查询结果可能包含
cases.submitter_id
TCGA-A2-A04P-01  # 同一患者原发肿瘤
TCGA-A2-A04P-01  # 重复的同一样本
TCGA-B1-A123-01  
TCGA-B1-A123-01  # 另一个重复样本
```

## 解决方案策略

### 1. **简单去重 (First Occurrence)**
```r
# 保留第一次出现的样本
cnv_results$patient_id <- substr(cnv_results$cases.submitter_id, 1, 12)
cnv_results_filtered <- cnv_results[!duplicated(cnv_results$patient_id), ]
```

### 2. **样本类型过滤**
```r
# 只保留原发肿瘤样本 (Sample Type = 01)
sample_type <- substr(cnv_results$cases.submitter_id, 14, 15)
cnv_results_filtered <- cnv_results[sample_type == "01", ]
```

### 3. **质量优先过滤**
```r
# 根据数据质量指标选择最佳样本
cnv_results_filtered <- cnv_results %>%
  group_by(patient_id) %>%
  slice_max(order_by = quality_score, n = 1) %>%
  ungroup()
```

### 4. **手动检查过滤**
```r
# 检查重复样本的具体情况
duplicated_samples <- cnv_results[duplicated(cnv_results$patient_id) | 
                                  duplicated(cnv_results$patient_id, fromLast = TRUE), ]
View(duplicated_samples)  # 手动检查
```

## 不同数据类型的重复情况

| 数据类型 | 重复频率 | 主要原因 | 推荐策略 |
|----------|----------|----------|----------|
| **mRNA** | 低 | 技术重复 | 简单去重 |
| **miRNA** | 低 | 批次差异 | 样本类型过滤 |
| **DNAm** | 中等 | 平台升级 | 质量优先 |
| **RPPA** | 低 | 蛋白芯片批次 | 简单去重 |
| **CNV** | **高** | 多算法/平台 | **综合策略** |
| **SNV** | 中等 | 流程更新 | 版本选择 |

## 最佳实践建议

### 1. **分析前诊断**
```r
# 检查重复样本分布
query_results <- getResults(query)
patient_counts <- table(substr(query_results$cases.submitter_id, 1, 12))
duplicated_patients <- names(patient_counts)[patient_counts > 1]
cat("Duplicated patients:", length(duplicated_patients), "\n")
```

### 2. **选择性下载**
```r
# 针对特定研究需求调整策略
if (study_focus == "primary_tumor") {
  # 只下载原发肿瘤
  sample_filter <- "01"
} else if (study_focus == "tumor_normal_pairs") {
  # 下载配对样本
  sample_filter <- c("01", "11")
}
```

### 3. **记录处理过程**
```r
# 详细记录去重过程
cat("Original samples:", nrow(cnv_results), "\n")
cat("Duplicated samples:", sum(duplicated(cnv_results$patient_id)), "\n")
cat("Final samples:", nrow(cnv_results_filtered), "\n")
cat("Removed samples:", nrow(cnv_results) - nrow(cnv_results_filtered), "\n")
```

### 4. **验证结果一致性**
```r
# 确保去重后样本在不同数据类型间一致
load("mRNA.rda"); mRNA_patients <- substr(colnames(assay(data)), 1, 12)
load("CNV.rda"); CNV_patients <- substr(rownames(assay(data)), 1, 12)
common_patients <- intersect(mRNA_patients, CNV_patients)
cat("Common patients across data types:", length(common_patients), "\n")
```

## 总结

TCGA样本重复主要源于：
1. **数据获取**: 同一样本的多次处理或测序
2. **平台差异**: 不同技术平台产生的数据
3. **版本更新**: TCGA数据库的持续更新
4. **质量控制**: 不同质量标准下的数据保留

**重复检测依据**:
- 主要基于患者ID (前12位条形码)
- 考虑样本类型 (第14-15位)
- GDCprepare的内置检测机制

**解决原则**:
1. **保持一致性**: 所有数据类型使用相同的样本集
2. **优先质量**: 选择质量最高的样本
3. **记录过程**: 详细记录去重决策
4. **验证结果**: 确保下游分析的可靠性

这种重复问题在大规模基因组数据中很常见，正确的处理策略对确保分析结果的可靠性至关重要。 