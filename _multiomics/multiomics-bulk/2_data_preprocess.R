library(reshape2)
library(kableExtra)
library(plotly)
library(vsn)
library(tibble)
library(pheatmap)
library(SummarizedExperiment)
source('./2_data_functions.R')
setwd('~/Bioinfor/Bioinfor-MMBEMB-Private/data')

project <- 'BRCA'
dataset <- 'TCGA'
trait <- 'paper_BRCA_Subtype_PAM50'
  
# -------------------------------------------------------------------------
# META File Generation ----------------------------------------------------
# -------------------------------------------------------------------------
# The meta data is (typically) located in the coldata of the mRNA gene expression experiment. 
load(paste0('./data/TCGA-',project,'/mRNA/mRNA.rda'))
# Create coldata and condition table --------------------------------------
coldata <- colData(data)
datMeta <- as.data.frame(coldata[,c('patient','race' , 'gender' , 'sample_type' , trait)])
datMeta <- datMeta[!(is.na(datMeta[[trait]])) , ]
datMeta <- datMeta[!(duplicated(datMeta[ , c('patient' , trait)])) , ] 
datMeta[[trait]] <- factor(datMeta[[trait]])
rownames(datMeta) <- datMeta$patient
write.csv(datMeta , file = paste0('./data/TCGA-',project,'/datMeta.csv'))

# -------------------------------------------------------------------------
# mRNA pre-processing -----------------------------------------------------
# -------------------------------------------------------------------------
# Pull in Count Matrices --------------------------------------------------
load(paste0('./data/TCGA-',project,'/mRNA/mRNA.rda'))
count_mtx <- assay(data)
colnames(count_mtx) <- substr(colnames(count_mtx) , 1, 12)
count_mtx <- count_mtx[, !(duplicated(colnames(count_mtx)))]

# Pull in Meta File
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Get intersection of count and meta
common_idx <- intersect(colnames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[ , common_idx]
datMeta <- datMeta[common_idx , ]

# Perform differential expression analysis
diff_expr_res <- diff_expr(count_mtx , datMeta , trait , 500 , 'mRNA')

# Save differential expression results (only 500 genes)
datExpr <- diff_expr_res$datExpr
datMeta <- diff_expr_res$datMeta
dds <- diff_expr_res$dds
top_genes <- diff_expr_res$top_genes
save(datExpr, datMeta, dds, top_genes, file=paste0('./data/',dataset,'/',project,'/mRNA_processed.RData'))

# -------------------------------------------------------------------------
# miRNA preprocessing -----------------------------------------------------
# -------------------------------------------------------------------------
 
#Pull in count data
load(paste0('./data/TCGA-',project,'/miRNA/miRNA.rda'))

# Get Count Matrices and filter for reads
read_count <- data.frame(row.names = data$miRNA_ID)
read_per_million <- data.frame(row.names = data$miRNA_ID)
for (i in 2:dim(data)[2]) {
  if (i%%3 == 2) {
    read_count <- cbind(read_count , data[ , i] )
  }
  if (i%%3 == 0) {
    read_per_million <- cbind(read_per_million , data[ , i])
  }
}

colname_read_count <- c()
colname_read_per_million <- c()
for (i in 2:dim(data)[2]) {
  if (i%%3 == 2) {
    colname_read_count <- c(colname_read_count , substr(strsplit(colnames(data)[i] , '_')[[1]][3], 1, 12))
  }
  if (i%%3 == 0) {
    colname_read_per_million <- cbind(colname_read_per_million , substr(strsplit(colnames(data)[i] , '_')[[1]][6], 1, 12))
  }
}
colnames(read_count) <- colname_read_count
colnames(read_per_million) <- colname_read_per_million

# Pull in Meta File
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Get Intersection of ID's
count_mtx <- read_count
count_mtx <- count_mtx[ , !(duplicated(colnames(count_mtx)))] 
common_idx <- intersect(colnames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[,common_idx]
datMeta <- datMeta[common_idx , ]

# Perform differential expression analysis
diff_expr_res <- diff_expr(count_mtx , datMeta , trait , 200 , 'miRNA')

# Save differential expression results (only 200 genes)
datExpr <- diff_expr_res$datExpr
datMeta <- diff_expr_res$datMeta
dds <- diff_expr_res$dds
top_genes <- diff_expr_res$top_genes
save(datExpr, datMeta, dds, top_genes, file=paste0('./data/',dataset,'/',project,'/miRNA_processed.RData'))

# -------------------------------------------------------------------------
# DNAm preprocessing ------------------------------------------------------
# -------------------------------------------------------------------------

# Load CpG Counts ---------------------------------------------------------
load(paste0('./data/TCGA-',project,'/DNAm/DNAm.rda'))
count_mtx <- assay(data)

to_keep = complete.cases(count_mtx) #removed 191928 cpg sites
length(to_keep) - sum(to_keep)

count_mtx <- t(count_mtx[to_keep,])
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)

# Compute the variance across CpG sites
cpg_variances <- colVars(count_mtx)

# Sort the variances in descending order and get the indices
sorted_indices <- order(cpg_variances, decreasing = TRUE)

# Select the top 300000 most variable CpG sites
num_top_cpg <- 200000
top_cpg_indices <- sorted_indices[1:num_top_cpg]

count_mtx <- count_mtx[ , top_cpg_indices]

# Pull in Meta File
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Get Intersection of ID's
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]

# Run glmnet to get CpG's associated with phenotypes of interest ------------
phenotypes <- datMeta[,c('patient' , trait )]
colnames(phenotypes)

traits <- c(trait)

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, traits, nFolds = 10)
})

cpg_sites <- c()
for (res in traitResults) { 
  trait_coefs <- coef(res$model , s = "lambda.min")
  cpg_sites_tmp <- c()
  for (coefs in trait_coefs) {
    class_coefs <- rownames(coefs)[which(coefs != 0)]
    class_coefs <- class_coefs[2:length(class_coefs)]
    cpg_sites_tmp <- unique(c(cpg_sites_tmp ,class_coefs ))
  }
  cpg_sites[[res$trait]] <- cpg_sites_tmp
}

# Save CpG sites and expression data
datExpr <- count_mtx
save(cpg_sites , datExpr , datMeta , file = paste0('./data/',dataset,'/',project,'/DNAm_processed.RData'))

# ----------------------------------------------------------------------------------------
# Protein (RPPA) pre-processing ----------------------------------------------------------
# ----------------------------------------------------------------------------------------
load(paste0('./data/TCGA-',project,'/RPPA/RPPA.rda'))
count_mtx <- t(data[  , 6:ncol(data) ])
colnames(count_mtx) <- data$peptide_target
rownames(count_mtx) <- substr(rownames(count_mtx) , 1,12)

# Load Meta Data -----------------------------------------------------------
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Subset to match ID's ----------------------------------------------------
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]

# Run glmnet R2 regression to identify proteins of interest -------------------------------
count_mtx <- count_mtx[,colSums(is.na(count_mtx))<0.5*nrow(count_mtx)] #remove columns with more than 50% NA

for(i in 1:ncol(count_mtx)){
  count_mtx[is.na(count_mtx[,i]), i] <- mean(count_mtx[,i], na.rm = TRUE)
}

phenotypes <- datMeta[,c('patient' , trait , 'race' , 'gender')]
colnames(phenotypes)

traits <- c(trait)

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx, phenotypes, trait, nFolds = 10)
}) 

protein_sites <- c()
for (res in traitResults) { 
  trait_coefs <- coef(res$model , s = "lambda.min")
  protein_sites_tmp <- c()
  for (coefs in trait_coefs) {
    class_coefs <- rownames(coefs)[which(coefs != 0)]
    class_coefs <- class_coefs[2:length(class_coefs)]
    protein_sites_tmp <- unique(c(protein_sites_tmp ,class_coefs ))
  }
  protein_sites[[res$trait]] <- protein_sites_tmp
}

# Save proteins and expression data
datExpr <- count_mtx
save(protein_sites , datExpr , datMeta , file = paste0('./data/',dataset,'/',project,'/RPPA_processed.RData'))

# -------------------------------------------------------------------------
# CNV pre-processing ------------------------------------------------------
# -------------------------------------------------------------------------

# Read in Count Matrix
load(paste0('./data/TCGA-',project,'/CNV/CNV.rda'))

count_mtx <- t(assay(data))
rownames_mtx <- c()
for (name in strsplit(rownames(count_mtx) , ',')) {
  rownames_mtx <- c(rownames_mtx , substr(name[1] ,1, 12))
}
rownames(count_mtx) <- rownames_mtx
count_mtx <- count_mtx[,colSums(is.na(count_mtx))<0.5*nrow(count_mtx)] #remove columns with more than 50% NA

# Read in Meta Data
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

#Intersect to common IDs
count_mtx <- count_mtx[!(duplicated(rownames(count_mtx))) , ] 
common_idx <- intersect(rownames(count_mtx) , rownames(datMeta))
count_mtx <- count_mtx[common_idx , ]
datMeta <- datMeta[common_idx , ]

# Replace NA's with 0's 
count_mtx[is.na(count_mtx)] <- 0 

# Perform log transform on count matrix to give normal distribution resemblance
count_mtx_log <- log(count_mtx)

# Run glmnet R2 regression to get CNV's of interest
phenotypes <- datMeta[,c('patient' , trait , 'race' , 'gender')]
colnames(phenotypes)

traits <- c(trait)

traitResults <- lapply(traits, function(trait) {
  cvTrait(count_mtx_log, phenotypes, traits, nFolds = 10)
})

cnv_sites <- c()
for (res in traitResults) { 
  trait_coefs <- coef(res$model , s = "lambda.min")
  cnv_sites_tmp <- c()
  for (coefs in trait_coefs) {
    class_coefs <- rownames(coefs)[which(coefs != 0)]
    class_coefs <- class_coefs[2:length(class_coefs)]
    cnv_sites_tmp <- unique(c(cnv_sites_tmp ,class_coefs ))
  }
  cnv_sites[[res$trait]] <- cnv_sites_tmp
}

rm(count_mtx_log)

# Save CNV's and expression
datExpr <- count_mtx
save(cnv_sites , datExpr , datMeta , file = paste0('./data/',dataset,'/',project,'/CNV_processed.RData'))


# -------------------------------------------------------------------------
# SNV pre-processing ------------------------------------------------------
# -------------------------------------------------------------------------

# Read in SNV Data (MAF format)
load(paste0('./data/TCGA-',project,'/SNV/SNV.rda'))

# Read in Meta Data
datMeta <- read.csv(paste0('./data/TCGA-',project,'/datMeta.csv') , row.names = 1)

# Check SNV data structure
print("SNV data structure:")
print(str(data))
print("SNV data dimensions:")
print(dim(data))

# Extract sample IDs from SNV data (Tumor_Sample_Barcode column)
if("Tumor_Sample_Barcode" %in% colnames(data)) {
  snv_samples <- substr(data$Tumor_Sample_Barcode, 1, 12)
  data$patient_id <- snv_samples
} else {
  stop("Tumor_Sample_Barcode column not found in SNV data")
}

# Get intersection of SNV samples and meta data
common_idx <- intersect(unique(data$patient_id), rownames(datMeta))
print(paste0("Common samples between SNV and meta data: ", length(common_idx)))

# Filter SNV data for common samples
data_filtered <- data[data$patient_id %in% common_idx, ]
datMeta_snv <- datMeta[common_idx, ]

print(paste0("Filtered SNV mutations: ", nrow(data_filtered)))
print(paste0("Filtered samples: ", nrow(datMeta_snv)))

# Create mutation matrix (samples x genes)
# First, get all unique genes with mutations
mutated_genes <- unique(data_filtered$Hugo_Symbol)
print(paste0("Total mutated genes: ", length(mutated_genes)))

# Initialize mutation matrix
mutation_matrix <- matrix(0, 
                         nrow = length(common_idx), 
                         ncol = length(mutated_genes),
                         dimnames = list(common_idx, mutated_genes))

# Fill mutation matrix (1 = mutated, 0 = wild-type)
for(i in 1:nrow(data_filtered)) {
  sample_id <- data_filtered$patient_id[i]
  gene <- data_filtered$Hugo_Symbol[i]
  
  if(sample_id %in% rownames(mutation_matrix) && gene %in% colnames(mutation_matrix)) {
    mutation_matrix[sample_id, gene] <- 1
  }
}

print("Mutation matrix created")
print(paste0("Matrix dimensions: ", paste(dim(mutation_matrix), collapse = " x ")))

# Calculate mutation frequencies per gene
gene_mutation_freq <- colSums(mutation_matrix) / nrow(mutation_matrix)
print("Top 20 most frequently mutated genes:")
print(head(sort(gene_mutation_freq, decreasing = TRUE), 20))

# Filter genes by mutation frequency (keep genes mutated in at least 5% of samples)
min_mutation_freq <- 0.05
frequent_genes <- names(gene_mutation_freq)[gene_mutation_freq >= min_mutation_freq]
print(paste0("Genes with mutation frequency >= ", min_mutation_freq*100, "%: ", length(frequent_genes)))

# Create filtered mutation matrix
mutation_matrix_filtered <- mutation_matrix[, frequent_genes, drop = FALSE]

# Calculate mutation burden per sample
mutation_burden <- rowSums(mutation_matrix_filtered)
print("Mutation burden statistics:")
print(summary(mutation_burden))

# Add mutation burden to meta data
datMeta_snv$mutation_burden <- mutation_burden[rownames(datMeta_snv)]

# Identify significantly mutated genes using Fisher's exact test
trait_levels <- levels(as.factor(datMeta_snv[[trait]]))
print(paste0("Trait levels for comparison: ", paste(trait_levels, collapse = ", ")))

significant_genes <- c()
fisher_results <- list()

if(length(trait_levels) >= 2) {
  for(gene in frequent_genes) {
    # Create contingency table for each gene
    gene_mutations <- mutation_matrix_filtered[, gene]
    trait_groups <- datMeta_snv[[trait]]
    
    # For binary comparison (first two levels)
    group1_samples <- trait_groups == trait_levels[1]
    group2_samples <- trait_groups == trait_levels[2]
    
    # Contingency table: [mutated_group1, mutated_group2, wt_group1, wt_group2]
    contingency_table <- matrix(c(
      sum(gene_mutations[group1_samples]),  # mutated in group1
      sum(gene_mutations[group2_samples]),  # mutated in group2
      sum(!gene_mutations[group1_samples]), # wild-type in group1
      sum(!gene_mutations[group2_samples])  # wild-type in group2
    ), nrow = 2, byrow = TRUE)
    
    # Fisher's exact test
    if(all(contingency_table >= 0)) {
      fisher_test <- fisher.test(contingency_table)
      fisher_results[[gene]] <- list(
        p_value = fisher_test$p.value,
        odds_ratio = fisher_test$estimate,
        contingency_table = contingency_table
      )
    }
  }
  
  # Extract p-values and adjust for multiple testing
  p_values <- sapply(fisher_results, function(x) x$p_value)
  p_values_adj <- p.adjust(p_values, method = "fdr")
  
  # Select significantly different genes (FDR < 0.05)
  significant_genes <- names(p_values_adj)[p_values_adj < 0.05 & !is.na(p_values_adj)]
  print(paste0("Significantly differential genes (FDR < 0.05): ", length(significant_genes)))
  
  if(length(significant_genes) > 0) {
    print("Top significant genes:")
    sig_results <- data.frame(
      gene = significant_genes,
      p_value = p_values[significant_genes],
      p_value_adj = p_values_adj[significant_genes],
      mutation_freq = gene_mutation_freq[significant_genes]
    )
    sig_results <- sig_results[order(sig_results$p_value_adj), ]
    print(head(sig_results, 10))
  }
}

# Create final mutation data for analysis
if(length(significant_genes) > 0) {
  # Use significantly different genes
  final_genes <- significant_genes
} else {
  # If no significant genes, use top frequently mutated genes
  top_n_genes <- min(100, length(frequent_genes))
  final_genes <- names(sort(gene_mutation_freq[frequent_genes], decreasing = TRUE)[1:top_n_genes])
  print(paste0("No significant genes found, using top ", top_n_genes, " frequently mutated genes"))
}

# Final mutation matrix
datExpr_snv <- mutation_matrix_filtered[, final_genes, drop = FALSE]

# Ensure sample order matches
common_samples_final <- intersect(rownames(datExpr_snv), rownames(datMeta_snv))
datExpr_snv <- datExpr_snv[common_samples_final, , drop = FALSE]
datMeta_snv <- datMeta_snv[common_samples_final, ]

print("Final SNV data summary:")
print(paste0("Samples: ", nrow(datExpr_snv)))
print(paste0("Genes: ", ncol(datExpr_snv)))
print(paste0("Total mutations: ", sum(datExpr_snv)))

# Create summary statistics
snv_summary <- list(
  total_mutations = nrow(data_filtered),
  total_samples = length(common_idx),
  total_genes = length(mutated_genes),
  frequent_genes = frequent_genes,
  significant_genes = significant_genes,
  final_genes = final_genes,
  mutation_frequencies = gene_mutation_freq[final_genes],
  fisher_results = if(exists("fisher_results")) fisher_results[final_genes] else NULL
)

# Save SNV data
datExpr <- datExpr_snv
datMeta <- datMeta_snv
save(datExpr, datMeta, snv_summary, file = paste0('./data/',dataset,'/',project,'/SNV_processed.RData'))

print("SNV preprocessing completed and saved!")
print(paste0("Saved file: ./data/", dataset, "/", project, "/SNV_processed.RData"))

