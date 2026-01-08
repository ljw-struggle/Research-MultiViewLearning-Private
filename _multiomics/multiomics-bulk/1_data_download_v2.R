library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
options(future.globals.maxSize = 10 * 1024^3)  # set the memory size
getwd()

# 1. Check available projects
getGDCInfo() # Get GDC information
getGDCprojects() # Get available projects
getGDCprojects()$project_id # Get available projects

# 2. Check information of a specific project
tcga_project <- "TCGA-BRCA"
# getProjectSummary(tcga_project) # Get project summary

# 3. Download TCGA data for a specific project (Check which data types are missing and download only those)
if (dir.exists(paste0('./data/', tcga_project))) {
  cat("TCGA data folder already exists, checking for missing data types...\n")
} else {
  cat("Downloading TCGA data...\n")
  dir.create(paste0('./data/', tcga_project), recursive = TRUE)
}

# DNA Methylation 
if (!file.exists(paste0('./data/', tcga_project, '/DNAm.rda'))) {
  cat("Downloading DNA Methylation data...\n")
  query.met <- GDCquery(project = tcga_project, data.category = "DNA Methylation", data.type = "Methylation Beta Value", platform = "Illumina Human Methylation 450")
  GDCdownload(query = query.met, method = "api", files.per.chunk = 50)
  data_DNAm <- GDCprepare(query = query.met, save = TRUE, save.filename = paste0('./data/', tcga_project, '/DNAm/DNAm.rda'))
} else {
  cat("DNAm.rda already exists, skipping...\n")
}

# mRNA Gene Expression
if (!file.exists(paste0('./data/', tcga_project, '/mRNA.rda'))) {
  cat("Downloading mRNA data...\n")
  query.exp <- GDCquery(project = tcga_project, data.category = "Transcriptome Profiling", data.type = "Gene Expression Quantification", workflow.type = "STAR - Counts")
  GDCdownload(query.exp, method = "api", files.per.chunk = 50)
  data_mRNA <- GDCprepare(query = query.exp, save = TRUE, save.filename = paste0('./data/', tcga_project, '/mRNA/mRNA.rda'))
} else {
  cat("mRNA.rda already exists, skipping...\n")
}

# miRNA
if (!file.exists(paste0('./data/', tcga_project, '/miRNA.rda'))) {
  cat("Downloading miRNA data...\n")
  query.mirna <- GDCquery(project = tcga_project, data.category = "Transcriptome Profiling", data.type = "miRNA Expression Quantification", experimental.strategy = "miRNA-Seq")
  GDCdownload(query.mirna, method = "api", files.per.chunk = 50)
  data_miRNA <- GDCprepare(query = query.mirna, save = TRUE, save.filename = paste0('./data/', tcga_project, '/miRNA/miRNA.rda'))
} else {
  cat("miRNA.rda already exists, skipping...\n")
}

# RPPA (Proteome Profiling)
if (!file.exists(paste0('./data/', tcga_project, '/RPPA.rda'))) {
  cat("Downloading RPPA data...\n")
  query.rppa <- GDCquery(project = tcga_project, data.category = "Proteome Profiling", data.type = "Protein Expression Quantification")
  GDCdownload(query.rppa, method = "api", files.per.chunk = 50)
  data_RPPA <- GDCprepare(query = query.rppa, save = TRUE, save.filename = paste0('./data/', tcga_project, '/RPPA/RPPA.rda'))
} else {
  cat("RPPA.rda already exists, skipping...\n")
}

# CNV - Handle duplicate samples
if (!file.exists(paste0('./data/', tcga_project, '/CNV.rda'))) {
  cat("Downloading CNV data...\n")
  query.cnv <- GDCquery(project = tcga_project, data.category = "Copy Number Variation", data.type = "Gene Level Copy Number")
  
  # Check for duplicate samples by patient ID
  cnv_results <- getResults(query.cnv)
  cat("Query CNV files: ", nrow(cnv_results), "\n")
  cnv_results$patient_id <- substr(cnv_results$cases.submitter_id, 1, 12)
  duplicate_patients <- cnv_results$patient_id[duplicated(cnv_results$patient_id)]
  cat("Duplicate patients found:", length(unique(duplicate_patients)), "\n")
  if (length(duplicate_patients) > 0) {
    cnv_results_filtered <- cnv_results[!duplicated(cnv_results$patient_id), ] # For duplicated patients, keep only the first occurrence
    cat("Filtered CNV files (removed duplicates):", nrow(cnv_results_filtered), "\n")
    # Create new query with filtered files
    query.cnv.filtered <- query.cnv
    query.cnv.filtered$results[[1]] <- cnv_results_filtered
    GDCdownload(query.cnv.filtered, method = "api", files.per.chunk = 50)
    data_CNV <- GDCprepare(query.cnv.filtered, save = TRUE, save.filename = paste0('./data/', tcga_project, '/CNV/CNV.rda'))
  } else {
    cat("No duplicates found, proceeding with all samples...\n")
    GDCdownload(query.cnv, method = "api", files.per.chunk = 50)
    data_CNV <- GDCprepare(query.cnv, save = TRUE, save.filename = paste0('./data/', tcga_project, '/CNV/CNV.rda'))
  }
} else {
  cat("CNV.rda already exists, skipping...\n")
}

# SNV (masked somatic mutation)
if (!file.exists(paste0('./data/', tcga_project, '/SNV.rda'))) {
  cat("Downloading SNV data...\n")
  query.snv <- GDCquery(project = tcga_project, data.category = "Simple Nucleotide Variation", data.type = "Masked Somatic Mutation", access = "open")
  GDCdownload(query.snv, method = "api", files.per.chunk = 50)
  data_SNV <- GDCprepare(query.snv, save = TRUE, save.filename = paste0('./data/', tcga_project, '/SNV/SNV.rda'))
} else {
  cat("SNV.rda already exists, skipping...\n")
}

cat("Data download completed successfully!\n")