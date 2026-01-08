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

# 3. Download TCGA data for a specific project
if (dir.exists(paste0('./data/', tcga_project))) {
  cat("TCGA data folder already exists, skipping download...\n")
} else {
  cat("Downloading TCGA data...\n")
  dir.create(paste0('./data/', tcga_project), recursive = TRUE)

  # DNA Methylation 
  # query.met <- GDCquery(project = tcga_project, data.category = "DNA Methylation", data.type = "Methylation Beta Value", platform = "Illumina Human Methylation 450")
  # # getResults(query.met) # get the query results under the upper filter options (e.g., data.category, data.type, platform, etc.)
  # GDCdownload(query = query.met, method = "api", files.per.chunk = 50) # download the query results
  # data_DNAm <- GDCprepare(query = query.met, save = TRUE, save.filename = paste0('./data/', tcga_project, '/DNAm/DNAm.rda')) # For default, include the clinical data and other metadata
  # # mRNA Gene Expression
  # query.exp <- GDCquery(project = tcga_project, data.category = "Transcriptome Profiling", data.type = "Gene Expression Quantification", workflow.type = "STAR - Counts")
  # # getResults(query.met) # get the query results under the upper filter options (e.g., data.category, data.type, platform, etc.)
  # GDCdownload(query.exp, method = "api", files.per.chunk = 50)
  # data_mRNA <- GDCprepare(query = query.exp, save = TRUE, save.filename = paste0('./data/',tcga_project,'/mRNA/mRNA.rda'))
  # # miRNA
  # query.mirna <- GDCquery(project = tcga_project, experimental.strategy = "miRNA-Seq", data.category = "Transcriptome Profiling", data.type = "miRNA Expression Quantification")
  # # getResults(query.met) # get the query results under the upper filter options (e.g., data.category, data.type, platform, etc.)
  # GDCdownload(query.mirna, method = "api", files.per.chunk = 50)
  # data_miRNA <- GDCprepare(query = query.mirna, save = TRUE, save.filename = paste0('./data/',tcga_project,'/miRNA/miRNA.rda'))
  # # RPPA (Proteome Profiling)
  # query.rppa <- GDCquery(project = tcga_project, data.category = "Proteome Profiling", data.type = "Protein Expression Quantification")
  # # getResults(query.met) # get the query results under the upper filter options (e.g., data.category, data.type, platform, etc.)
  # GDCdownload(query.rppa, method = "api", files.per.chunk = 50)
  # data_RPPA <- GDCprepare(query = query.rppa, save = TRUE, save.filename = paste0('./data/',tcga_project,'/RPPA/RPPA.rda'))
  # CNV
  query.cnv <- GDCquery(project = tcga_project, data.category = "Copy Number Variation", data.type = "Gene Level Copy Number")
  # getResults(query.met) # get the query results under the upper filter options (e.g., data.category, data.type, platform, etc.)
  GDCdownload(query.cnv, method = "api", files.per.chunk = 50)
  data_CNV <- GDCprepare(query.cnv, save = TRUE, save.filename = paste0('./data/',tcga_project,'/CNV/CNV.rda'))

  # SNV (masked somatic mutation, maftools)
  query.snv <- GDCquery(project = tcga_project, data.category = "Simple Nucleotide Variation", data.type = "Masked Somatic Mutation", access = "open")
  # getResults(query.met) # get the query results under the upper filter options (e.g., data.category, data.type, platform, etc.)
  GDCdownload(query.snv, method = "api", files.per.chunk = 50)
  data_SNV <- GDCprepare(query.snv, save = TRUE, save.filename = paste0('./data/',tcga_project,'/SNV/SNV.rda'))
}

# # 4. Process the data (mRNA for example)
# data <- data_mRNA
# assayNames(data)  # Get available matrices, e.g., "counts", "logcounts", "normcounts"
# Exp <- assay(data) %>% as.data.frame() # Extract counts data
# TPM <- as.data.frame(assay(data, i = "tpm_unstrand")) # Extract TPM data
# FPKM <- as.data.frame(assay(data, i = "fpkm_unstrand")) # Extract FPKM data

# ## Extract gene annotation information
# ann <- rowRanges(data) 
# ann <- as.data.frame(ann)
# rownames(ann) <- ann$gene_id
# ann <- ann[rownames(Exp),] # Keep the same genes as in the counts data
# fwrite(ann, paste0("./", project_id,"/ann.csv"), row.names = TRUE)# Gene annotation information
# ann <- ann [,c(11:12)] # gene_type, gene_name

# ## Merge gene annotation and counts data
# Exp <- cbind(data.frame(Gene = ann), Exp)
# fwrite(Exp, paste0("./", project_id,"/Counts.csv"), row.names = TRUE)

# # Merge gene annotation and FPKM data
# FPKM <- cbind(data.frame(Gene = ann), FPKM )
# fwrite(FPKM, paste0("./", project_id,"/FPKM.csv"), row.names = TRUE)

# ## Merge gene annotation and TPM data
# TPM <- cbind(data.frame(Gene = ann), TPM )
# fwrite(TPM, paste0("./", project_id,"/TPM.csv"), row.names = TRUE)

# ## Process the data (clinical)
# clinical <- GDCquery_clinic(project= project_id, type = "clinical") # Extract clinical information
# write.csv(clinical, paste0("./", project_id,"/clinical.csv"), row.names = FALSE) # Clinical annotation information
