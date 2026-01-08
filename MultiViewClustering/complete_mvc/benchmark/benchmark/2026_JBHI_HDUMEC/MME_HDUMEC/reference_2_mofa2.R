library(Seurat)
library(Signac)
library(MOFA2) # need python==3.6, seurat==4.4.0
library(magrittr)
library(reticulate)


MOFA_run <- function(object, ASSAY_list){
  library(MOFA2)
  reticulate::use_condaenv("mofapy2_py36")
  ASSAY_1 <- ASSAY_list[[1]]
  ASSAY_2 <- ASSAY_list[[2]]
  ASSAY_3 <- ASSAY_list[[3]]
  # Manually extract data and limit features to avoid memory issues
  # MOFA expects features x samples (rows = features, cols = samples)
  data_list <- list()
  for (assay_name in c(ASSAY_1, ASSAY_2, ASSAY_3)) {
    # Get variable features or all features
    var_features <- tryCatch({
      VariableFeatures(object, assay = assay_name)
    }, error = function(e) {
      NULL
    })
    # Get count data (use slot for Seurat v4)
    count_data <- tryCatch({
      GetAssayData(object, assay = assay_name, slot = "counts")
    }, error = function(e) {
      tryCatch({
        GetAssayData(object, assay = assay_name, slot = "data")
      }, error = function(e2) {
        # Fallback to layer for Seurat v5 compatibility
        tryCatch({
          GetAssayData(object, assay = assay_name, layer = "counts")
        }, error = function(e3) {
          GetAssayData(object, assay = assay_name, layer = "data")
        })
      })
    })
    # If variable features exist, use them; otherwise use all
    if (!is.null(var_features) && length(var_features) > 0 && all(var_features %in% rownames(count_data))) {
      count_data <- count_data[var_features, , drop = FALSE]
    }
    # Limit feature number to avoid memory issues
    # For ATAC, limit to top 500 features if more than that
    if (assay_name == "atac" && nrow(count_data) > 100) {
      feature_vars <- apply(count_data, 1, var)
      top_features <- names(sort(feature_vars, decreasing = TRUE))[1:100]
      count_data <- count_data[top_features, , drop = FALSE]
      cat(sprintf("  Limited %s to top 300 features (from %d)\n", assay_name, length(feature_vars)))
    }
    # For RNA, limit to top 200 if more than that
    if (assay_name == "RNA" && nrow(count_data) > 100) {
      feature_vars <- apply(count_data, 1, var)
      top_features <- names(sort(feature_vars, decreasing = TRUE))[1:100]
      count_data <- count_data[top_features, , drop = FALSE]
      cat(sprintf("  Limited %s to top 200 features (from %d)\n", assay_name, length(feature_vars)))
    }
    data_list[[assay_name]] <- as.matrix(count_data)
  }
  # Print data dimensions
  cat("Data dimensions:\n")
  for (assay_name in names(data_list)) {
    cat(sprintf("  %s: %d features x %d samples\n", assay_name, nrow(data_list[[assay_name]]), ncol(data_list[[assay_name]])))
  }
  # Create MOFA object from data list
  mofa <- create_mofa(data_list, use_basilisk = FALSE)
  model_opts <- get_default_model_options(mofa)
  # Reduce factors based on sample size
  n_samples <- min(sapply(data_list, ncol))
  model_opts$num_factors <- min(15, max(5, floor(n_samples / 50)))  # 5-15 factors
  cat(sprintf("Using %d factors (samples: %d)\n", model_opts$num_factors, n_samples))
  mofa <- prepare_mofa(mofa, model_options = model_opts)
  mofa <- run_mofa(mofa)
  # plot_factor_cor(mofa)
  factors <- 1:get_dimensions(mofa)[["K"]]
  mofa <- run_umap(mofa, factors = factors, n_neighbors = 15, min_dist = 0.30)
  mofaUMAP <- mofa@dim_red$UMAP
  rownames(mofaUMAP) <- paste0(mofaUMAP$sample)
  assertthat::assert_that(all(rownames(mofaUMAP  %in% colnames(object))))
  assertthat::assert_that(all(colnames(object) %in% rownames(mofaUMAP)))
  mofaUMAP$sample <- NULL
  colnames(mofaUMAP) <- paste0("UMAP_", 1:2)
  mofaUMAP <- mofaUMAP[colnames(object), ]
  object[["MOFA_UMAP"]] <- CreateDimReducObject(embeddings=as.matrix(mofaUMAP), key="mofa", assay=ASSAY_1)
  #factors <- 1:get_dimensions(mofa)[["K"]]
  Z <- get_factors(mofa, factors = factors, groups = "all")[[1]]
  Z <- Z[colnames(object), ]
  object[["MOFA"]] <- CreateDimReducObject(embeddings=as.matrix(Z), key="factor", assay=ASSAY_1)
  return(object)
}

for (dataset in c("BRCA", "KIPAN", "LGG", "DOGMA", "TEA", "NEAT")) {
    if (dataset %in% c("BRCA", "KIPAN", "LGG")) {
        data_dir <- "./data/data_bulk_multiomics/"
        rna_csv <- read.table(paste0("./data/data_bulk_multiomics/", dataset, "/modality_mrna.csv"), header = TRUE, sep = ",")
        pt_csv <- read.table(paste0("./data/data_bulk_multiomics/", dataset, "/modality_mirna.csv"), header = TRUE, sep = ",")
        atac_csv <- read.table(paste0("./data/data_bulk_multiomics/", dataset, "/modality_meth.csv"), header = TRUE, sep = ",")
        label <- rna_csv[, 1] # first column is label, remove it from data
        rna_data <- as.matrix(rna_csv[, -1])
        rownames(rna_data) <- paste0("sample-", 1:nrow(rna_data)) # shape: (n_cells, n_features)
        pt_data <- as.matrix(pt_csv[, -1])
        rownames(pt_data) <- paste0("sample-", 1:nrow(pt_data)) # shape: (n_cells, n_features)
        atac_data <- as.matrix(atac_csv[, -1])
        rownames(atac_data) <- paste0("sample-", 1:nrow(atac_data)) # shape: (n_cells, n_features)
        rna <- t(rna_data)  # transpose: (n_features, n_cells) for Seurat
        pt <- t(pt_data)    # transpose: (n_features, n_cells) for Seurat
        atac <- t(atac_data) # transpose: (n_features, n_cells) for Seurat
        obj <- CreateSeuratObject(rna)
        obj <- NormalizeData(obj) %>% FindVariableFeatures(nfeatures = 300) %>% ScaleData()
        obj <- RunPCA(obj, reduction.name = "rpca", npcs = 10)
        obj[["pt"]] <- CreateAssayObject(pt)
        DefaultAssay(obj) <- "pt"
        obj <- NormalizeData(obj, normalization.method = "CLR", margin = 2) %>%  FindVariableFeatures(nfeatures = 30) %>% ScaleData()
        obj <- RunPCA(obj, reduction.name = "apca", npcs = 10)
        obj[["atac"]] <- CreateAssayObject(atac)
        DefaultAssay(obj) <- "atac"
        obj <- Signac::RunTFIDF(obj) %>% FindTopFeatures(min.cutoff = 'q0') %>% RunSVD() %>% ScaleData()
        mofa_res <- MOFA_run(obj, list("RNA", "atac", "pt"))
        mofa_embedding <- Seurat::Embeddings(mofa_res, reduction="MOFA")
        embedding_df <- as.data.frame(mofa_embedding)
        embedding_df$label <- label[as.integer(gsub("sample-", "", rownames(mofa_embedding)))]
        embedding_df <- embedding_df[, c("label", setdiff(colnames(embedding_df), "label"))]
        write.table(embedding_df, row.names = FALSE, col.names = TRUE, paste0(data_dir, dataset, "/embedding_mofa2.csv"), quote = FALSE, sep = ",")
    } else if (dataset %in% c("DOGMA", "TEA", "NEAT")) {
        data_dir <- "./data/data_sc_multiomics/"
        rna_csv <- read.table(paste0("./data/data_sc_multiomics/", dataset, "/modality_rna_raw.csv"), header = TRUE, sep = ",")
        pt_csv <- read.table(paste0("./data/data_sc_multiomics/", dataset, "/modality_protein_raw.csv"), header = TRUE, sep = ",")
        atac_csv <- read.table(paste0("./data/data_sc_multiomics/", dataset, "/modality_atac_raw.csv"), header = TRUE, sep = ",")
        label <- rna_csv[, 1] # first column is label, remove it from data
        rna_data <- as.matrix(rna_csv[, -1])
        rownames(rna_data) <- paste0("sample-", 1:nrow(rna_data)) # shape: (n_cells, n_features)
        pt_data <- as.matrix(pt_csv[, -1])
        rownames(pt_data) <- paste0("sample-", 1:nrow(pt_data)) # shape: (n_cells, n_features)
        atac_data <- as.matrix(atac_csv[, -1])
        rownames(atac_data) <- paste0("sample-", 1:nrow(atac_data)) # shape: (n_cells, n_features)
        rna <- t(rna_data)  # transpose: (n_features, n_cells) for Seurat
        pt <- t(pt_data)    # transpose: (n_features, n_cells) for Seurat
        atac <- t(atac_data) # transpose: (n_features, n_cells) for Seurat
        obj <- CreateSeuratObject(rna)
        obj <- NormalizeData(obj) %>% FindVariableFeatures(nfeatures = 300) %>% ScaleData()
        obj <- RunPCA(obj, reduction.name = "rpca", npcs = 10)
        obj[["pt"]] <- CreateAssayObject(pt)
        DefaultAssay(obj) <- "pt"
        obj <- NormalizeData(obj, normalization.method = "CLR", margin = 2) %>%  FindVariableFeatures(nfeatures = 30) %>% ScaleData()
        # Calculate PCA with max possible dimensions (min of nfeatures and n_cells-1)
        max_apca_dims <- min(10, ncol(pt) - 1, nrow(pt))
        obj <- RunPCA(obj, reduction.name = "apca", npcs = max_apca_dims)
        obj[["atac"]] <- CreateAssayObject(atac)
        DefaultAssay(obj) <- "atac"
        obj <- Signac::RunTFIDF(obj) %>% FindTopFeatures(min.cutoff = 'q0') %>% RunSVD() %>% ScaleData()
        # Get actual dimensions computed for each reduction
        rpca_dims <- min(10, ncol(obj@reductions$rpca))
        apca_dims <- min(10, ncol(obj@reductions$apca))
        lsi_dims <- min(10, ncol(obj@reductions$lsi))
        # Ensure at least 2 dimensions for LSI (since we exclude the first)
        # If lsi_dims < 2, we can't exclude the first dimension, so use all available
        if (lsi_dims < 2) {
            warning(paste0("LSI has only ", lsi_dims, " dimensions. Cannot exclude first dimension. Using all available dimensions."))
            lsi_dims_list <- 1:lsi_dims
        } else {
            lsi_dims_list <- 2:lsi_dims
        }
        mofa_res <- MOFA_run(obj, list("RNA", "atac", "pt"))
        mofa_embedding <- Seurat::Embeddings(mofa_res, reduction="MOFA")
        embedding_df <- as.data.frame(mofa_embedding)
        embedding_df$label <- label[as.integer(gsub("sample-", "", rownames(mofa_embedding)))]
        embedding_df <- embedding_df[, c("label", setdiff(colnames(embedding_df), "label"))]
        write.table(embedding_df, row.names = FALSE, col.names = TRUE, paste0(data_dir, dataset, "/embedding_mofa2.csv"), quote = FALSE, sep = ",")
    }
}
