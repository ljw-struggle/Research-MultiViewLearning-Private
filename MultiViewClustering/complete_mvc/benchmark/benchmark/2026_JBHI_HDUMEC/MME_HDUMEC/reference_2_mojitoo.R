library(Seurat)
library(Signac)
library(MOJITOO)
library(magrittr)

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
        obj <- mojitoo(object = obj, reduction.list = list("rpca", "apca", "lsi"), dims.list = list(1:10, 1:10, 2:10), reduction.name='MOJITOO', assay="RNA") ## exclude 1st dimension of LSI because it correlates with sequencing depth (technical variation) rather than biological variation
        embedding_mojitoo <- Seurat::Embeddings(obj, reduction="MOJITOO")
        # obj@reductions$MOJITOO@cell.embeddings
        embedding_df <- as.data.frame(embedding_mojitoo)
        embedding_df$label <- label[as.integer(gsub("sample-", "", rownames(embedding_mojitoo)))]
        embedding_df <- embedding_df[, c("label", setdiff(colnames(embedding_df), "label"))]
        write.table(embedding_df, row.names = FALSE, col.names = TRUE, paste0(data_dir, dataset, "/embedding_mojitoo.csv"), quote = FALSE, sep = ",")
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
        obj <- mojitoo(object = obj, reduction.list = list("rpca", "apca", "lsi"), dims.list = list(1:rpca_dims, 1:apca_dims, lsi_dims_list), reduction.name='MOJITOO', assay="RNA") ## exclude 1st dimension of LSI because it correlates with sequencing depth (technical variation) rather than biological variation
        embedding_mojitoo <- Seurat::Embeddings(obj, reduction="MOJITOO")
        # obj@reductions$MOJITOO@cell.embeddings
        embedding_df <- as.data.frame(embedding_mojitoo)
        embedding_df$label <- label[as.integer(gsub("sample-", "", rownames(embedding_mojitoo)))]
        embedding_df <- embedding_df[, c("label", setdiff(colnames(embedding_df), "label"))]
        write.table(embedding_df, row.names = FALSE, col.names = TRUE, paste0(data_dir, dataset, "/embedding_mojitoo.csv"), quote = FALSE, sep = ",")
    }
}