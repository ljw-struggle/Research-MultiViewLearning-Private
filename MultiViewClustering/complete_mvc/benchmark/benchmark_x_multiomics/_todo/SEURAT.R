rna <- read.table("../Symsim2/seed_1/RNA_batch1.txt", header = FALSE)
pt <- read.table("../Symsim2/seed_1/Protein_batch1.txt", header = FALSE)
atac <- read.table("../Symsim2/seed_1/Atac_batch1.txt", header = FALSE)
library(Seurat)
library(Signac)
obj1 <- CreateSeuratObject(rna)
obj1 <- NormalizeData(obj1) %>% FindVariableFeatures(nfeatures = 300) %>% ScaleData()
obj1 <- RunPCA(obj1, reduction.name = "rpca", npcs = 10)
obj1[["pt"]] <- CreateAssayObject(pt)
DefaultAssay(obj1) <- "pt"
obj1 <- NormalizeData(obj1, normalization.method = "CLR", margin = 2) %>% ScaleData()
obj1 <- FindVariableFeatures(obj1, nfeatures = 30)
obj1 <- RunPCA(obj1, reduction.name = "apca", npcs = 10)
obj1[["atac"]] <- CreateAssayObject(atac)
DefaultAssay(obj1) <- "atac"
obj1 <- Signac::RunTFIDF(obj1)
obj1 <- FindTopFeatures(obj1, min.cutoff = 'q0')
obj1 <- RunSVD(obj1)
obj1 <- ScaleData(obj1, assay = "atac")
obj1 <- mojitoo(object = obj1,      
               reduction.list = list("rpca", "lsi", "apca"),
               dims.list = list(1:10, 2:10, 1:5), ## exclude 1st dimension of LSI
               reduction.name='MOJITOO',
               assay="RNA"
)
tmp <- Seurat::Embeddings(obj1, reduction="MOJITOO")
# obj@reductions$MOJITOO@cell.embeddings
write.table(tmp, row.names = TRUE, col.names = TRUE, "../mojitoo_embd_batch1.tsv", quote = FALSE)

library(MOFA2)
MOFA_run <- function(object, ASSAY_list, name = "skin"){
  library(MOFA2)
  ASSAY_1 <- ASSAY_list[[1]]
  ASSAY_2 <- ASSAY_list[[2]]
  ASSAY_3 <- ASSAY_list[[3]]
  mofa <- create_mofa(object, assays = c(ASSAY_1, ASSAY_2, ASSAY_3))
  model_opts <- get_default_model_options(mofa)
  model_opts$num_factors <- 15
  mofa <- prepare_mofa(mofa, model_options = model_opts)
  mofa <- run_mofa(mofa)
  plot_factor_cor(mofa)
  factors <- 1:get_dimensions(mofa)[["K"]]
  if (name == "pbmc"){
    factors <- factors[!factors%in%c(4, 7)]
  }
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

reticulate::use_condaenv("C:/development\\Miniconda\\envs\\r-reticulate")
mofa_res <- MOFA_run(obj1, list("RNA", "atac", "pt"))
tmp <- Seurat::Embeddings(mofa_res, reduction="MOFA")
write.table(tmp, row.names = TRUE, col.names = TRUE, "../mofa_embd_batch1.tsv", quote = FALSE)
multiome <- FindMultiModalNeighbors(obj1,
                                    reduction.list = list("rpca", "lsi", "apca"),
                                    dims.list = list(1:10, 2:10, 1:5))
keep_wsnn <- multiome@graphs$wsnn
multiome <- FindClusters(multiome, graph.name = "wsnn")
silh_obj <- cluster::silhouette(as.integer(factor(multiome$celltype)), as.dist(1-keep_wsnn))
mean(silh_obj[, 'sil_width'])
keep_wsnn_res <- as.sparse(keep_wsnn)
Matrix::writeMM(keep_wsnn_res, "../seurat_embd_batch1.tsv")