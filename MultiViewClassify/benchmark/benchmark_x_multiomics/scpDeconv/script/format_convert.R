library(Seurat)
library(SeuratDisk)

## Converting from AnnData to Seurat via h5Seurat
## somtimes, error will occurs, delete some obs columns, only batch and celltype is useful
setwd("/aaa/fionafyang/project1/gelseywang/scBERT_revise/Data/human_15organ")
Convert("human_15organ_subset_normed.h5ad", dest = "h5seurat", overwrite = FALSE)
test_data <- LoadH5Seurat("Data/test.hidden.h5seurat")

## Converting from Seurat to AnnData via h5Seurat
SaveH5Seurat(pbmc3k.final, filename = "pbmc3k.h5Seurat")
Convert("pbmc3k.h5Seurat", dest = "h5ad")

## Converting from Seurat to SingleCellExperiment
library(scater)
pbmc.sce <- as.SingleCellExperiment(pbmc)

## Converting from SingleCellExperiment to Seurat
manno <- readRDS(file = "../data/manno_human.rds")
manno.seurat <- as.Seurat(manno, counts = "counts", data = "logcounts")
Idents(manno.seurat) <- "cell_type1"
