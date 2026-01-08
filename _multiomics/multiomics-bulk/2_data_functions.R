# library(MethylPipeR)
library(glmnet)
library(igraph)
library(DESeq2)
library(dplyr)
library(ggplot2)
library(edgeR)
# library(MethylPipeR)
library(WGCNA)

removeTraitNAs <- function(traitDF, otherDFs, trait) {
  rowsToKeep <- !is.na(traitDF[[trait]])
  traitDF <- traitDF[rowsToKeep, ]
  otherDFs <- lapply(otherDFs, function(df) {
    if (is.data.frame(df) || is.matrix(df)) {
      df[rowsToKeep, ]
    } else if (is.null(df)) {
      # For example, if foldID is NULL in cvTrait
      df
    } else {
      # Assumes df is a vector
      df[rowsToKeep]
    }
  })
  list(traitDF = traitDF, otherDFs = otherDFs)
}

cvTrait <- function(trainMethyl, trainPhenotypes, trait, nFolds) {
  print(paste0('Removing rows with missing ', trait, ' from training data.'))
  trainRemoveNAResult <- removeTraitNAs(trainPhenotypes, list(trainMethyl = trainMethyl), trait)
  trainPhenotypes <- trainRemoveNAResult$traitDF
  trainMethyl <- trainRemoveNAResult$otherDFs$trainMethyl
  
  print('Fitting lasso model')
  methylModel <- cv.glmnet(x = trainMethyl,
                           y = as.factor(trainPhenotypes[[trait]]),
                           seed = 42,
                           family = 'multinomial',
                           type.measure = "class",
                           alpha = 1,
                           nFolds = nFolds,
                           parallel = TRUE,
                           trace.it = 1)
  print(methylModel)
  list(trait = trait, model = methylModel)
}

diff_expr <- function(count_mtx , datMeta , trait , n_genes , modality) {
  
  # Remove Genes with low level of expression -------------------------------
  to_keep = rowSums(count_mtx) > 0 #removed 1157 genes
  print(paste0('Removing ',length(to_keep) - sum(to_keep),' Genes with all 0'))
  
  count_mtx <- count_mtx[to_keep,]
  datExpr <- count_mtx
  
  if (modality != 'miRNA') {
    to_keep = filterByExpr(datExpr , group = datMeta[[trait]])
    
    print(paste0('keeping ',sum(to_keep) ,' genes'))
    print(paste0("Removing ",length(to_keep) - sum(to_keep)," Genes"))
    
    count_mtx = count_mtx[to_keep,]
  }
  
  # Remove Outliers ---------------------------------------------------------
  print('removing outliers')
  absadj = count_mtx %>% bicor %>% abs
  netsummary = fundamentalNetworkConcepts(absadj)
  ku = netsummary$Connectivity
  z.ku = (ku-mean(ku))/sqrt(var(ku))

  to_keep = z.ku > -2
  print(paste0("Keeping ",sum(to_keep)," Samples"))
  print(paste0("Removed ",length(to_keep) - sum(to_keep), " Samples"))

  count_mtx <- count_mtx[,to_keep] #removed 36
  datMeta <- datMeta[to_keep,]
  
  # Normalisation Using DESeq -----------------------------------------------
  plot_data = data.frame('ID'=rownames(count_mtx), 'Mean'=rowMeans(count_mtx), 'SD'=apply(count_mtx,1,sd))
  
  plot_data %>% ggplot(aes(Mean, SD)) + geom_point(color='#0099cc', alpha=0.1) + geom_abline(color='black') +
    scale_x_log10() + scale_y_log10() + theme_minimal()  + theme(plot.title = element_text(hjust = 0.5)) 
  
  datMeta[[trait]] <- as.factor(datMeta[[trait]])
  dds = DESeqDataSetFromMatrix(countData = count_mtx, colData = datMeta , design = formula(paste("~ 0 +",trait)))
  
  print('performing DESeq')
  dds = DESeq(dds)
  
  # DEA Plots ---------------------------------------------------------------
  DE_info = results(dds)
  DESeq2::plotMA(DE_info, main= 'Original LFC values')
  
  # VST Transformation of Data ----------------------------------------------
  nsub_check = sum( rowMeans( counts(dds, normalized=TRUE)) > 5 )
  if (nsub_check < 1000) {
    vsd = vst(dds , nsub= nsub_check)
  } else {
    vsd = vst(dds)
  }
  
  datExpr_vst = assay(vsd)
  datMeta_vst = colData(vsd)
  
  meanSdPlot(datExpr_vst, plot=FALSE)$gg + theme_minimal() + ylim(c(0,2))
  
  plot_data = data.frame('ID'=rownames(datExpr_vst), 'Mean'=rowMeans(datExpr_vst), 'SD'=apply(datExpr_vst,1,sd))
  
  plot_data %>% ggplot(aes(Mean, SD)) + geom_point(color='#0099cc', alpha=0.2) + geom_smooth(color = 'gray') +
    scale_x_log10() + scale_y_log10() + theme_minimal()
  
  subtypes <- levels(as.factor(datMeta[[trait]]))
  top_genes = c()
  for (subtype1 in subtypes[1:length(subtypes)-1]) {
    subtypes = subtypes[subtypes != subtype1]
    for (subtype2 in subtypes)  {
      if (subtype1 != subtype2) {
        res <- results(dds , contrast = c(trait , subtype1 , subtype2))
      }
      top_genes = unique(c(top_genes , head(order(res$padj) , n_genes) ))
    }
  }
  
  list(dds = dds , datExpr = datExpr_vst, datMeta = datMeta_vst , top_genes = top_genes)
}

