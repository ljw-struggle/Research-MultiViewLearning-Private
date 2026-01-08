library(tidyverse)
library(ggplot2)
library(Seurat)
library(SeuratDisk)
library(ggVennDiagram)
library(survival)
library(survminer)
library(dplyr)
library(corrplot)
library(zellkonverter)
library(SingleCellExperiment)
library(patchwork)
library(DescTools)

setwd("D://workbase//MyWork//scDeconvolution")

###### Figure2a ######
data = read.csv("source_data//source_data_fig2a.csv", row.names=NULL)
data = data[data$Methods %in% c("DEEP","Scaden","MuSiC","BayesPrism","DestVI"),]
data %>%
	mutate(Methods = factor(Methods, levels = c("DEEP","Scaden","MuSiC","BayesPrism","DestVI"))) %>%
	mutate(dataset = factor(dataset, levels = c("human_breast_atlas", "mouse_cellline", "human_cellline", "mel2mon_cellcycle", "mon2mel_cellcycle"))) %>%
	ggplot(aes(dataset,value)) +  
	geom_boxplot(aes(fill = Methods), size =0.4) + 
	scale_fill_brewer(palette = "Set1") +  
	guides(fill = guide_legend(title = "Methods")) +  
	labs(x = "Datasets", y = "CCC") +
	theme_bw() +
	theme(panel.grid = element_blank()) +
	geom_vline(xintercept = c(1.5,2.5,3.5,4.5), size = .25, linetype = "dotted") +
	scale_y_continuous(breaks = seq(-0.5,1,0.2))

###### EDFigure1a ######
temp = read.csv("source_data//source_data_EDfig1a.csv", row.names=NULL)
temp = temp[temp$Methods %in% c("DEEP","Scaden","MuSiC","BayesPrism","DestVI"),]
data$value = temp$value
data %>%
	mutate(Methods = factor(Methods, levels = c("DEEP","Scaden","MuSiC","BayesPrism","DestVI"))) %>%
	mutate(dataset = factor(dataset, levels = c("human_breast_atlas", "mouse_cellline", "human_cellline", "mel2mon_cellcycle", "mon2mel_cellcycle"))) %>%
	ggplot(aes(dataset,value)) +  
	geom_boxplot(aes(fill = Methods), size =0.4) + 
	scale_fill_brewer(palette = "Set1") +  
	guides(fill = guide_legend(title = "Methods")) +  
	labs(x = "Datasets", y = "RMSE") +
	theme_bw() +
	theme(panel.grid = element_blank()) +
	geom_vline(xintercept = c(1.5,2.5,3.5,4.5), size = .25, linetype = "dotted")

###### EDFigure1b ######
temp = read.csv("source_data//source_data_EDfig1b.csv", row.names=NULL)
temp = temp[temp$Methods %in% c("DEEP","Scaden","MuSiC","BayesPrism","DestVI"),]
data$value = temp$value
data %>%
	mutate(Methods = factor(Methods, levels = c("DEEP","Scaden","MuSiC","BayesPrism","DestVI"))) %>%
	mutate(dataset = factor(dataset, levels = c("human_breast_atlas", "mouse_cellline", "human_cellline", "mel2mon_cellcycle", "mon2mel_cellcycle"))) %>%
	ggplot(aes(dataset,value)) +  
	geom_boxplot(aes(fill = Methods), size =0.4) + 
	scale_fill_brewer(palette = "Set1") +  
	guides(fill = guide_legend(title = "Methods")) +  
	labs(x = "Datasets", y = "Cor") +
	theme_bw() +
	theme(panel.grid = element_blank()) +
	geom_vline(xintercept = c(1.5,2.5,3.5,4.5), size = .25, linetype = "dotted") +
	scale_y_continuous(breaks = seq(-0.5,1,0.2))

###### Figure2b ######
scrna_path1 = "D://workbase//MyWork//scProteome//Data//scProteome//human breast atlas//Data//Human_Breast_Atlas_scRNA_normed_aligned_individual1.h5ad"
scp_path1 = "D://workbase//MyWork//scProteome//Data//scProteome//human breast atlas//Data//Human_Breast_Atlas_scProteome_normed_aligned_individual1.h5ad"
scp_path3 = "D://workbase//MyWork//scProteome//Data//scProteome//human breast atlas//Data//Human_Breast_Atlas_scProteome_normed_aligned_individual3.h5ad"

data = readH5AD(scrna_path1)
names(assays(data))="counts"
data <- as.Seurat(data, counts = "counts", data = "counts")
all.genes <- rownames(data)
data <- ScaleData(data, features = all.genes)
celltypes = names(table(data@meta.data$cell_type))[1:6]
RNA_celltype = matrix(0, 6, 33)
for(i in 1:6)
{
	data_subset = data@assays$originalexp@counts[,rownames(data@meta.data[which(data@meta.data$cell_type == celltypes[i]),])]
	RNA_celltype[i,] = rowMeans(data_subset)
}
rownames(RNA_celltype) = celltypes
colnames(RNA_celltype) = rownames(data)

data2 = readH5AD(scp_path1)
names(assays(data2))="counts"
data2 <- as.Seurat(data2, counts = "counts", data = "counts")
data2 <- ScaleData(data2, features = all.genes)

data3 = readH5AD(scp_path3)
names(assays(data3))="counts"
data3 <- as.Seurat(data3, counts = "counts", data = "counts")
data3 <- ScaleData(data3, features = all.genes)

# data2 <- RunPCA(data2, features = all.genes)
# data2 <- FindNeighbors(data2, dims = 1:20)
# data2 <- FindClusters(data2, resolution = 0.5)
# data2 <- RunUMAP(data2, dims = 1:20)
# DimPlot(data2, reduction = "umap")

protein_celltype1 = matrix(0, 6, 33)
for(i in 1:6)
{
	data_subset = data2@assays$originalexp@counts[,rownames(data2@meta.data[which(data2@meta.data$cell_type == celltypes[i]),])]
	protein_celltype1[i,] = rowMeans(data_subset)
}
rownames(protein_celltype1) = celltypes
colnames(protein_celltype1) = rownames(data2)

protein_celltype3 = matrix(0, 6, 33)
for(i in 1:6)
{
	data_subset = data3@assays$originalexp@counts[,rownames(data3@meta.data[which(data3@meta.data$cell_type == celltypes[i]),])]
	protein_celltype3[i,] = rowMeans(data_subset)
}
rownames(protein_celltype3) = celltypes
colnames(protein_celltype3) = rownames(data3)

bk <- c(seq(0,6,by=0.01),seq(0,2,by=0.01))
pheatmap::pheatmap(t(rbind(RNA_celltype, protein_celltype1, protein_celltype3)),cluster_cols = F,cluster_rows = F,color = c(colorRampPalette(colors = c("white","#EA7F01"))(length(bk)/2),colorRampPalette(colors = c("#EA7F01","#E41A1C"))(length(bk)/2)),border_color=NA,fontsize=8)

###### Figure2c ######
# plot correlation heatmap between RNA and protein celltypes
cor = matrix(0,6,6)
for(i in 1:6)
{
	for(j in 1:6)
	{
		if(i == j)
			cor[i,j] = cor(RNA_celltype[i,], protein_celltype3[j,])
	}
}
rownames(cor) = colnames(cor) = celltypes
corrplot(cor, method = "pie", col.lim = c(0, 1),col = colorRampPalette(colors = c("blue","white","firebrick"))(10), tl.col = "black", tl.cex = 0.8)

cor = matrix(0,6,6)
for(i in 1:6)
{
	for(j in 1:6)
	{
		if(i == j)
			cor[i,j] = cor(protein_celltype1[i,], protein_celltype3[j,])
	}
}
rownames(cor) = colnames(cor) = celltypes
corrplot(cor, method = "pie", col.lim = c(0, 1),col = colorRampPalette(colors = c("blue","white","firebrick"))(10), tl.col = "black", tl.cex = 0.8)

# # plot correlation heatmap between RNA and protein
# cor = matrix(0,33,33)
# for(i in 1:33)
# {
# 	for(j in 1:33)
# 	{
# 		if(i == j)
# 			cor[i,j] = cor(RNA_celltype[,i], protein_celltype1[,j])
# 	}
# }
# rownames(cor) = colnames(cor) = colnames(RNA_celltype)
# corrplot(cor, method = "pie", col.lim = c(0, 1),col = colorRampPalette(colors = c("blue","white","firebrick"))(10), tl.col = "black", tl.cex = 0.8)

###### Figure2c_1_new ######
gt = read.csv("Result//Human_Breast_Atlas//RNA1_to_protein3//DEEP_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withoutHVPs_0//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:6, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//Human_Breast_Atlas//RNA1_to_protein3//DEEP_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withoutHVPs_0//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:6, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
    theme_bw() +
    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
    labs(x = "Predicted proportions", y = "True proportions", title = "All cell types") + 
    annotate("text",x=0,y=max(df$comp_gt)-0.05,label=score_label,hjust = 0,size=3) +
    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
    scale_x_continuous(limits = c(0,0.6)) +
    scale_y_continuous(limits = c(0,0.6))

###### Figure2c_2_new ######
gt = read.csv("Result//Human_Breast_Atlas//protein1_to_protein3//DEEP_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withoutHVPs_0//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:6, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//Human_Breast_Atlas//protein1_to_protein3//DEEP_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withoutHVPs_0//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:6, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
    theme_bw() +
    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
    labs(x = "Predicted proportions", y = "True proportions", title = "All cell types") + 
    annotate("text",x=0,y=max(df$comp_gt)-0.05,label=score_label,hjust = 0,size=3) +
    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
    scale_x_continuous(limits = c(0,0.6)) +
    scale_y_continuous(limits = c(0,0.6))

###### Figure2e ######
data = read.csv("source_data//source_data_fig2e.csv")
data %>%
	ggplot(aes(x=order, y=value, shape=direction, color=method))+
	geom_point(size=3)+
	scale_colour_manual(values = c("DEEP"="#E41A1C","Scaden"="#377EB8","MuSiC"="#4DAF4A","BayesPrism"="#984EA3","DestVI"="#FF7F00")) +
	theme_bw() +
	theme(panel.grid = element_blank(), axis.title.x = element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
	scale_y_continuous(breaks = seq(0,1,0.2)) +
	labs(y = "CCC")

###### Figure2e_2 ######
data = read.csv("source_data//source_data_fig2e_RMSE.csv")
data %>%
	ggplot(aes(x=order, y=value, shape=direction, color=method))+
	geom_point(size=3)+
	scale_colour_manual(values = c("DEEP"="#E41A1C","Scaden"="#377EB8","MuSiC"="#4DAF4A","BayesPrism"="#984EA3","DestVI"="#FF7F00")) +
	theme_bw() +
	theme(panel.grid = element_blank(), axis.title.x = element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
	scale_y_continuous(breaks = seq(0,0.2,0.02)) +
	labs(y = "RMSE")

###### Figure2f ######
gt = read.csv("Result//mouse_scp//N2_to_nanoPOTS//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize15_feature_withHVPs_500//target_gt_fraction.csv", row.names = 1)
pred = read.csv("Result//mouse_scp//N2_to_nanoPOTS//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize15_feature_withHVPs_500//target_predicted_fraction.csv", row.names = 1)
df = cbind(pred, gt)
colnames(df) = c("C10_pred","SVEC_pred","RAW_pred","C10_gt","SVEC_gt","RAW_gt")

tmp.ccc = CCC(df$C10_gt, df$C10_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$C10_gt, df$C10_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$C10_gt - df$C10_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = C10_pred, y = C10_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "C10") + 
	    annotate("text",x=0,y=max(df$C10_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$C10_gt))) +
	    scale_y_continuous(limits = c(0,max(df$C10_gt)))

tmp.ccc = CCC(df$SVEC_gt, df$SVEC_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$SVEC_gt, df$SVEC_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$SVEC_gt - df$SVEC_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df, mapping = aes(x = SVEC_pred, y = SVEC_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "SVEC") + 
	    annotate("text",x=0,y=max(df$SVEC_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$SVEC_gt))) +
	    scale_y_continuous(limits = c(0,max(df$SVEC_gt)))

tmp.ccc = CCC(df$RAW_gt, df$RAW_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$RAW_gt, df$RAW_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$RAW_gt - df$RAW_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df, mapping = aes(x = RAW_pred, y = RAW_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "RAW") + 
	    annotate("text",x=0,y=max(df$RAW_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$RAW_gt))) +
	    scale_y_continuous(limits = c(0,max(df$RAW_gt)))

p1 + p2 + p3

###### Figure3c ######
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//pSCoPE_Huffman//PDAC//limmaCorrected_normed_prePCA_PDAC_Coverage_PIF50_mrri10.csv", row.names = 1)
genelist1 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//pSCoPE_Leduc//processed_data//t6.csv", row.names = 1)
genelist2 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//SCoPE2_Leduc//prot_proc.csv", row.names = 1)
genelist3 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//T-SCP//Data//T_SCP_HelaCell_normed_proteinlist.csv", row.names = 1)
genelist4 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//plexDIA//Data//plexDIA_PDAC_Melanoma_U937_normed_proteinlist.csv", row.names = 1)
genelist5 = rownames(data)

x <- list(A=genelist3,
          B=genelist1,
          C=genelist2,
          D=genelist4,
          E=genelist5)
ggVennDiagram(x, category.names = c("SCoPE2_Leduc","pSCoPE_Huffman","pSCoPE_Leduc","T-SCP","plexDIA"), set_size=4, lty="longdash", label="count", label_alpha=0, label_size=5, edge_size=0.8) + 
	scale_color_brewer(palette = "Set2") +
	scale_fill_gradient(low="gray100",high = "gray95",guide="none")

###### Figure3d_1 ######
genelist1 = rownames(read.csv("source_data//pSCoPE_Huffman_PDAC+pSCoPE_Leduc+SCoPE2_Leduc_proteinlist.csv", row.names = 1))
genelist2 = rownames(read.csv("source_data//T-SCP+plexDIA_proteinlist.csv", row.names = 1))
shared_proteinlist = intersect(genelist1,genelist2)

scp_path = "D://workbase//MyWork//scProteome//Data//scProteome//T-SCP+plexDIA//merged_scProteomic_minmax.h5ad"
# Convert(scp_path, dest = "h5seurat", overwrite = TRUE)
data <- LoadH5Seurat(paste0(unlist(strsplit(scp_path, split=".h5ad"))[1], ".h5seurat"))
all.genes <- rownames(data)
data <- ScaleData(data, features = all.genes)
data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = 400)
HVP = VariableFeatures(data)

x <- list(A=shared_proteinlist,
          B=HVP)
ggVennDiagram(x, category.names = c("Shared_protein","HVP_target"), set_size=4, lty="longdash", label="count", label_alpha=0, label_size=5, edge_size=0.8) + 
	scale_color_brewer(palette = "Dark2") +
	scale_fill_gradient(low="gray100",high = "gray95",guide="none")

###### Figure3d_2 ######
special_protein = setdiff(HVP, shared_proteinlist)
data_subset = data[special_protein,]
cell_order = rownames(data_subset@meta.data[order(data_subset@meta.data$cell_type),])

ann_colors = list(
    scp = c('T-SCP' = "white", plexDIA = "firebrick"),
    cell_type = c('Hela cell' = "#1A9D77", Melanoma = "#D95E01", PDAC = "#7570B2", 'U-937' = "#E7288A")    
)
p = pheatmap::pheatmap(data_subset@assays$RNA@scale.data[,cell_order],show_rownames=F,show_colnames=F,cluster_cols = F,cluster_rows = T,kmeans_k=6,
	annotation_col = data_subset@meta.data[cell_order,c("cell_type","scp")], annotation_colors = ann_colors, treeheight_row = 0)

saveRDS(p,"source_data//figure3c_2_heatmap.rds")

###### Figure3e_1 ######
df = read.csv("source_data//source_data_fig3e_1.csv")

df %>%
    mutate(HVP_num = factor(HVP_num, levels = c(0,400,500,600,700))) %>%
    ggplot(aes(x=HVP_num, y=value, color = method))+
    geom_point(size=3) +
    scale_fill_manual(values = c("#EA7F01")) +
    theme_bw() +
    theme(panel.grid = element_blank()) +
    labs(y = "CCC")

# df %>%
# 	mutate(method = factor(method, levels = c("DEEP","Scaden","MuSiC"))) %>%
# 	mutate(condition = factor(condition, levels = c("withDVP","withoutDVP"))) %>%
# 	ggplot(mapping = aes(x = method, y = value, fill = condition)) + geom_bar(stat = 'identity', width = 0.5, position = position_dodge(0.7)) +
# 	scale_fill_manual(values = c("#EA7F01", "#F9CA00")) +  
# 	guides(fill = guide_legend(title = "method")) +  
# 	labs(x = "Methods", y = "CCC") +
# 	theme_bw() +
# 	theme(panel.grid = element_blank())

###### Figure3e_2 ######
df = read.csv("source_data//source_data_fig3e_2.csv")

df %>%
    mutate(HVP_num = factor(HVP_num, levels = c(0,400,500,600,700))) %>%
    ggplot(aes(x=HVP_num, y=value, color = method))+
    geom_point(size=3) +
    scale_fill_manual(values = c("#F9CA00")) +
    theme_bw() +
    theme(panel.grid = element_blank()) +
    labs(y = "RMSE")

# df %>%
# 	mutate(method = factor(method, levels = c("DEEP","Scaden","MuSiC"))) %>%
# 	mutate(condition = factor(condition, levels = c("withDVP","withoutDVP"))) %>%
# 	ggplot(mapping = aes(x = method, y = value, fill = condition)) + geom_bar(stat = 'identity', width = 0.5, position = position_dodge(0.7)) +
# 	scale_fill_manual(values = c("#EA7F01", "#F9CA00")) +  
# 	guides(fill = guide_legend(title = "method")) +  
# 	labs(x = "Methods", y = "RMSE") +
# 	theme_bw() +
# 	theme(panel.grid = element_blank())

###### Figure3f ######
gt = read.csv("Result//human_scp//SCoPEm_to_plexDIAm//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withoutHVPs_400//target_gt_fraction.csv", row.names = 1)
pred = read.csv("Result//human_scp//SCoPEm_to_plexDIAm//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withoutHVPs_400//target_predicted_fraction.csv", row.names = 1)
df = cbind(pred, gt)
colnames(df) = c("Melanoma_pred","PDAC_pred","U.937_pred", "Hela.cell_pred","Melanoma_gt","PDAC_gt","U.937_gt", "Hela.cell_gt")

tmp.ccc = CCC(df$Melanoma_gt, df$Melanoma_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$Melanoma_gt, df$Melanoma_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$Melanoma_gt - df$Melanoma_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = Melanoma_pred, y = Melanoma_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "Melanoma") + 
	    annotate("text",x=0,y=max(df$Melanoma_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$Melanoma_gt))) +
	    scale_y_continuous(limits = c(0,max(df$Melanoma_gt)))

tmp.ccc = CCC(df$PDAC_gt, df$PDAC_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$PDAC_gt, df$PDAC_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$PDAC_gt - df$PDAC_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df, mapping = aes(x = PDAC_pred, y = PDAC_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "PDAC") + 
	    annotate("text",x=0,y=max(df$PDAC_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$PDAC_gt))) +
	    scale_y_continuous(limits = c(0,max(df$PDAC_gt)))

tmp.ccc = CCC(df$U.937_gt, df$U.937_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$U.937_gt, df$U.937_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$U.937_gt - df$U.937_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df, mapping = aes(x = U.937_pred, y = U.937_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "U_937") + 
	    annotate("text",x=0,y=max(df$U.937_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$U.937_gt))) +
	    scale_y_continuous(limits = c(0,max(df$U.937_gt)))

tmp.ccc = CCC(df$Hela.cell_gt, df$Hela.cell_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$Hela.cell_gt, df$Hela.cell_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$Hela.cell_gt - df$Hela.cell_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df, mapping = aes(x = Hela.cell_pred, y = Hela.cell_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "Hela cell") + 
	    annotate("text",x=0,y=max(df$Hela.cell_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$Hela.cell_gt))) +
	    scale_y_continuous(limits = c(0,max(df$Hela.cell_gt)))

p1 + p2 + p3 + p4

###### Figure3g ######
gt = read.csv("Result//human_scp//SCoPEm_to_plexDIAm//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withHVPs_500//target_gt_fraction.csv", row.names = 1)
pred = read.csv("Result//human_scp//SCoPEm_to_plexDIAm//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withHVPs_500//target_predicted_fraction.csv", row.names = 1)
df = cbind(pred, gt)
colnames(df) = c("Melanoma_pred","PDAC_pred","U.937_pred", "Hela.cell_pred","Melanoma_gt","PDAC_gt","U.937_gt", "Hela.cell_gt")

tmp.ccc = CCC(df$Melanoma_gt, df$Melanoma_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$Melanoma_gt, df$Melanoma_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$Melanoma_gt - df$Melanoma_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = Melanoma_pred, y = Melanoma_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "Melanoma") + 
	    annotate("text",x=0,y=max(df$Melanoma_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$Melanoma_gt))) +
	    scale_y_continuous(limits = c(0,max(df$Melanoma_gt)))

tmp.ccc = CCC(df$PDAC_gt, df$PDAC_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$PDAC_gt, df$PDAC_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$PDAC_gt - df$PDAC_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df, mapping = aes(x = PDAC_pred, y = PDAC_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "PDAC") + 
	    annotate("text",x=0,y=max(df$PDAC_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$PDAC_gt))) +
	    scale_y_continuous(limits = c(0,max(df$PDAC_gt)))

tmp.ccc = CCC(df$U.937_gt, df$U.937_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$U.937_gt, df$U.937_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$U.937_gt - df$U.937_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df, mapping = aes(x = U.937_pred, y = U.937_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "U_937") + 
	    annotate("text",x=0,y=max(df$U.937_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$U.937_gt))) +
	    scale_y_continuous(limits = c(0,max(df$U.937_gt)))

tmp.ccc = CCC(df$Hela.cell_gt, df$Hela.cell_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$Hela.cell_gt, df$Hela.cell_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$Hela.cell_gt - df$Hela.cell_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df, mapping = aes(x = Hela.cell_pred, y = Hela.cell_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "Hela cell") + 
	    annotate("text",x=0,y=max(df$Hela.cell_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$Hela.cell_gt))) +
	    scale_y_continuous(limits = c(0,max(df$Hela.cell_gt)))

p1 + p2 + p3 + p4

###### Figure4a ######
mel = read.csv("source_data//melanoma_bulk_cellcycle_marker_proteome.csv", row.names = 1)
mon = read.csv("source_data//monocyte_bulk_cellcycle_marker_proteome.csv", row.names = 1)
data = t(cbind(mel, mon))
rownames(data) = c("mel_G1","mel_S","mel_G2","mon_G1","mon_S","mon_G2")
bk <- c(seq(-2,-0.1,by=0.01),seq(0,2,by=0.01))
pheatmap::pheatmap(data,cluster_cols = F,cluster_rows = F,color = c(colorRampPalette(colors = c("#4CAF49","white"))(length(bk)/2),colorRampPalette(colors = c("white","#964FA0"))(length(bk)/2)),scale = "column",border_color=NA)

###### Figure4b ######
data = read.csv("source_data//source_data_fig4b.csv")

data %>%
	mutate(Term = factor(Term, levels = rev(data$Term))) %>%
	mutate(type = factor(type, levels = c("G1","S","G2"))) %>%
	ggplot(aes(Pvalue..log10.,Term)) +  
	geom_bar(aes(fill = type),stat = "identity",width = 0.5) + 
	scale_fill_manual(values = c("#BC80BD","#CCEBC5","#FFED6F")) +
	theme_bw() +
	theme(panel.grid = element_blank(), axis.text.y = element_text(size = 10)) +
	labs(x = "Pvalue(-log10)", y = "Term")

###### Figure4c ######
mel = read.csv("source_data//melanoma_bulk_cellcycle_marker_proteome.csv", row.names = 1)
mon = read.csv("source_data//monocyte_bulk_cellcycle_marker_proteome.csv", row.names = 1)
cor = matrix(0,3,3)
for(i in 1:3)
{
	for(j in 1:3)
	{
		cor[i,j] = cor(mel[,i], mon[,j])
	}
}
rownames(cor) = c("mel_G1","mel_S","mel_G2")
colnames(cor) = c("mon_G1","mon_S","mon_G2")
pheatmap::pheatmap(cor,cluster_cols = F,cluster_rows = F,color = colorRampPalette(colors = c("white","firebrick"))(10))

###### Figure4d_1 ######
gt = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0_final//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0_final//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt, colour =celltype_pred)) + 
    geom_point(size = 0.5, alpha = 1) +
    theme_bw() +
    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
    annotate("text",x=0,y=max(df$comp_gt)-0.05,label=score_label,hjust = 0,size=3) +
    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
    scale_y_continuous(limits = c(0,max(df$comp_gt))) +
    scale_colour_manual(values = c("G1"="#BC80BD","S"="#CCEBC5","G2"="#FFED6F"))

###### Figure4d_2 ######
gt = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt, colour =celltype_pred)) + 
    geom_point(size = 0.5, alpha = 1) +
    theme_bw() +
    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
    annotate("text",x=0,y=max(df$comp_gt)-0.05,label=score_label,hjust = 0,size=3) +
    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
    scale_y_continuous(limits = c(0,max(df$comp_gt))) +
    scale_colour_manual(values = c("G1"="#BC80BD","S"="#CCEBC5","G2"="#FFED6F"))

###### Figure4e ######
data = read.csv("source_data//source_data_fig4e_1.csv")
data %>%
    mutate(type = factor(type, levels = c("G1","S","G2"))) %>%
    ggplot(aes(x=order, y=value, shape=direction, color=method))+
    geom_point(size=3)+
    facet_wrap( ~ type)+
    scale_colour_manual(values = c("DEEP"="#E41A1C","Scaden"="#377EB8","MuSiC"="#4DAF4A","BayesPrism"="#984EA3","DestVI"="#FF7F00")) +
    theme_bw() +
    theme(panel.grid = element_blank(), axis.title.x = element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
    scale_y_continuous(limits = c(-0.4,0.9),breaks = seq(-0.3,1,0.2)) +
    labs(y = "CCC")

data = read.csv("source_data//source_data_fig4e_2.csv")
data %>%
    mutate(type = factor(type, levels = c("G1","S","G2"))) %>%
    ggplot(aes(x=order, y=value, shape=direction, color=method))+
    geom_point(size=3)+
    facet_wrap( ~ type)+
    scale_colour_manual(values = c("DEEP"="#E41A1C","Scaden"="#377EB8","MuSiC"="#4DAF4A","BayesPrism"="#984EA3","DestVI"="#FF7F00")) +
    theme_bw() +
    theme(panel.grid = element_blank(), axis.title.x = element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
    scale_y_continuous(limits = c(0.08,0.24),breaks = seq(0,0.24,0.02)) +
    labs(y = "RMSE")

###### Figure5b ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")
melanoma <- melanoma %>%
  as.data.frame() %>%
  pivot_longer(cols = 11:17,
               names_to = "CellType",
               values_to = "Composition")
melanoma %>%
	mutate(CellType = factor(CellType, levels = c("Vertical",	"Radial",	"CD146.high",	"CD146.low",	"Insitu",	"Melanocytes",	"Stroma"))) %>%
	ggplot() + 
	geom_bar(aes(x = Sample.name, fill = CellType, y = Composition), stat = "identity") +
	scale_fill_brewer(palette = "RdGy") + 
	theme_bw() +
	theme(panel.grid = element_blank(),axis.text.x = element_blank(), axis.text.y = element_text(size = 8), axis.title = element_text(size = 10))


###### Figure5c ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")

melanoma <- melanoma %>%
  as.data.frame() %>%
  pivot_longer(cols = 11:17,
               names_to = "CellType",
               values_to = "Composition")

melanoma %>%
	mutate(CellType = factor(CellType, levels = c("Vertical",	"Radial",	"CD146.high",	"CD146.low",	"Insitu",	"Melanocytes",	"Stroma"))) %>%
	ggplot(aes(CellType,Composition)) +  
	geom_boxplot(aes(fill = CellType),size = 0.3,width = 0.6,outlier.color = "white") + 
	scale_fill_brewer(palette = "RdGy") +  
	labs(x = "Melanoma class", y = "Composition") +
	theme_bw() +
	theme(panel.grid = element_blank(),axis.text.x = element_text(size = 8, angle = 60, vjust = 1, hjust = 1), axis.text.y = element_text(size = 8), axis.title = element_text(size = 10))

###### Figure5d ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")
melanoma <- as_tibble(melanoma)
melanoma_subset = melanoma[melanoma$Biopsy.site %in% c("Bone", "Brain", "LN", "Lung", "SC"),]
melanoma_subset_avg = aggregate(melanoma_subset[,11:17], by=list(type=melanoma_subset$Biopsy.site),mean)
cor = matrix(0,5,5)
rownames(cor) = colnames(cor) = c("Bone", "Brain", "LN", "Lung", "SC")
for(i in 1:5)
{
	for(j in 1:5)
	{
			cor[i,j] = cor(unlist(melanoma_subset_avg[i,2:8]), unlist(melanoma_subset_avg[j,2:8]))
	}
}
bk = unique(c(seq(0.9,1, length=10)))
pheatmap::pheatmap(cor,cluster_cols = F,cluster_rows = F,color = colorRampPalette(colors = c("white","firebrick"))(10),breaks = bk)

###### Figure5d_new ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")
melanoma <- as_tibble(melanoma)

melanoma_subset = melanoma[melanoma$BRAF.mutation != "",]
sfit <- survfit(Surv(Overall.survival..months., Death)~BRAF.mutation, data=melanoma_subset)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("No", "Yes"), legend.title="BRAF_mutation", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### Figure5e_1 ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")
melanoma <- as_tibble(melanoma)

melanoma$Vertical.label = cut(melanoma$Vertical, breaks=c(0, median(melanoma$Vertical), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~Vertical.label, data=melanoma)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="Vertical", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### Figure5e_2 ######
melanoma$Radial.label = cut(melanoma$Radial, breaks=c(0, median(melanoma$Radial), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~Radial.label, data=melanoma)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="Radial", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5a ######
melanoma$CD146.high.label = cut(melanoma$CD146.high, breaks=c(0, median(melanoma$CD146.high), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~CD146.high.label, data=melanoma)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="CD146.high", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5b ######
melanoma$CD146.low.label = cut(melanoma$CD146.low, breaks=c(0, median(melanoma$CD146.low), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~CD146.low.label, data=melanoma)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="CD146.low", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5c ######
melanoma$Insitu.label = cut(melanoma$Insitu, breaks=c(0, median(melanoma$Insitu), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~Insitu.label, data=melanoma)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="Insitu", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5d ######
melanoma$Melanocytes.label = cut(melanoma$Melanocytes, breaks=c(0, median(melanoma$Melanocytes), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~Melanocytes.label, data=melanoma)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="Melanocytes", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5e ######
melanoma$Stroma.label = cut(melanoma$Stroma, breaks=c(0, median(melanoma$Stroma), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~Stroma.label, data=melanoma)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="Stroma", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)


###### FigureS5f ######
melanoma_subset = melanoma[melanoma$BRAF.mutation == "N",]
melanoma_subset$Vertical.label = cut(melanoma_subset$Vertical, breaks=c(0, median(melanoma_subset$Vertical), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~ Vertical.label, data=melanoma_subset)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE, 
		   legend.labs=c("low", "high"), legend.title="Vertical",
           xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5g ######
melanoma_subset = melanoma[melanoma$BRAF.mutation == "N",]
melanoma_subset$Radial.label = cut(melanoma_subset$Radial, breaks=c(0, median(melanoma_subset$Radial), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~Radial.label, data=melanoma_subset)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="Radial", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5h ######
melanoma_subset = melanoma[melanoma$BRAF.mutation == "Y",]
melanoma_subset$Vertical.label = cut(melanoma_subset$Vertical, breaks=c(0, median(melanoma_subset$Vertical), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~ Vertical.label, data=melanoma_subset)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE, 
		   legend.labs=c("low", "high"), legend.title="Vertical",
           xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5i ######
melanoma_subset = melanoma[melanoma$BRAF.mutation == "Y",]
melanoma_subset$Radial.label = cut(melanoma_subset$Radial, breaks=c(0, median(melanoma_subset$Radial), Inf), labels=c("low", "high"))

sfit <- survfit(Surv(Overall.survival..months., Death)~Radial.label, data=melanoma_subset)
ggsurvplot(sfit, conf.int=TRUE, pval=TRUE,
           legend.labs=c("low", "high"), legend.title="Radial", xlab = "Time (Months)",
           palette="uchicago",risk.table = TRUE)

###### FigureS5j ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")
melanoma = melanoma[melanoma$BRAF.mutation != "",]
melanoma <- melanoma %>%
  as.data.frame() %>%
  pivot_longer(cols = 11:17,
               names_to = "CellType",
               values_to = "Composition")
melanoma %>%
	mutate(CellType = factor(CellType, levels = c("Vertical",	"Radial",	"CD146.high",	"CD146.low",	"Insitu",	"Melanocytes",	"Stroma"))) %>%
	ggplot(aes(CellType,Composition)) +  
	geom_boxplot(aes(fill = BRAF.mutation),size = 0.3,width = 0.6,outlier.color = "white") + 
	scale_fill_brewer(palette = 7) +  
	labs(x = "Melanoma class", y = "Composition") +
	theme_bw() +
	theme(panel.grid = element_blank(),axis.text.x = element_text(size = 8, angle = 60, vjust = 1, hjust = 1), axis.text.y = element_text(size = 8), axis.title = element_text(size = 10))

###### FigureS5k ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")
melanoma$Vertical.label = cut(melanoma$Vertical, breaks=c(0, median(melanoma$Vertical), Inf), labels=c("low", "high"))
melanoma$Radial.label = cut(melanoma$Radial, breaks=c(0, median(melanoma$Radial), Inf), labels=c("low", "high"))
fit.coxph <- coxph(Surv(Overall.survival..months., Death) ~ BRAF.mutation + Vertical.label + Radial.label, data = melanoma)
ggforest(fit.coxph, data = melanoma,fontsize = 1)

###### EDFigure5a_new ######
melanoma = read.csv("D://workbase//MyWork//scDeconvolution//Result//clinical_melanoma//cGAN_AEimpute_ours_Epoch_100_BatchSize_50_LearningRate_0.0001_Scaling_min_max_feature_use_share_mixup_sample_1000_None//Target_metadata.csv")
melanoma_sub = melanoma[melanoma$Sample.name %in% c("PD73","TIL31","TIL72"),]
melanoma_sub = melanoma_sub[,c("Vertical",	"Radial",	"CD146.high",	"CD146.low",	"Insitu",	"Melanocytes",	"Stroma")]
data <- data.frame(
    group=c("Vertical",	"Radial",	"CD146.high",	"CD146.low",	"Insitu",	"Melanocytes",	"Stroma"),
    value=as.numeric(melanoma_sub[1,])
)
data %>% 
    mutate(group = factor(group, levels = c("Vertical",	"Radial",	"CD146.high",	"CD146.low",	"Insitu",	"Melanocytes",	"Stroma"))) %>%
    ggplot(aes(x="", y=value, fill=group)) +
    geom_bar(stat="identity", width=1) +
    coord_polar("y", start=0)+theme_void()+scale_fill_brewer(palette = "RdGy")

###### EDFigure2a_1 ######
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//pSCoPE_Huffman//PDAC//limmaCorrected_normed_prePCA_PDAC_Coverage_PIF50_mrri10.csv", row.names = 1)
genelist1 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//pSCoPE_Leduc//processed_data//t6.csv", row.names = 1)
genelist2 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//SCoPE2_Leduc//prot_proc.csv", row.names = 1)
genelist3 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//T-SCP//Data//T_SCP_HelaCell_normed_proteinlist.csv", row.names = 1)
genelist4 = rownames(data)
data = read.csv("D://workbase//MyWork//scProteome//Data//scProteome//plexDIA//Data//plexDIA_PDAC_Melanoma_U937_normed_proteinlist.csv", row.names = 1)
genelist5 = rownames(data)

x <- list(A=genelist3,
          B=genelist1,
          C=genelist2)
ggVennDiagram(x, category.names = c("SCoPE2_Leduc","pSCoPE_Huffman","pSCoPE_Leduc"), set_size=4, lty="longdash", label="count", label_alpha=0, label_size=5, edge_size=0.8) + 
	scale_color_brewer(palette = "Set2") +
	scale_fill_gradient(low="gray100",high = "gray95",guide="none")

###### EDFigure2a_2 ######
x <- list(A=genelist4,
          B=genelist5)
ggVennDiagram(x, category.names = c("T-SCP","plexDIA"), set_size=4, lty="longdash", label="count", label_alpha=0, label_size=5, edge_size=0.8) + 
	scale_colour_manual(values = c("#E78AC3","#A6D854")) +
	scale_fill_gradient(low="gray100",high = "gray95",guide="none")

###### EDFigure2a_3 ######
genelist1 = rownames(read.csv("source_data//pSCoPE_Huffman_PDAC+pSCoPE_Leduc+SCoPE2_Leduc_proteinlist.csv", row.names = 1))
genelist2 = rownames(read.csv("source_data//T-SCP+plexDIA_proteinlist.csv", row.names = 1))
x <- list(A=genelist1,
          B=genelist2)
ggVennDiagram(x, category.names = c("reference","target"), set_size=4, lty="longdash", label="count", label_alpha=0, label_size=5, edge_size=0.8) + 
	scale_color_brewer(palette = "Dark2") +
	scale_fill_gradient(low="gray100",high = "gray95",guide="none")

###### EDFigure2b ######
shared_proteinlist = intersect(genelist1,genelist2)

scp_path = "D://workbase//MyWork//scProteome//Data//scProteome//T-SCP+plexDIA//merged_scProteomic_minmax.h5ad"
missfeaturedata_path = "source_data//human_cellline_AE_Recon_Source_missingfeature.h5ad"
target_scp = readH5AD(scp_path)
names(assays(target_scp))="counts"
target_scp <- as.Seurat(target_scp, counts = "counts", data = "counts")

miss_scp = readH5AD(missfeaturedata_path)
names(assays(miss_scp))="counts"
miss_scp <- as.Seurat(miss_scp, counts = "counts", data = "counts")

# Convert(scp_path, dest = "h5seurat", overwrite = TRUE)
# data <- LoadH5Seurat(paste0(unlist(strsplit(scp_path, split=".h5ad"))[1], ".h5seurat"))
# all.genes <- rownames(data)
# data <- ScaleData(data, features = all.genes)
# data <- FindVariableFeatures(data, selection.method = "vst", nfeatures = 400)
# HVP = VariableFeatures(data)

special_protein = rownames(miss_scp)
data_subset = target_scp[special_protein,]
data_subset <- ScaleData(data_subset, features = special_protein)
cell_order = rownames(data_subset@meta.data[order(data_subset@meta.data$cell_type),])

ann_colors = list(
    scp = c('T-SCP' = "white", plexDIA = "firebrick"),
    cell_type = c('Hela cell' = "#1A9D77", Melanoma = "#D95E01", PDAC = "#7570B2", 'U-937' = "#E7288A")    
)

bk <- c(seq(-6,-0.1,by=0.01),seq(0,6,by=0.01))
pheatmap::pheatmap(t(data_subset@assays$originalexp@scale.data[,cell_order]),show_rownames=F,show_colnames=T,cluster_cols = T,cluster_rows = F,
	color = c(colorRampPalette(colors = c("#4CAF49","white"))(length(bk)/2),colorRampPalette(colors = c("white","#964FA0"))(length(bk)/2)),
	annotation_row = data_subset@meta.data[cell_order,c("cell_type","scp")], annotation_colors = ann_colors, treeheight_col = 0, scale = "row", fontsize = 5)

###### EDFigure1f ######
data = read.csv("source_data//source_data_EDfig1f.csv", row.names=NULL)
data %>%
    mutate(reference_sample_number = factor(reference_sample_number, levels = c("2000","4000","6000"))) %>%
    ggplot(aes(reference_sample_number,CCC)) +  
    geom_boxplot(aes(fill = reference_sample_number), size = 0.4, width = 0.4) + 
    scale_fill_brewer(palette = 1) +  
    guides(fill = guide_legend(title = "none")) +  
    labs(x = "mixed-up reference sample number", y = "CCC") +
    theme_bw() +
    theme(panel.grid = element_blank()) +
    scale_y_continuous(limits=c(0,1)) +
    geom_vline(xintercept = c(1.5,2.5,3.5,4.5), size = .25, linetype = "dotted")

###### EDFigure1g ######
data = read.csv("source_data//source_data_EDfig1g.csv", row.names=NULL)
data %>%
    mutate(reference_sample_size = factor(reference_sample_size, levels = c("5","10","15","20"))) %>%
    ggplot(aes(reference_sample_size,CCC)) +  
    geom_boxplot(aes(fill = reference_sample_size), size = 0.4, width = 0.4) + 
    scale_fill_brewer(palette = 7) +  
    guides(fill = guide_legend(title = "none")) +  
    labs(x = "mixed-up reference sample size", y = "CCC") +
    theme_bw() +
    theme(panel.grid = element_blank()) +
    scale_y_continuous(limits=c(0,1)) +
    geom_vline(xintercept = c(1.5,2.5,3.5,4.5), size = .25, linetype = "dotted")

###### EDFigure1f ######
data = read.csv("source_data//source_data_EDfig1f_RMSE.csv", row.names=NULL)
data %>%
    mutate(reference_sample_number = factor(reference_sample_number, levels = c("2000","4000","6000"))) %>%
    ggplot(aes(reference_sample_number,RMSE)) +  
    geom_boxplot(aes(fill = reference_sample_number), size = 0.4, width = 0.4) + 
    scale_fill_brewer(palette = 1) +  
    guides(fill = guide_legend(title = "none")) +  
    labs(x = "mixed-up reference sample number", y = "RMSE") +
    theme_bw() +
    theme(panel.grid = element_blank()) +
    scale_y_continuous(limits=c(0,0.5)) +
    geom_vline(xintercept = c(1.5,2.5,3.5,4.5), size = .25, linetype = "dotted")

###### EDFigure1g ######
data = read.csv("source_data//source_data_EDfig1g_RMSE.csv", row.names=NULL)
data %>%
    mutate(reference_sample_size = factor(reference_sample_size, levels = c("5","10","15","20"))) %>%
    ggplot(aes(reference_sample_size,RMSE)) +  
    geom_boxplot(aes(fill = reference_sample_size), size = 0.4, width = 0.4) + 
    scale_fill_brewer(palette = 7) +  
    guides(fill = guide_legend(title = "none")) +  
    labs(x = "mixed-up reference sample size", y = "RMSE") +
    theme_bw() +
    theme(panel.grid = element_blank()) +
    scale_y_continuous(limits=c(0,0.5)) +
    geom_vline(xintercept = c(1.5,2.5,3.5,4.5), size = .25, linetype = "dotted")

###### EDFigure1g_new ######
gt = read.csv("Result//mouse_scp//N2_to_nanoPOTS//DEEP_noimpute_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize15_feature_withHVPs_500//target_gt_fraction.csv", row.names = 1)
pred = read.csv("Result//mouse_scp//N2_to_nanoPOTS//DEEP_noimpute_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize15_feature_withHVPs_500//target_predicted_fraction.csv", row.names = 1)
df = cbind(pred, gt)
colnames(df) = c("C10_pred","SVEC_pred","RAW_pred","C10_gt","SVEC_gt","RAW_gt")

tmp.ccc = CCC(df$C10_gt, df$C10_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$C10_gt, df$C10_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$C10_gt - df$C10_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = C10_pred, y = C10_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "C10") + 
	    annotate("text",x=0,y=max(df$C10_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$C10_gt))) +
	    scale_y_continuous(limits = c(0,max(df$C10_gt)))

tmp.ccc = CCC(df$SVEC_gt, df$SVEC_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$SVEC_gt, df$SVEC_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$SVEC_gt - df$SVEC_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df, mapping = aes(x = SVEC_pred, y = SVEC_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "SVEC") + 
	    annotate("text",x=0,y=max(df$SVEC_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$SVEC_gt))) +
	    scale_y_continuous(limits = c(0,max(df$SVEC_gt)))

tmp.ccc = CCC(df$RAW_gt, df$RAW_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$RAW_gt, df$RAW_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$RAW_gt - df$RAW_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df, mapping = aes(x = RAW_pred, y = RAW_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "RAW") + 
	    annotate("text",x=0,y=max(df$RAW_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$RAW_gt))) +
	    scale_y_continuous(limits = c(0,max(df$RAW_gt)))

p1 + p2 + p3

###### FigureS1d ######
data1 = read.csv("D://workbase//MyWork//scDeconvolution//Data//scProteome//N2_zscore//N2_SCP_zscore.csv", row.names = 1)
data1[is.na(data1)]=0
meta1 = read.csv("D://workbase//MyWork//scDeconvolution//Data//scProteome//N2_zscore//N2_meta.csv", row.names = 1)
meta1 = meta1[colnames(data1),]

data2 = read.csv("D://workbase//MyWork//scDeconvolution//Data//scProteome//nanoPOTS_zscore//nanoPOTS_SCP_zscore.csv", row.names = 1)
data2[is.na(data2)]=0
meta2 = read.csv("D://workbase//MyWork//scDeconvolution//Data//scProteome//nanoPOTS_zscore//nanoPOTS_meta_zscore.csv", row.names = 1)
meta2 = meta2[colnames(data2),]

features = intersect(rownames(data1),rownames(data2))
N2 <- CreateSeuratObject(counts = data1[features,], meta.data = meta1, min.cells = 0, min.features = 0)
nanoPOTS <- CreateSeuratObject(counts = data2[features,], meta.data = meta2, min.cells = 0, min.features = 0)

data = merge(N2, nanoPOTS)
data <- ScaleData(data)
data <- FindVariableFeatures(data, nfeatures = 500)
data <- RunPCA(data, features = VariableFeatures(object = data))
data <- FindNeighbors(data, dims = 1:20)
data <- FindClusters(data, resolution = 0.5)
data <- RunTSNE(data, dims = 1:20)
DimPlot(data, reduction = "tsne", group.by = "Source", cols = c("#4566A2", "#FFC000"))

###### EDFigure2c_heatmap ######
protein_ordered = readRDS("source_data//EDfigure2c_2d_protein_order.rds")
cell_order = readRDS("source_data//EDfigure2c_2d_cell_order.rds")
data_raw = readH5AD("Result//human_scp//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withHVPs_500_final//Pseudo_Bulk_Source_4000.h5ad")
names(assays(data_raw))="counts"
data_raw <- as.Seurat(data_raw, counts = "counts", data = "counts")
all.genes <- rownames(data_raw)
data_raw <- ScaleData(data_raw, features = all.genes)
p = pheatmap::pheatmap(data_raw@assays$originalexp@scale.data[protein_ordered,cell_order],show_rownames=F,show_colnames=F,cluster_cols = F,cluster_rows = F,color = colorRampPalette(c("navy", "white", "firebrick3"))(100))
ggsave("Figure//Figure_update//EDFigure2c_heatmap.png", p , width = 10, height = 3, dpi = 300)

###### EDFigure2d_heatmap ######
data = readH5AD("Result//human_scp//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withHVPs_500_final//AE_Recon_Source.h5ad")
names(assays(data))="counts"
data <- as.Seurat(data, counts = "counts", data = "counts")
all.genes <- rownames(data)
data <- ScaleData(data, features = all.genes)
p = pheatmap::pheatmap(data@assays$originalexp@scale.data[protein_ordered,cell_order],show_rownames=F,show_colnames=F,cluster_cols = F,cluster_rows = F,color = colorRampPalette(c("navy", "white", "firebrick3"))(100))
ggsave("Figure//Figure_update//EDFigure2d_heatmap.png", p , width = 10, height = 3, dpi = 300)

###### EDFigure2e_heatmap ######
data = readH5AD("Result//human_scp//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_cell_4000_1000_source_samplesize50_feature_withHVPs_500_final//Pseudo_Bulk_Target_1000.h5ad")
names(assays(data))="counts"
data <- as.Seurat(data, counts = "counts", data = "counts")
all.genes <- rownames(data)
data <- ScaleData(data, features = all.genes)
p = pheatmap::pheatmap(data@assays$originalexp@scale.data[protein_ordered,],show_rownames=F,show_colnames=F,cluster_cols = T,cluster_rows = F,color = colorRampPalette(c("navy", "white", "firebrick3"))(100),treeheight_col = 0)
ggsave("Figure//Figure_update//EDFigure2e_heatmap.png", p , width = 10, height = 3, dpi = 300)

###### EDFigure3a ######
gt = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0_final//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0_final//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure3b ######
gt = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//Scaden_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//Scaden_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure3c ######
gt = read.csv("other_method//MuSiC//Data//Cell_cycle//melanoma_to_monocyte//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("other_method//MuSiC//Data//Cell_cycle//melanoma_to_monocyte//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure3d ######
gt = read.csv("other_method//BayesPrism//Result//cell_cycle//melanoma_to_monocyte//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("other_method//BayesPrism//Result//cell_cycle//melanoma_to_monocyte//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure3e ######
gt = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//DestVI_target1000//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//melanomaB_to_monocyteB//DestVI_target1000//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure4a ######
gt = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//DEEP_Epoch_30_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure4b ######
gt = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//Scaden_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//Scaden_Epoch_60_BatchSize_50_dropout_LearningRate_0.0001_Scaling_min_max_mixup_sample_4000_1000_source_samplesizeNone_feature_withoutHVPs_0//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure4c ######
gt = read.csv("other_method//MuSiC//Data//Cell_cycle//monocyte_to_melanoma//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("other_method//MuSiC//Data//Cell_cycle//monocyte_to_melanoma//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure4d ######
gt = read.csv("other_method//BayesPrism//Result//cell_cycle//monocyte_to_melanoma//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("other_method//BayesPrism//Result//cell_cycle//monocyte_to_melanoma//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### EDFigure4e ######
gt = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//DestVI_target1000//target_gt_fraction.csv", row.names = 1)
gt <- gt %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")
pred = read.csv("Result//cell_cycle//monocyteB_to_melanomaB//DestVI_target1000//target_predicted_fraction.csv", row.names = 1)
pred <- pred %>%
    as.data.frame() %>%
    pivot_longer(cols = 1:3, names_to = "CellType", values_to = "Composition")

df = cbind(pred, gt)
colnames(df) = c("celltype_pred","comp_pred","celltype_gt", "comp_gt")

tmp.ccc = CCC(df$comp_gt, df$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df$comp_gt, df$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df$comp_gt - df$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p1 = ggplot(data = df, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "All cell cycle states") + 
	    annotate("text",x=0,y=max(df$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df$comp_gt)))

df_subset = df[df$celltype_pred == "G1",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p2 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G1") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt)))

df_subset = df[df$celltype_pred == "S",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p3 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "S") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

df_subset = df[df$celltype_pred == "G2",]
tmp.ccc = CCC(df_subset$comp_gt, df_subset$comp_pred)
CCC = round(tmp.ccc$rho.c[,1], digits = 3)
Cor = round(cor(df_subset$comp_gt, df_subset$comp_pred, method = "pearson"), digits = 3)
RMSE = round(sqrt(mean((df_subset$comp_gt - df_subset$comp_pred)^2)), digits = 3)
score_label = paste0("CCC = ", CCC, " \n RMSE = ", RMSE, " \n r = ", Cor)
p4 = ggplot(data = df_subset, mapping = aes(x = comp_pred, y = comp_gt)) + 
	    geom_point(colour = "#1E78B7", size = 0.3, alpha = 0.5) +
	    theme_bw() +
	    theme(panel.grid = element_blank(), plot.title = element_text(hjust = 0.5, size = 10)) +
	    labs(x = "Predicted proportions", y = "True proportions", title = "G2") + 
	    annotate("text",x=0,y=max(df_subset$comp_gt)-0.1,label=score_label,hjust = 0,size=3) +
	    stat_smooth(method = 'lm', colour = "#B22222", size = 0.5) +
	    scale_x_continuous(limits = c(0,max(df_subset$comp_gt))) +
	    scale_y_continuous(limits = c(0,max(df_subset$comp_gt))) 

p1 + p2 + p3 + p4

###### Figure5b_heatmap ######
data=readH5AD("Result//clinical_melanoma//melanoma_clinical_tissue_proteomic.h5ad")
names(assays(data))="counts"
data <- as.Seurat(data, counts = "counts", data = "counts")
p = pheatmap::pheatmap(data@assays$originalexp@counts,show_rownames=F,show_colnames=F,cluster_cols = F,cluster_rows = T,color = colorRampPalette(c("#4D4D4D", "white", "#B1172B"))(100),treeheight_row = 0)
ggsave("Figure//Figure_update//figure5b_heatmap.png", p , width = 10, height = 3, dpi = 300)

###### EDFigure5b_heatmap1 ######
data=readH5AD("Result//clinical_melanoma//melanoma_clinical_tissue_proteomic.h5ad")
names(assays(data))="counts"
data <- as.Seurat(data, counts = "counts", data = "counts")
pheatmap::pheatmap(data@assays$originalexp@counts[,c("PD73","TIL31","TIL72")],show_rownames=F,show_colnames=F,cluster_cols = F,cluster_rows = T,color = colorRampPalette(c("#4D4D4D", "white", "#B1172B"))(100),treeheight_row = 0)

###### EDFigure5a_heatmap2 ######
embed_data=readH5AD("Result//clinical_melanoma//target_embedding_adata.h5ad")
names(assays(embed_data))="counts"
embed_data <- as.Seurat(embed_data, counts = "counts", data = "counts")
bk <- c(seq(-2,-0.01,by=0.01),seq(0,2,by=0.01))
pheatmap::pheatmap(embed_data@assays$originalexp@counts[,c("PD73","TIL31","TIL72")],show_rownames=F,show_colnames=F,cluster_cols = F,cluster_rows = F,color = c(colorRampPalette(colors = c("#4D4D4D","white"))(length(bk)/2),colorRampPalette(colors = c("white","#B1172B"))(length(bk)/2)),treeheight_row = 0,breaks = bk)
