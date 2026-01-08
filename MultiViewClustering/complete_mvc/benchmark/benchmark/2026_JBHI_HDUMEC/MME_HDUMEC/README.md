# HDUMEC
HDUMEC, a two-stage deep learning framework that learns robust and biologically meaningful multi-omics embeddings and performs clustering by combining Hierarchical Dynamic Uncertainty-Aware MultiModal Embedding with deep Embedded Clustering. 

## Overview

Bulk and single-cell multi-omics data provide unprecedented opportunities to decode complex biological systems, yet their integration remains challenging due to high dimensionality, pervasive noise, and heterogeneous data quality.  Existing approaches typically overlook the dynamic data quality across features, modalities, and samples, relying instead on uniform weighting of all signals. We present **HDUMEC**, a two-stage deep learning framework that learns robust and biologically meaningful multi-omics embeddings and performs clustering by combining **Hierarchical Dynamic Uncertainty-Aware MultiModal Embedding (HDUMEC)** with deep **Embedded Clustering (DEC)**. In the first stage, we introduce hierarchical uncertainty modeling that quantifies dynamic uncertainty across features, modalities, and samples. This hierarchical design ensures interpretability and adaptively suppresses low-quality signals while enhancing reliable signals. Moreover, we further employ a deep embedded clustering strategy to jointly optimize latent representations and cluster assignments through a KL-divergence objective. Comprehensive evaluations on both bulk cancer multi-omics datasets from **TCGA** and **single-cell multi-omics datasets from different sequencing technologies** demonstrate that HDUMEC  outperforms state-of-the-art baselines in clustering quality, few-shot classification, robustness to noise, and scalability. Furthermore, uncertainty estimates show strong correlations with data noise levels across different dimensions and highlight ambiguous cell or sample states, thereby providing interpretability. Together, these results suggest that HDUMEC is a promising tool for robust and interpretable multi-omics integration and clustering in both bulk and single-cell contexts. 

## 📦 Requirements
- scikit-learn  
- pytorch
- scanpy
- numpy
- pandas
- matplotlib
- scipy

## 📚 Data Sources and Implementation Notes

### Bulk cancer multi-omics datasets from **TCGA**
Omics data of LGG, KIPAN, and BRCA were obtained from The Cancer Genome Atlas Program (TCGA) through Broad GDAC Firehose (https://gdac.broadinstitute.org/). PAM50 breast cancer subtypes of TCGA BRCA patients were obtained through the TCGAbiolinks R package (v2.12.6, http://bioconductor.org/packages/release/bioc/html/TCGAbiolinks.html). Preprocessed data are provided at *./data/data_bulk_multiomics* folder of this repository.

### Single-cell multi-omics datasets from different sequencing technologies, including **DOGMA-seq**, **TEA-seq**, and **NEAT-seq**.
The DOGMA-seq dataset is downloaded from https://osf.io/6kr4v. TEA-seq dataset was downloaded from Gene Expression Omnibus (GEO) with accession number GSE158013. NEAT-seq dataset was obtained from GEO with accession number GSE178707. The preprocessed datasets can be accessed at *./data/data_sc_multiomics* folder of this repository.


## 🚀 Usage
To reproduce the results, simply run the following command:

```bash
python HDUMEC.py
```

## ⚠️ Disclaimer
This code is intended for **academic research only** and is **not approved for clinical use**. If you have any questions, please contact me via the following email [ljwstruggle@gmail.com](ljwstruggle@gmail.com).

## 📜 License
All rights reserved. Redistribution or commercial use is prohibited without prior written permission.
