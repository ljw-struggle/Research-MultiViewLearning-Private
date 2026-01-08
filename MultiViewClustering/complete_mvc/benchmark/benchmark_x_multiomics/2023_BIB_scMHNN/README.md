# scMHNN: a novel hypergraph neural network for integrative analysis of single-cell epigenomic, transcriptomic and proteomic data

We present scMHNN to integrate single-cell multi-omics data based on a hypergraph neural network: 

*  We generate the multi-omics hypergraph by combination of the modality-specific hyperedges to model high-order and heterogeneous data relationships. 
*  We propose a dual-contrastive loss to learn discriminative cell representation in a self-supervised manner with intra-cell and inter-cell loss together.
*  Based on the pretrained hypergraph encoder, scMHNN can achieve more accurate cell type annotation by fine-tuning in the case of only a small number of labeled cells.

PAPER: [scMHNN](https://doi.org/10.1093/bib/bbad391)
GITHUB: [scMHNN](https://github.com/zhanglabNKU/scMHNN)
