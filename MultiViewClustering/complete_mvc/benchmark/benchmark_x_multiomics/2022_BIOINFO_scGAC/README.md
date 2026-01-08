# scGAC: a graph attentional architecture for clustering single-cell RNA-seq data

scGAC updates cell features by means of graph attentional autoencoder which involves cell-cell similarity and introduces self-optimizing module for further clustering.

Here, we propose a novel unsupervised clustering method, scGAC (single-cell Graph Attentional Clustering), for scRNA-seq data. scGAC firstly constructs a cell graph and refines it by network denoising. Then, it learns clustering-friendly representation of cells through a graph attentional autoencoder, which propagates information across cells with different weights and captures latent relationship among cells. Finally, scGAC adopts a selfoptimizing method to obtain the cell clusters. Experiments on 16 real scRNA-seq datasets show that scGAC achieves excellent performance and outperforms existing state-of-art single-cell clustering methods.

- [Code](https://github.com/Joye9285/scGAC)
- [Paper](https://watermark.silverchair.com/btac099.pdf)

## Usage

### Data preparation
All the original tested datasets ([Yan](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE36552), [Biase](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249), [Klein](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525), [Romanov](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74672), [Muraro](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85241), [Björklund](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580), [PBMC](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc6k), [Zhang](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108989), [Guo](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE99254), [Brown.1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137710), [Brown.2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137710), [Chung](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75688), [Sun.1](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066), [Sun.2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066), [Sun.3](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066) and [Habib](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE104525)) can be downloaded. 

For example, the original expression matrix `data_raw.tsv` of dataset [Biase](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249) is downloaded and put into `/data/Biase`. Before clustering, low-quality cells and genes can be filtered by running the following command: 
```Bash
python preprocess.py Biase
```
And a pre-processed expression matrix `data.tsv` is produced under `/data/Biase`. 

### Run the scGAC
To use scGAC, you should specify the two parameters, `dataset_str` and `n_clusters`, and run the following command:
```Bash
python main.py dataset_str n_clusters
```
where `dataset_str` is the name of dataset and `n_clusters` is the number of clusters.<br>

For example, for dataset `Biase`, you can run the following command:
```Bash
python main.py Biase 3
```

For your own dataset named `Dataset_X`, you can first create a new folder under `/data`, and put the expression matrix file `data.tsv` into `/data/Dataset_X`, then run scGAC on it.<br>
Please note that we recommend you use the `raw count` expression matrix as the input of scGAC. 

### Outputs
You can obtain the predicted clustering result `pred_DatasetX.txt` and the learned cell embeddings `hidden_DatasetX.tsv` under the folder `/result`.

### Optional parameters
To see the optional parameters, you can run the following command:
```Bash
python main.py -h
```

For example, if you want to evaluate the clustering results (by specifing `--subtype_path`) and change the number of nearest neighbors (by specifing `--k`), you can run the following command:
```Bash
python main.py Biase 3 --subtype_path data/Biase/subtype.ann --k 4
```
Results in the paper were obtained with default parameters.
