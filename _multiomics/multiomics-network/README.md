# Data Oreoricess


## Sequence Data and Gene Regulatory Network of TFLink Database (Species Specific)

```shell
# 1/ Data process.
$ python 1.1.generate_data_sequnce.py

# 2/ Split the data to training and testing set.
$ python 1.2.split_train_test_sequence.py

# 3/ Feature Extraction of DNA Sequences.
$ python 1.3.feature_dna_sequence.py

# 4/ Feature Extraction of Protein Sequences.
$ python 1.4.feature_prot_sequence.py
```

## Single-Cell Transcriptomic Data and Gene Regulatory Network of Benchmark Datasets (Cell Type Specific)

```shell
# Data process.
$ nohup bash 2.benchmark_data_preprocess.sh > 2.benchmark_data_preprocess.log 2>&1 &
```

## Single-Cell Transcriptomic Data and Gene Regulatory Network of GRNdb Database (Tissue Specific)

```shell
# Data process.
$ nohup bash 3.tissue_specific_data_preprocess.sh > 3.tissue_specific_data_preprocess.log 2>&1 &
```

## Single-Cell Transcriptomic Data and Gene Regulatory Network of GRNdb Database (Cell Type Specific)

```shell
# Data process.
$ nohup bash 4.cell_type_specific_data_preprocess.sh > 4.cell_type_specific_data_preprocess.log 2>&1 &
```

# Simulation Data

```shell
# Data process.
$ nohup bash 5.generate_simulation_data.sh > 5.generate_simulation_data.log 2>&1 &
```


# Reference
[1] [TFLink: An integrated gateway to access transcription factor - target gene interactions for multiple species](https://tflink.net/)
[2] [Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data](https://www.nature.com/articles/s41592-019-0690-6)
[3] [GRNdb: decoding the gene regulatory networks in diverse human and mouse conditions](http://www.grndb.com/)
[4] [SEACells infers transcriptional and epigenomic cellular states from single-cell genomics data](https://www.nature.com/articles/s41587-023-01716-9)
[5] [Joint reconstruction of multiple gene regulatory networks with common hub genes](https://github.com/wenpingd/JRmGRN)
