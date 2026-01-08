# Traditional Clustering Methods

- Comparison of traditional clustering methods: [Link](./sklearn_clustering.ipynb)
- Scikit-Learn Paper: [Link](https://dl.acm.org/doi/10.5555/1953048.2078195)
- Comparison of different clustering algorithms in Scikit-learn: [Link](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py)
- Comparison of different clustering algorithms in Scanpy: [Link](https://scanpy.readthedocs.io/)
- [Awesome Clustering Algorithms](https://github.com/dreamhomes/awesome-clustering-algorithms)
- [Cluster Analysis](https://github.com/haoyuhu/cluster-analysis)

## Traditional Methods
- K-means
- AP（Affinity Propagation）
- Mean Shift
- Spectral Clustering
- Hierarchical Clustering
- Birch
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- OPTICS (Ordering Points To Identify Clustering Structure)
- GMM (Gaussian Mixture Models)

## Additional Methods
- FINCH (First Integer Neighbor Clustering Hierarchy)
  - [paper](https://dreamhomes.github.io/posts/202005202124.html)
  - [code](https://github.com/ssarfraz/FINCH-Clustering)
- LSC (Large Scale Spectral Clustering with Landmark-Based Representation)
  - [paper](https://arxiv.org/abs/1706.03762)
  - [code](https://github.com/int8/Large-Spectral-Clustering)
  - [code](https://github.com/cylindricalcow/FastSpectralClustering)
- Leiden ( Leiden Clustering Algorithm) (single-cell clustering analysis)
  - [paper](https://www.nature.com/articles/s41598-019-41695-z)
  - [code](https://github.com/vtraag/leidenalg)
  - [docs](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html)
  - [docs](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html)
- Louvain (Louvain Algorithm) (single-cell clustering analysis)
  - [paper](https://arxiv.org/abs/0803.0476)
  - [code](https://github.com/patapizza/pylouvain)
  - [docs](https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.louvain.html)
  - [docs](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html)

## References

scikit-learn:
``` bibtex
@article{pedregosa2011scikit,
  title={Scikit-learn: Machine Learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={The Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011},
  publisher={JMLR. org}
}
```

SCANPY:
``` bibtex
@article{wolf2018scanpy,
  title={SCANPY: Large-Scale Single-Cell Gene Expression Data Analysis},
  author={Wolf, F Alexander and Angerer, Philipp and Theis, Fabian J},
  journal={Genome biology},
  volume={19},
  number={1},
  pages={15},
  year={2018},
  publisher={Springer}
}
```

Leiden:
``` bibtex
@article{traag2019louvain,
  title={From Louvain to Leiden: Guaranteeing Well-Connected Communities},
  author={Traag, Vincent A and Waltman, Ludo and Van Eck, Nees Jan},
  journal={Scientific Reports},
  volume={9},
  number={1},
  pages={1--12},
  year={2019},
  publisher={Nature Publishing Group}
}
```

Louvain:
``` bibtex
@article{blondel2008fast,
  title={Fast Unfolding of Communities in Large Networks},
  author={Blondel, Vincent D and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne},
  journal={Journal of Statistical Mechanics: Theory and Experiment},
  volume={2008},
  number={10},
  pages={P10008},
  year={2008}
}
```

LSC:
``` bibtex
@inproceedings{chen2011large,
  title={Large Scale Spectral Clustering with Landmark-Based Representation},
  author={Chen, Xinlei and Cai, Deng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={25},
  number={1},
  pages={313--318},
  year={2011}
}
```

Spectral Clustering:
``` bibtex
@article{ng2001spectral,
  title={On Spectral Clustering: Analysis and an Algorithm},
  author={Ng, Andrew and Jordan, Michael and Weiss, Yair},
  journal={Advances in Neural Information Processing Systems},
  volume={14},
  year={2001}
}
```

KMeans:
``` bibtex
@article{lloyd1982least,
  title={Least Squares Quantization in PCM},
  author={Lloyd, Stuart},
  journal={IEEE Transactions on Information Theory},
  volume={28},
  number={2},
  pages={129--137},
  year={1982},
  publisher={IEEE}
}
```

Hierarchical Clustering: (Agglomerative Clustering)
``` bibtex
@article{murtagh2012algorithms,
  title={Algorithms for Hierarchical Clustering: An Overview},
  author={Murtagh, Fionn and Contreras, Pedro},
  journal={WIREs Data Mining and Knowledge Discovery},
  volume={2},
  number={1},
  pages={86--97},
  year={2012}
}
```

DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
``` bibtex
@article{ester1996density,
  title={A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise},
  author={Ester, Martin and Kriegel, Hans-Peter and Sander, Jörg and Xu, Xiaowei},
  journal={KDD},
  volume={96},
  number={34},
  pages={226--231},
  year={1996}
}
```

Birch (Balanced Iterative Reducing and Clustering using Hierarchies):
``` bibtex
@article{zhang1996birch,
  title={BIRCH: An Efficient Data Clustering Method for Very Large Databases},
  author={Zhang, Tian and Ramakrishnan, Raghu and Livny, Miron},
  journal={ACM SIGMOD Record},
  volume={25},
  number={2},
  pages={103--114},
  year={1996},
  publisher={ACM New York, NY, USA}
}
```

GMM (Gaussian Mixture Models):
``` bibtex
@book{mclachlan2000finite,
  title={Finite mixture models},
  author={McLachlan, Geoffrey J and Peel, David},
  year={2000},
  publisher={John Wiley \& Sons}
}
```

