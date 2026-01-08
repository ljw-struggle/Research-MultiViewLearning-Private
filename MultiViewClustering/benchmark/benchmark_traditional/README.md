# Traditional Clustering Methods
Comparison of traditional clustering methods: [Link](./sklearn_clustering.ipynb)

Comparison of different clustering algorithms in Scikit-learn: [Link](https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py)
Comparison of different clustering algorithms in Scanpy: [Link](https://scanpy.readthedocs.io/)

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
  title={Scikit-learn: Machine learning in Python},
  author={Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal={the Journal of machine Learning research},
  volume={12},
  pages={2825--2830},
  year={2011},
  publisher={JMLR. org}
}
```

SCANPY:
``` bibtex
@article{wolf2018scanpy,
  title={SCANPY: large-scale single-cell gene expression data analysis},
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
  title={From Louvain to Leiden: guaranteeing well-connected communities},
  author={Traag, Vincent A and Waltman, Ludo and Van Eck, Nees Jan},
  journal={Scientific reports},
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
  title={Fast unfolding of communities in large networks},
  author={Blondel, Vincent D and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne},
  journal={Journal of statistical mechanics: theory and experiment},
  volume={2008},
  number={10},
  pages={P10008},
  year={2008}
}
```

LSC:
``` bibtex
@inproceedings{chen2011large,
  title={Large scale spectral clustering with landmark-based representation},
  author={Chen, Xinlei and Cai, Deng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={25},
  number={1},
  pages={313--318},
  year={2011}
}
```