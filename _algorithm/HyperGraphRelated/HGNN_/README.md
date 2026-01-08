# Hypergraph Neural Networks (HGNN)

This work will appear in AAAI 2019. We proposed a novel framework(HGNN) for data representation learning, which could take multi-modal data and exhibit superior performance gain compared with single modal or graph-based multi-modal methods. You can also check our [paper](http://gaoyue.org/paper/HGNN.pdf) for a deeper introduction.

HGNN could encode high-order data correlation in a hypergraph structure. Confronting the challenges of learning representation for complex data in real practice, we propose to incorporate such data structure in a hypergraph, which is more flexible on data modeling, especially when dealing with complex data. In this method, a hyperedge convolution operation is designed to handle the data correlation during representation learning. In this way, traditional hypergraph learning procedure can be conducted using hyperedge convolution operations efficiently. HGNN is able to learn the hidden layer representation considering the high-order data structure, which is a general framework considering the complex data correlations.

In this repository, we release code and data for train a Hypergrpah Nerual Networks for node classification on ModelNet40 dataset and NTU2012 dataset. The visual objects' feature is extracted by [MVCNN(Su et al.)](http://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf) and [GVCNN(Feng et al.)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_GVCNN_Group-View_Convolutional_CVPR_2018_paper.pdf).


PAPER: [HGNN](http://gaoyue.org/paper/HGNN.pdf)
GITHUB: [HGNN](https://github.com/iMoonLab/HGNN)
