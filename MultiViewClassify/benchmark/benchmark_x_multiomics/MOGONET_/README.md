# MOGONET

MOGONET (Multi-Omics Graph cOnvolutional NETworks) is a novel multi-omics data integrative analysis framework for classification tasks in biomedical applications.

https://github.com/txWang/MOGONET

<sup>Illustration of MOGONET. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. MOGONET combines GCN for multi-omics specific learning and VCDN for multi-omics integration. For clear and concise illustration, an example of one sample is chosen to demonstrate the VCDN component for multi-omics integration. Pre-processing is first performed on each omics data type to remove noise and redundant features. Each omics-specific GCN is trained to perform class prediction using omics features and the corresponding sample similarity network generated from the omics data. The cross-omics discovery tensor is calculated from the initial predictions of omics-specific GCNs and forwarded to VCDN for final prediction. MOGONET is an end-to-end model and all networks are trained jointly.<sup>


### Usage

```bash
$ python main.py --data_dir ./data/BRCA --output_dir ./result/BRCA
$ python main.py --data_dir ./data/ROSMAP --output_dir ./result/ROSMAP
```

### Citation

```bibtex
@article{wang2021mogonet,
  title={MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification},
  author={Wang, Tongxin and Shao, Wei and Huang, Zhi and Tang, Haixu and Zhang, Jie and Ding, Zhengming and Huang, Kun},
  journal={Nature communications},
  volume={12},
  number={1},
  pages={3445},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
```