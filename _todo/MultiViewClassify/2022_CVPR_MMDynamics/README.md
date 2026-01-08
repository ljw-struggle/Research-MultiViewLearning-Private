## MMDynamics

This repository contains the code of our CVPR'2022 paper Multimodal Dynamics: Dynamical Fusion for Trustworthy Multimodal Classification. The data used in the paper can be obtained through the following links https://github.com/txWang/MOGONET. The code is implemented on the code provided by MOGONET. 


### Usage

```bash
$ python main.py --data_dir ./data/BRCA --output_dir ./result/BRCA
$ python main.py --data_dir ./data/ROSMAP --output_dir ./result/ROSMAP
```

### Citation

```bibtex
@inproceedings{han2022multimodal,
  title={Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification},
  author={Han, Zongbo and Yang, Fan and Huang, Junzhou and Zhang, Changqing and Yao, Jianhua},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={20707--20717},
  year={2022}
}
```