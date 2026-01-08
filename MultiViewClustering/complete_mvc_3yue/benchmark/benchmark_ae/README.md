# Benchmark methods based on AutoEncoder (AE)

Collection of Some Useful Github Repositories for Autoencoders:
- https://github.com/clementchadebec/benchmark_VAE
- https://github.com/AgatheSenellart/MultiVae
- https://huggingface.co/amaye15/autoencoder
- https://github.com/AntixK/PyTorch-VAE
- https://github.com/GoyalLab/ExPert/vae/src/modules/zinb.py
- https://github.com/GoyalLab/ExPert
- https://github.com/Kaixhin/Autoencoders
- https://github.com/YixinChen-AI/CVAE-GAN-zoos-PyTorch-Beginner
- https://github.com/jleechung/ZB4171-project
- https://github.com/rasbt/deeplearning-models
- https://modelzoo.co/
- https://github.com/RuiShu/vae-clustering
- https://github.com/jariasf/GMVAE


## Reducing the dimensionality of data with neural networks (AE)
- Paper: [https://www.cs.toronto.edu/~hinton/absps/science.pdf](https://www.cs.toronto.edu/~hinton/absps/science.pdf)
- Code: -
- Citation:
``` bibtex
@article{hinton2006reducing,
  title={Reducing the dimensionality of data with neural networks},
  author={Hinton, Geoffrey E and Salakhutdinov, Ruslan R},
  journal={science},
  volume={313},
  number={5786},
  pages={504--507},
  year={2006},
  publisher={American Association for the Advancement of Science}
}
```

## Extracting and Composing Robust Features with Denoising Autoencoders (DAE)
- Paper: [https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
- Code: -
- Citation:
``` bibtex
@inproceedings{vincent2008extracting,
  title={Extracting and composing robust features with denoising autoencoders},
  author={Vincent, Pascal and Larochelle, Hugo and Bengio, Yoshua and Manzagol, Pierre-Antoine},
  booktitle={Proceedings of the 25th international conference on Machine learning},
  pages={1096--1103},
  year={2008}
}
```

## Auto-Encoding Variational Bayes (VAE)
- Paper: [https://openreview.net/forum?id=33X9fd2-9FyZd](https://openreview.net/forum?id=33X9fd2-9FyZd)
- Code: -
- Citation:
``` bibtex
@article{kingma2013auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}
```

## Deep Generative Modeling for Single-Cell Transcriptomics (scVI, ZINBVAE)
"Zero-inflated negative binomial likelihoods were first systematically integrated into a variational autoencoder framework for scRNA-seq data in scVI (Lopez et al., 2018). Subsequent works such as DCA adopted ZINB as a reconstruction loss for denoising purposes."
- Paper: [https://www.nature.com/articles/s41592-018-0229-2](https://www.nature.com/articles/s41592-018-0229-2)
- Code: [https://github.com/romain-lopez/scVI-reproducibility](https://github.com/romain-lopez/scVI-reproducibility)
- Citation:
``` bibtex
@article{lopez2018deep,
  title={Deep generative modeling for single-cell transcriptomics},
  author={Lopez, Romain and Regier, Jeffrey and Cole, Michael B and Jordan, Michael I and Yosef, Nir},
  journal={Nature methods},
  volume={15},
  number={12},
  pages={1053--1058},
  year={2018},
  publisher={Nature Publishing Group US New York}
}
```

## Single-cell RNA-seq Denoising Using a Deep Count Autoencoder (DCA, ZINBAE)
- Paper: [https://www.nature.com/articles/s41467-018-07931-2](https://www.nature.com/articles/s41467-018-07931-2)
- Code: [https://github.com/theislab/dca](https://github.com/theislab/dca)
- Citation:
``` bibtex
@article{eraslan2019single,
  title={Single-cell RNA-seq denoising using a deep count autoencoder},
  author={Eraslan, G{\"o}kcen and Simon, Lukas M and Mircea, Maria and Mueller, Nikola S and Theis, Fabian J},
  journal={Nature communications},
  volume={10},
  number={1},
  pages={390},
  year={2019},
  publisher={Nature Publishing Group UK London}
}
```

## Uncertainty-aware Multi-view Representation Learning (DUANET)
- Paper: [https://ojs.aaai.org/index.php/AAAI/article/view/16924](https://ojs.aaai.org/index.php/AAAI/article/view/16924)
- Code: -
- Citation:
``` bibtex
@inproceedings{geng2021uncertainty,
  title={Uncertainty-aware multi-view representation learning},
  author={Geng, Yu and Han, Zongbo and Zhang, Changqing and Hu, Qinghua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={9},
  pages={7545--7553},
  year={2021}
}
```

## Multi-VAE: Learning Disentangled View-Common and View-Peculiar Visual Representations for Multi-View Clustering (Multi-VAE)
- Paper: [https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Multi-VAE_Learning_Disentangled_View-Common_and_View-Peculiar_Visual_Representations_for_Multi-View_ICCV_2021_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Multi-VAE_Learning_Disentangled_View-Common_and_View-Peculiar_Visual_Representations_for_Multi-View_ICCV_2021_paper.pdf)
- Code: [https://github.com/SubmissionsIn/Multi-VAE](https://github.com/SubmissionsIn/Multi-VAE)
- Citation:
``` bibtex
@InProceedings{Xu_2021_ICCV,
  author    = {Xu, Jie and Ren, Yazhou and Tang, Huayi and Pu, Xiaorong and Zhu, Xiaofeng and Zeng, Ming and He, Lifang},
  title= {Multi-{VAE}: Learning Disentangled View-Common and View-Peculiar Visual Representations for Multi-View Clustering},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  pages     = {9234-9243}
}