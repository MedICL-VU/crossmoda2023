# :trophy: 1st place solution for CrossMoDA 2023 challenge  
 [![arXiv](https://img.shields.io/badge/arXiv-2311.12437-blue)]([https://arxiv.org/abs/2203.08483](https://arxiv.org/pdf/2311.12437.pdf)) [![video](https://img.shields.io/badge/video-Workshop-red)](xx) [![website](https://img.shields.io/badge/Challenge%20website-50d13d)](https://www.synapse.org/#!Synapse:syn51236108/wiki/621615)
 
This is the implementation for the paper:
[Learning site-specific styles for multi-institutional unsupervised cross-modality domain adaptation](https://arxiv.org/pdf/2311.12437.pdf)

### Problem formulation: unsupervised domain adaptation (UDA)
In the CrossMoDA challenge, participants are provided with the **labeled** source domain images (T1 contrast-enhanced MRIs) and the **unlabeled** target domain images (high-resolution T2 MRIs). The goal of this challenge is to obtain a segmentation model for the **target** domain with no manual labeling. In the 2023 edition, the datasets were collected from multiple institutions, and such heterogeous data lead to extra challenges to the UDA problem.
![gif](https://github.com/han-liu/crossmoda2023/blob/main/vandy365_gif1.gif)

### Overview of our solution
![overview](https://github.com/han-liu/crossmoda2023/blob/main/vandy365_fig1.png)


If you find our code/paper helpful for your research, please kindly consider citing our work:
```
@article{liu2023learning,
  title={Learning Site-specific Styles for Multi-institutional Unsupervised Cross-modality Domain Adaptation},
  author={Liu, Han and Fan, Yubo and Xu, Zhoubing and Dawant, Benoit M and Oguz, Ipek},
  journal={arXiv preprint arXiv:2311.12437},
  year={2023}
}
```
If you have any questions, feel free to contact han.liu@vanderbilt.edu or open an Issue in this repo. 

