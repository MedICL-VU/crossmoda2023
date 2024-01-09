# :trophy: 1st place solution for CrossMoDA 2023 challenge  
 [![arXiv](https://img.shields.io/badge/arXiv-2311.12437-blue)]([https://arxiv.org/abs/2203.08483](https://arxiv.org/pdf/2311.12437.pdf)) [![video](https://img.shields.io/badge/video-Workshop-red)](xx) [![website](https://img.shields.io/badge/Challenge%20website-50d13d)](https://www.synapse.org/#!Synapse:syn51236108/wiki/621615)
 
This is the implementation for the paper:
[Learning site-specific styles for multi-institutional unsupervised cross-modality domain adaptation](https://arxiv.org/pdf/2311.12437.pdf)

### Problem formulation: unsupervised domain adaptation (UDA)
In the CrossMoDA challenge, participants are provided with the **labeled** source domain images (T1 contrast-enhanced MRIs) and the **unlabeled** target domain images (high-resolution T2 MRIs). The goal of this challenge is to obtain a segmentation model for the **target** domain with no manual labeling. In the 2023 edition, the datasets were collected from multiple institutions, and such heterogeous data lead to extra challenges to the UDA problem.

<p align="center">
<img src="https://github.com/han-liu/crossmoda2023/blob/main/vandy365_gif1.gif" alt="gif" width="600"/>
</p>

### Our solution: image-level domain alignment
#### Step 1: unpaired image translation
We extended the [QS-Attn model](https://github.com/sapphire497/query-selected-attention) to 3D and modified the generator to a dynamic network. The dynamic network can generate controllable output styles by conditioning on a one-hot site code. Details and the codes for image synthesis are provided in the 'synthesis' folder.

#### Step 2: train with only synthetic images
We used [nnU-Netv2](https://github.com/MIC-DKFZ/nnUNet) for segmentation tasks. Specifically, we created a customized trainer and designed two intensity augmentation techniques to augment the local intensities of the structures-of-interest. Our customized trainer and the augmentation scripts are provided in the 'segmentation' folder. 

Once you download the nnU-Net repo, you can add the 'challenge' folder to the nnUNet/nnunetv2/training/nnUNetTrainer/variants, and replace the 'masking.py' in nnUNet/nnunetv2/training/data_augmentation/custom_transforms.

<p align="center">
<img src="https://github.com/han-liu/crossmoda2023/blob/main/vandy365_fig1.png" alt="drawing" width="600"/>
</p>

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

