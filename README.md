# :trophy: 1st place solution for CrossMoDA 2023 challenge  
 [![arXiv](https://img.shields.io/badge/arXiv-2311.12437-blue)]([https://arxiv.org/abs/2203.08483](https://arxiv.org/pdf/2311.12437.pdf)) [![video](https://img.shields.io/badge/video-Workshop-red)](xx) [![cite](https://img.shields.io/badge/cite-BibTex-yellow)]([](https://scholar.googleusercontent.com/scholar.bib?q=info:9zj1AnVBBtUJ:scholar.google.com/&output=citation&scisdr=ClExUpxMEK2Z_AvZk6o:AFWwaeYAAAAAZZ3fi6pJgXOulOYxyYEV66GqNAM&scisig=AFWwaeYAAAAAZZ3fi-guUbVecsV1zrygH7CdJjE&scisf=4&ct=citation&cd=-1&hl=en)) [![website](https://img.shields.io/badge/Challenge%20website-50d13d)](https://www.synapse.org/#!Synapse:syn51236108/wiki/621615)
 
This is the implementation for the paper:
[Learning site-specific styles for multi-institutional unsupervised cross-modality domain adaptation](https://arxiv.org/pdf/2311.12437.pdf)

### Problem formulation: unsupervised domain adaptation (UDA)
In the CrossMoDA challenge, participants are provided with the **labeled** source domain images (T1 contrast-enhanced MRIs) and the **unlabeled** target domain images (high-resolution T2 MRIs). The goal of this challenge is to obtain a segmentation model for the **target** domain with no manual labeling. In the 2023 edition, the datasets were collected from multiple institutions, and such heterogeous data lead to extra challenges to the UDA problem.

<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_gif1.gif" alt="gif" width="600"/></p>

### Overview of our solution: image-level domain alignment
<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_fig1.png" alt="drawing" width="550"/></p>

#### Step 1: unpaired image translation (our major innovation)
We extended the [QS-Attn](https://github.com/sapphire497/query-selected-attention) to 3D, and modified the generator to a dynamic network. The dynamic network can generate controllable output styles by conditioning on a one-hot site code. Details and the codes for image synthesis are provided in the '[**synthesis**](https://github.com/han-liu/crossmoda2023/tree/main/synthesis)' folder.
#### Step 2: train only with synthetic images
We used [nnU-Netv2](https://github.com/MIC-DKFZ/nnUNet) for segmentation tasks. 
We created a customized trainer and designed two augmentation techniques to augment the local intensities of the structures-of-interest. Our customized trainer and the augmentation script are provided in the '[**segmentation**](https://github.com/han-liu/crossmoda2023/tree/main/segmentation)' folder. 

Once you download the nnU-Net repo, you need to
- add the 'challenge' folder to the nnUNet/nnunetv2/training/nnUNetTrainer/variants
- replace the 'masking.py' in nnUNet/nnunetv2/training/data_augmentation/custom_transforms.

#### Step 3: self-training
Real target domain images are included for training to further reduce the domain gap. Here, you can simply use the model obtained by step 2 to make inference on unlabeled target domain images. Then re-train the nnU-Net using both the synthetic imaegs (w/ real labels) and real images (w/ pseudo labels). For our solution, we simply filter out the poor pseudo labels by connected component analysis. Feel free to explore more advanced pseudo label filtering methods!

### Challenge results
<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_fig3.png" alt="results" width="550"/></p>


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
