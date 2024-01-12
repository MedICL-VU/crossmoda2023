# :trophy: 1st place solution for CrossMoDA 2023 challenge  
 [![arXiv](https://img.shields.io/badge/arXiv-2311.12437-blue)](https://arxiv.org/pdf/2311.12437.pdf) [![video](https://img.shields.io/badge/video-Workshop-red)](xx) [![cite](https://img.shields.io/badge/cite-BibTex-yellow)](https://scholar.googleusercontent.com/scholar.bib?q=info:9zj1AnVBBtUJ:scholar.google.com/&output=citation&scisdr=ClExUpxMEI-AnDJEduU:AFWwaeYAAAAAZZ5CbuUPMgRPxcRpotbJifS-0XI&scisig=AFWwaeYAAAAAZZ5Cbonby7wKO8QbhZvpIS096kY&scisf=4&ct=citation&cd=-1&hl=en) [![website](https://img.shields.io/badge/Challenge%20website-50d13d)](https://www.synapse.org/#!Synapse:syn51236108/wiki/621615)
 
This is the implementation for the paper:
[Learning site-specific styles for multi-institutional unsupervised cross-modality domain adaptation](https://arxiv.org/pdf/2311.12437.pdf)

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

### 🔥 Quick start
In this [**playground**](synthesis/README.md#playground), you can download our pre-trained sythesis model and some preprocessed data. You will explore how to generate synthetic T2 images with **controllable** styles!

### CrossMoDA 2023: unsupervised domain adaptation (UDA) meets multi-site data
Participants are provided with the **labeled** source domain images (T1 contrast-enhanced MRIs), and the **unlabeled** target domain images (high-resolution T2 MRIs). The goal of this challenge is to obtain a segmentation model for the **target** domain (T2) with no manual labeling. Particularly, the datasets were collected from **multiple institutions**, and such heterogeous data lead to extra challenges to the UDA problem.

<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_challenge_review.png" alt="intro" width="600"/></p>

### Overview of our solution
Our UDA framework is an **image-level domain alignment** approach and consists of three steps, as shown below. To tackle the data heterogeneity, we aim to make the segmentation model robust to any T2 styles by training the model on images with **diverse** T2 stlyes. 
>Instead of removing site-specific styles by data harmonization, we generate them! 🤓 

<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_fig1.png" alt="overview" width="550"/></p>

#### Step 1: unpaired image translation (our major innovation)
>We performed unpaired image translation to translate T1CE MRIs to T2. We extended the [QS-Attn](https://github.com/sapphire497/query-selected-attention) to 3D, and modified the generator to a dynamic network. The dynamic network can generate controllable output styles by conditioning on a one-hot site code. Details and demos are provided in the '[**synthesis**](https://github.com/han-liu/crossmoda2023/tree/main/synthesis)' folder.

#### Step 2: train only with synthetic images
>We used [nnU-Netv2](https://github.com/MIC-DKFZ/nnUNet) for segmentation tasks. 
We created a customized trainer and designed two augmentation techniques to augment the local intensities of the structures-of-interest. Our customized trainer and the augmentation script are provided in the '[**segmentation**](https://github.com/han-liu/crossmoda2023/tree/main/segmentation)' folder.

>Once you download the nnU-Net repo, you need to
>- add the 'challenge' folder to the nnUNet/nnunetv2/training/nnUNetTrainer/variants
>- replace the 'masking.py' in nnUNet/nnunetv2/training/data_augmentation/custom_transforms.

#### Step 3: self-training
>To further reduce the domain gap, we include real target domain images for training. We use the model obtained by step 2 to generate pseudo labels on the unlabeled target domain images. Then we re-train the nnU-Net using the combined data, i.e., synthetic images (w/ real labels) and real images (w/ pseudo labels). Of course, the quality of pseudo labels matters! In our solution, we simply filter out the poor pseudo labels via connected component analysis. Feel free to explore more advanced pseudo label filtering methods!

### Conditional synthesis results
Each column belongs to the same site/style. From each column on the right panel, we can see that our synthesized images (top 3 rows) match the style of the real image (bottom row) from the same site. 
<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_results.png" alt="results" width="550"/></p>

### Challenge results
<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_fig3.png" alt="results" width="550"/></p>

### Acknowledgements
- NSF grant 2220401
- NIH grant T32EB021937
- We thank the authors of [CUT](https://github.com/taesungp/contrastive-unpaired-translation), [QS-Attn](https://github.com/sapphire497/query-selected-attention), and [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for kindly make their codes publicly accessible.

