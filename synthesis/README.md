# Step 1: unpaired image translation with dynamic network
Our synthesis network is used for 3D unpaired image translation, with controllable output styles by conditioning on a one-hot code.

<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_fig2.png" alt="method" width="600"/></p>

## Data preparation
CrossMoDA 2023 dataset can be officially downloaded [here](https://www.synapse.org/#!Synapse:syn51236108/wiki/621732).

Once the dataset is downloaded, we perform the following preprocessing steps:
- Resample to the median resolution of the dataset, i.e., 0.41 x 0.41 x 1 mm<sup>3</sup>
- Based on the cochlea locations, crop the images to 256 x 144 x 32 sub-volumes.

<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_preprocess.png" alt="preprocess" width="600"/></p>

Once the preprocessing is done, we need to arrange the data in the following format.
Note that A and B correspond to the source and target domain, respectively. The filenames in the ImagesA and LabelsA should be the same. EdgesA stores the weighting maps used for re-weighting the edge loss. If EdgesA is not provided, the edge loss will simply treat each voxel equally.

```
Data/
|-----ImagesA
|-----ImagesB
|-----LabelsA
|-----EdgesA (optional)
```

For source domain, ImagesA and LabelsA, the filenames should be arranged as follows.
```
ImagesA/                                         LabelsA/
|-----crossmoda2023_etz_1.nii.gz                 |-----crossmoda2023_etz_1.nii.gz
|-----crossmoda2023_etz_2.nii.gz                 |-----crossmoda2023_etz_2.nii.gz
|-----...                                        |-----...
|-----crossmoda2023_ldn_1.nii.gz                 |-----crossmoda2023_ldn_1.nii.gz
|-----crossmoda2023_ldn_2.nii.gz                 |-----crossmoda2023_ldn_2.nii.gz
|-----...                                        |-----...
|-----crossmoda2023_ukm_1.nii.gz                 |-----crossmoda2023_ukm_1.nii.gz
|-----crossmoda2023_ukm_2.nii.gz                 |-----crossmoda2023_ukm_2.nii.gz
|-----...                                        |-----...
```

For target domain, ImagesB, the filename should include a **tag** that specifies the site information (or any other categorical information), which will be taken by the network as the condition to generate controllable styles. In our example, the tags are 'etz', 'ldn', and 'ukm'.
```
ImagesB/
|-----crossmoda2023_etz_106.nii.gz
|-----crossmoda2023_etz_107.nii.gz
|-----...
|-----crossmoda2023_ldn_80.nii.gz
|-----crossmoda2023_ldn_81.nii.gz
|-----...
|-----crossmoda2023_ukm_44.nii.gz
|-----crossmoda2023_ukm_45.nii.gz
|-----...
```

# Create a virtual environment using Anaconda
Here we create a virtual environment called 'crossmoda2023'
```
conda create -n crossmoda2023 python=3.9 -y
conda activate crossmoda2023
```

## Train
```
python train.py -n YourExperimentName -d /Data 
```

## Test
```
python test.py -n YourExperimentName --code 0 0 1 -i /Data/ImagesA
```

## Playground for controllable output style
If you just want to play with our pre-trained model to have fun with the controllable styles, here we are!

### Step 1: download the sample data and the pre-trained model to your local machine
I have prepared several preprocessed data and our pre-trained model in this '[**sample**](xx)' folder. Once you finish downloading this folder, you need to move the 'pretrained' folder to the checkpoint directory.

### Step 2: time to play!
You can either generate the known **site-specific** styles using the codes that the network has been trained with, i.e., (0, 0, 1), (0, 1, 0), (1, 0, 0),
or any codes you like (even some crazy ones) to generate some **unknown** T2 styles. 

You will find that different codes will only have impacts on the image style, not the anatomy! Disentanglement, huh?

```
python test.py -n pretrained --src_image SourceImagePath --src_label SourceLabelPath --save_dir OutputFolder --code 1 2 3
```
Some examples are shown blow:
<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_playground.png" alt="playground" width="600"/></p>

