# Step 1: unpaired image translation with dynamic network
Our synthesis network is used for 3D unpaired image translation, with controllable output styles by conditioning on a one-hot code.

<p align="center"><img src="https://github.com/han-liu/crossmoda2023/blob/main/figs/vandy365_fig2.png" alt="drawing" width="600"/></p>

## Data preparation
A and B correspond to the source and target domain, respectively. The filenames in the ImagesA and LabelsA should be the same. EdgesA stores the weighting maps used for re-weighting the edge loss. If EdgesA is not provided, the edge loss will simply treat each voxel equally.

CrossMoDA 2023 dataset can be officially downloaded [here](https://www.synapse.org/#!Synapse:syn51236108/wiki/621732).

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

For target domain, ImagesB, the filenames should include a **tag** that specifies the site information (or any other categorical information). This information will be used as the condition to the network to generate controllable styles. In our example, the tags are 'etz', 'ldn', and 'ukm'.
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

## Train
```
python train.py -n YourExperimentName -d /Data 
```

## Test
```
python test.py -n YourExperimentName -epoch latest --code 0 0 1 -i /Data/ImagesA
```
