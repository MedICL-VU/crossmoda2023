# Dynamic network for unpaired image translation
Our synthesis network is used for 3D unpaired image translation. It can generate controllable output styles by conditioning on a one-hot code.

## Data preparation
A and B correspond to the source and target domain, respectively. The filenames in the ImagesA and LabelsA should be the same. EdgesA stores the weighting maps used for re-weighting the edge loss. If EdgesA is not provided, the edge loss will simply treat each voxel equally.

```
Data/
|-----ImagesA
|-----ImagesB
|-----LabelsA
|-----EdgesA (optional)
```

In the target domain folder, ImagesB, the filenames should include a **tag** that specifies the site information (or any other categorical information). This information will be used as the condition to the network to generate controllable styles. In our example, the tags are 'etz', 'ldn', and 'ukm'.
```
ImagesB/
|-----crossmoda2023_etz_106
|-----crossmoda2023_etz_107
|-----...
|-----crossmoda2023_ldn_80
|-----crossmoda2023_ldn_81
|-----...
|-----crossmoda2023_ukm_44
|-----crossmoda2023_ukm_45
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
