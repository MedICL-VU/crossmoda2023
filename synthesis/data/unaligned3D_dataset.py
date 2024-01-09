import os.path as osp
from glob import glob
from data.base_dataset import BaseDataset
import random
import numpy as np
import torch
from monai.transforms import *


class Unaligned3DDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self._prepare_paths(opt)
        self.A_size = len(self.A_img_paths)  
        self.B_size = len(self.B_img_paths)
        self.transform = DataTransform(opt.crop_size)

    def _prepare_paths(self, opt):
        self.dir_A = osp.join(opt.dataroot, 'ImagesA') 
        self.dir_B = osp.join(opt.dataroot, 'ImagesB') 
        self.A_img_paths = sorted(glob(self.dir_A + '/*.nii.gz'))
        self.A_msk_paths = [path.replace('Images', 'Labels') for path in self.A_img_paths]   
        self.B_img_paths = sorted(glob(self.dir_B + '/*.nii.gz'))        
        self.A_edge_paths = None

        if osp.exists(osp.join(opt.dataroot, 'EdgesA')):
            self.A_edge_paths = [path.replace('Images', 'Edges') for path in self.A_img_paths]

    def __getitem__(self, index):
        A_idx = index % self.A_size
        A_img_path = self.A_img_paths[A_idx] 
        A_msk_path = self.A_msk_paths[A_idx] 

        if self.opt.serial_batches:   
            index_B = index % self.B_size
        else:   
            index_B = random.randint(0, self.B_size - 1)
        B_img_path = self.B_img_paths[index_B]

        data_dict = {
            'A': A_img_path, 
            'B': B_img_path, 
            'A_msk': A_msk_path, 
        }

        if self.A_edge_paths:
            data_dict['A_edge'] = self.A_edge_paths[A_idx]
        return self.transform.train(data_dict)

    def __len__(self):
        return max(self.A_size, self.B_size)


class DataTransform(object):
    def __init__(self, crop_size=(256, 144, 8)):
        self.train = Compose([        
            GetCoded(keys='B'),
            LoadImaged(keys=['A', 'B', 'A_msk', 'A_edge'], allow_missing_keys=True),
            AddChanneld(keys=['A', 'B', 'A_msk', 'A_edge'], allow_missing_keys=True),
            RandAdjustContrastd(keys=['A'], prob=0.4, gamma=(0.8, 1.2)),
            RandAdjustContrastd(keys=['B'], prob=0.1, gamma=(0.8, 1.2)), # smaller prob for B to preserve site-specific styles
            NormalizeForegroundd(keys=['A', 'B']),
            ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
            ScaleIntensityRangePercentilesd(keys=['B'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
            crossmodaTransformd(keys=['A']),  # <-------------- Customize for your own dataset; you can simply remove this line :)
            RandFlipd(keys=['A', 'A_msk', 'A_edge'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=['B'], prob=0.5, spatial_axis=0),
            RandSpatialCropd(keys=['A', 'B', 'A_msk', 'A_edge'], roi_size=crop_size, 
                random_center=True, random_size=False, allow_missing_keys=True),
            ReWeightEdged(keys='A_edge'),
            CastToTyped(
                keys=['A', 'B', 'A_msk', 'A_edge'], 
                dtype=(np.float32, np.float32, np.uint8, np.uint8), 
                allow_missing_keys=True),
            ToTensord(keys=['A', 'B', 'A_msk', 'A_edge'], allow_missing_keys=True),])


###############################################################################
# Customized data transformations
###############################################################################


class GetCoded(MapTransform):
    """ Yup, we hard-code this part for now.
    """
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.key = keys

    def __call__(self, data):
        if 'ukm' in data[self.key]:
            data['code'] = torch.tensor([0,0,1])
        elif 'etz' in data[self.key]:
            data['code'] = torch.tensor([0,1,0])
        elif 'ldn' in data[self.key]:
            data['code'] = torch.tensor([1,0,0])
        return data


class NormalizeForegroundd(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        for k in self.keys:
            img = data[k]
            _mean = img[img!=0].mean()
            _std = img[img!=0].mean()
            data[k] = (data[k] - _mean)/_std
        return data


class ReWeightEdged(MapTransform):
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.key = keys

    def __call__(self, data):
        if self.key in data.keys():
            data['A_edge'][data['A_edge']==2] = 10
            data['A_edge'][data['A_edge']==1] = 2 
        return data


class crossmodaTransformd(MapTransform):
    """We empirically found this transform useful for CrossMoDA dataset
    If you are running this code for other datasets, feel free to simply
    turn off this transform, or modify it based on your need.
    """
    def __init__(self, keys) -> None:
        MapTransform.__init__(self, keys)
        self.keys = keys

    def __call__(self, data):
        data['A'][data['A_msk']==3] = 1  # cochleae enhancement
        data['A'][(data['A_msk']==1) | (data['A_msk']==2)] -= (data['A'][(data['A_msk']==1) | (data['A_msk']==2)].mean() + 0.5)
        return data