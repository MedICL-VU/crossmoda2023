import os
import os.path as osp
from tqdm import tqdm 
import torch
import numpy as np
from glob import glob
from monai.inferers import sliding_window_inference
from monai.transforms import *
from options.test_options import TestOptions
from data import create_dataset
from data.unaligned3D_dataset import NormalizeForegroundd, crossmodaTransformd
from models import create_model
import util.util as util
from test import DataTransform, AddCoded, load_model


if __name__ == '__main__':
    opt = TestOptions().parse()
    assert osp.exists(opt.src_image) and osp.exists(opt.src_label), f'The input path of source domain image/label does not exist...'
    model = load_model(opt).netG
    transform = DataTransform(code=opt.code)
    tag = '_'.join([str(f) for f in opt.code])

    print('\n########################################################################################################')
    print('###################################### Welcome to the playground !######################################')
    print('########################################################################################################\n')

    print('During training, we only trained the model by conditioning on [0, 0, 1], [0, 1, 0], and [1, 0, 0].')
    print('In this playground, you can generate "site-specific" styles by using these codes, or play with any codes (even crazy ones) you like to generate some unseen T2 styles.')
    print('\nStart image translation...')

    with torch.no_grad():
        data = {'A': opt.src_image, 'A_msk': opt.src_label}
        data = transform(data)

        data['A'] = sliding_window_inference(
            inputs=data['A'].unsqueeze(0).cuda(), 
            roi_size=opt.crop_size, 
            sw_batch_size=opt.sw_batch_size, 
            predictor=model,
            overlap=opt.overlap,
            mode='gaussian',
            sigma_scale=0.125,
            padding_mode='constant',
            cval=-1,
            code=data['code'].unsqueeze(0).cuda())[0]

        Compose([
            SqueezeDimd(keys=['A'], dim=0),
            SqueezeDimd(keys=['A'], dim=0),
            SaveImaged(
                keys=['A'], 
                output_dir=f"{opt.save_dir}", 
                output_postfix=f'_{tag}', 
                output_ext='.nii.gz', 
                resample=False,
                separate_folder=False,
                print_log=False)])(data)

    print('Done')
    print(f'Output saved at: {osp.join(opt.save_dir, osp.basename(opt.src_image)[:-7] + f"_{tag}.nii.gz")}\n')
