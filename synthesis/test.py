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


    
def DataTransform(code):
    return Compose([
        LoadImaged(keys=['A', 'A_msk']),
        AddCoded(keys=['A'], code=code),
        AddChanneld(keys=['A', 'A_msk']),
        NormalizeForegroundd(keys=['A']),
        ScaleIntensityRangePercentilesd(keys=['A'], lower=0, upper=99.9, b_min=-1, b_max=1, clip=True, relative=False),
        crossmodaTransformd(keys=['A']),
        CastToTyped(keys=['A'], dtype=np.float32),
        ToTensord(keys=['A']),])


class AddCoded(MapTransform):
    def __init__(self, keys, code)-> None:
        MapTransform.__init__(self, keys)
        self.keys = keys
        self.code = code

    def __call__(self, data):
        data['code'] = torch.tensor(self.code)
        return data


def load_model(opt):
    opt.num_threads = 0  
    opt.batch_size = 1  
    opt.serial_batches = True
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()
    return model


if __name__ == '__main__':
    opt = TestOptions().parse()
    model = load_model(opt).netG
    A_paths = sorted(glob(opt.image_dir + '/*.nii.gz'))
    transform = DataTransform(code=opt.code)
    save_dir = osp.join(opt.checkpoints_dir, opt.name, 'result')
    tag = '_'.join([str(f) for f in opt.code])
    save_img_dir = osp.join(save_dir, f'fakeB_{tag}')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)

    with torch.no_grad():
        with tqdm(total=len(A_paths)) as pbar:
            for A_path in A_paths:
                data = {'A': A_path, 'A_msk': A_path.replace('Images', 'Labels')}
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
                        output_dir=f"{save_img_dir}", 
                        output_postfix='', 
                        output_ext='.nii.gz', 
                        resample=False,
                        separate_folder=False,
                        print_log=False)])(data),

                pbar.update(1)