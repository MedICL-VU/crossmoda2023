from typing import Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .nnUNetTrainer_crossmoda import nnUNetTrainer_crossmoda
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainer_FT(nnUNetTrainer_crossmoda):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 500
        self.current_epoch = 0

    @classmethod
    def build_network_architecture(
        cls, plans_manager, dataset_json, configuration_manager, num_input_channels, 
        enable_deep_supervision) -> nn.Module:
        network = get_network_from_plans(
            plans_manager, dataset_json, configuration_manager,
            num_input_channels, deep_supervision=enable_deep_supervision)
        # breakpoint()
        fname = '/data/nnUNetV2/data/nnUNet_results/Dataset511_MoreOS3/nnUNetTrainer_FT__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth'
        saved_model = torch.load(fname)
        pretrained_dict = saved_model['network_weights']
        network.load_state_dict(pretrained_dict)   
        return network  


class nnUNetTrainer_FT1(nnUNetTrainer_FT):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 500
        self.current_epoch = 0

    def compute_loss(self, output, target):
        if self.network.training:
            # highest probability
            cutoff = 0.4  # only the top 60% voxels are used for training.
            for i in range(len(output)):
                prob = nn.Softmax(dim=1)(output[i])
                max_val = torch.amax(prob, 1)  # B x H x W x D
                thresholds = torch.quantile(max_val.view(max_val.shape[0], -1), cutoff, 1)
                mask = max_val >= (thresholds[..., None, None, None] * torch.ones(max_val.shape).cuda())
                output[i] *= mask[:, None, ...]
                target[i] *= mask[:, None, ...]
        loss = self.loss(output, target)
        return loss



class nnUNetTrainer_FT2(nnUNetTrainer_FT):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 500
        self.current_epoch = 0
    
    def compute_loss(self, output, target):
        if self.network.training:
            # compute entropy map
            cutoff = 0.6
            for i in range(len(output)):
                prob = nn.Softmax(dim=1)(output[i])
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # uncertainty # B x H x W x D
                thresholds = torch.quantile(entropy.view(entropy.shape[0], -1), cutoff, 1) # bigger than this threshold, very uncertain
                mask = entropy <= (thresholds[..., None, None, None] * torch.ones(entropy.shape).cuda())
                output[i] *= mask[:, None, ...]
                target[i] *= mask[:, None, ...]
        loss = self.loss(output, target)
        return loss


class nnUNetTrainer_FT3(nnUNetTrainer_FT):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 500
        self.current_epoch = 0

    def compute_loss(self, output, target):
        if self.network.training:
            # highest probability
            cutoff = 0.2  # only the top 80% voxels are used for training.
            for i in range(len(output)):
                prob = nn.Softmax(dim=1)(output[i])
                max_val = torch.amax(prob, 1)  # B x H x W x D
                thresholds = torch.quantile(max_val.view(max_val.shape[0], -1), cutoff, 1)
                mask = max_val >= (thresholds[..., None, None, None] * torch.ones(max_val.shape).cuda())
                output[i] *= mask[:, None, ...]
                target[i] *= mask[:, None, ...]
        loss = self.loss(output, target)
        return loss



class nnUNetTrainer_FT4(nnUNetTrainer_FT):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 500
        self.current_epoch = 0
    
    def compute_loss(self, output, target):
        if self.network.training:
            # compute entropy map
            cutoff = 0.8
            for i in range(len(output)):
                prob = nn.Softmax(dim=1)(output[i])
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # uncertainty # B x H x W x D
                thresholds = torch.quantile(entropy.view(entropy.shape[0], -1), cutoff, 1) # bigger than this threshold, very uncertain
                mask = entropy <= (thresholds[..., None, None, None] * torch.ones(entropy.shape).cuda())
                output[i] *= mask[:, None, ...]
                target[i] *= mask[:, None, ...]
        loss = self.loss(output, target)
        return loss


class nnUNetTrainer_FT5(nnUNetTrainer_FT):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 500
        self.current_epoch = 0

    def compute_loss(self, output, target):
        if self.network.training:
            # highest probability
            cutoff = 0.1  # only the top 90% voxels are used for training.
            for i in range(len(output)):
                prob = nn.Softmax(dim=1)(output[i])
                max_val = torch.amax(prob, 1)  # B x H x W x D
                thresholds = torch.quantile(max_val.view(max_val.shape[0], -1), cutoff, 1)
                mask = max_val >= (thresholds[..., None, None, None] * torch.ones(max_val.shape).cuda())
                output[i] *= mask[:, None, ...]
                target[i] *= mask[:, None, ...]
        loss = self.loss(output, target)
        return loss



class nnUNetTrainer_FT6(nnUNetTrainer_FT):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 1
        self.num_epochs = 500
        self.current_epoch = 0
    
    def compute_loss(self, output, target):
        if self.network.training:
            # compute entropy map
            cutoff = 0.9
            for i in range(len(output)):
                prob = nn.Softmax(dim=1)(output[i])
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # uncertainty # B x H x W x D
                thresholds = torch.quantile(entropy.view(entropy.shape[0], -1), cutoff, 1) # bigger than this threshold, very uncertain
                mask = entropy <= (thresholds[..., None, None, None] * torch.ones(entropy.shape).cuda())
                output[i] *= mask[:, None, ...]
                target[i] *= mask[:, None, ...]
        loss = self.loss(output, target)
        return loss