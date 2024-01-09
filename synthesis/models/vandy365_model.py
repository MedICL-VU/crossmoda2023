import numpy as np
import torch
from monai.networks.utils import one_hot
from monai.losses import DiceCELoss
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from .segnet import UNet


"""##################################################################################

Our team 'Vandy365' achieved 1st place in CrossMoDA 2023 challenge.
Leaderboard available: https://www.synapse.org/#!Synapse:syn51236108/wiki/624029

Paper [https://arxiv.org/abs/2311.12437]: "Learning Site-specific Styles for Multi-institutional 
Unsupervised Cross-modality Domain Adaptation"

Here is the implementation of our synthesis network for 3D unpaired image translation.

If you find our paper/code useful, please consider citing our work:

@article{liu2023learning,
  title={Learning Site-specific Styles for Multi-institutional Unsupervised Cross-modality Domain Adaptation},
  author={Liu, Han and Fan, Yubo and Xu, Zhoubing and Dawant, Benoit M and Oguz, Ipek},
  journal={arXiv preprint arXiv:2311.12437},
  year={2023}
}

"""##################################################################################


class Vandy365Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_segA', type=float, default=0.5, help='weight for DiceCE loss for A segmentation')
        parser.add_argument('--lambda_segB', type=float, default=0, help='weight for DiceCE loss for B segmentation')
        parser.add_argument('--lambda_sobel', type=float, default=1.0, help='weight for edge loss')
        parser.add_argument('--nce_idt', type=util.str2bool, default=True, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance', type=util.str2bool, nargs='?', const=True, default=False, 
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0) 
        opt, _ = parser.parse_known_args()
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'DiceA', 'DiceB', 'sobelL1']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'real_A_mask', 'seg_A', 'seg_B', 'fake_sobel', 'real_sobel', 'edge_mask']

        if self.opt.segB:
            self.loss_names.append('DiceB')
            self.visual_names.append('seg_B')

        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.dce_loss = DiceCELoss(to_onehot_y=False, softmax=True)

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D', 'S']
        else:
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, 
            opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        
        self.netF = networks.define_F(
            opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, 
            opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        # This small U-Net is adapted from https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1
        """Hmmm, you may need to adjust (1) the num_pool, (2) pool_op_kernel_sizes, (3) conv_kernel_sizes
        for your own application because the size of your 3D image might be different from mine, i,e., 256 x 144 x 8
        """
        self.netS = UNet(
            input_channels=1, 
            base_num_features=16, 
            num_classes=opt.seg_output_nc, 
            num_pool=5, 
            pool_op_kernel_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 1, 1]], 
            conv_kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], 
            deep_supervision=False).cuda()

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, 
                opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.optimizer_G = torch.optim.Adam(list(self.netG.parameters())+list(self.netS.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self):
        bs_per_gpu = self.real_A.size(0) // len(self.opt.gpu_ids)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     
        if self.opt.isTrain:
            self.backward_D()                  
            self.backward_G()                   
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()                   
        self.set_requires_grad(self.netD, True) 
        self.optimizer_D.zero_grad()     
        self.backward_D()                
        self.optimizer_D.step()          
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.backward_G()                   
        self.optimizer_G.step()             
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_A_mask = input['A_msk' if AtoB else 'B_msk'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.code = input['code'].to(self.device)  
        self.edge_mask = None

        if 'A_edge' in input.keys():
            self.edge_mask = input['A_edge'].to(self.device)


    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake, self.seg = self.netG(self.real, code=torch.cat([self.code,]*self.real.shape[0], 0))
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.seg_A = self.seg[:self.real_A.size(0)]
        self.seg_B = self.netS(self.fake_B)
        self.fake_sobel = networks.sobelLayer(self.fake_B)
        self.real_sobel = networks.sobelLayer(self.real_A).detach()

        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        self.feat_k = self.netG(self.real_A, self.nce_layers, encode_only=True)

    def backward_D(self):
        if self.opt.lambda_GAN > 0.0:
            fake = self.fake_B.detach()
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            pred_real = self.netD(self.real_B)
            loss_D_real_unweighted = self.criterionGAN(pred_real, True)
            self.loss_D_real = loss_D_real_unweighted.mean()
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        else:
            self.loss_D_real, self.loss_D_fake, self.loss_D = 0.0, 0.0, 0.0

    def backward_G(self):
        fake = self.fake_B
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_SegA = self.calculate_seg_loss(self.seg_A, self.real_A_mask, self.opt.seg_output_nc) * self.opt.lambda_segA
        self.loss_SegB = self.calculate_seg_loss(self.seg_B, self.real_A_mask, self.opt.seg_output_nc) * self.opt.lambda_segB 
        self.loss_sobelL1 = self.calculate_edge_loss(self.fake_sobel, self.real_sobel, self.edge_mask) * self.opt.lambda_sobel  
        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_SegA + self.loss_SegB + self.loss_sobelL1
        self.loss_G.backward()

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids, attn_mats = self.netF(feat_k, self.opt.num_patches, None, None)
        feat_q_pool, _, _ = self.netF(feat_q, self.opt.num_patches, sample_ids, attn_mats)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_seg_loss(self, src, tgt, num_classes):
        tgt = one_hot(tgt, num_classes, dim=1)
        return self.dce_loss(src, tgt)
    
    def calculate_edge_loss(self, fake_sobel, real_sobel, weight_map=None):
        if weight_map is None:
            edge_loss = self.criterionL1(fake_sobel, real_sobel)
        else:
            edge_loss = self.criterionL1(fake_sobel * weight_map, real_sobel* weight_map)
        return edge_loss
  
    def update_lambda(self, epochNum):
        if epochNum <= 50:
            self.opt.lambda_segB = 0
        else:
            self.opt.lambda_segB = 0.5
        print('update segB lambda: %f' % (self.opt.lambda_segB))