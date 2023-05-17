import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import copy
import os
#import wandb

import PIL

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import cotta_utils

from utils.masking import Masking

from .dnn import DNN
import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

@torch.jit.script
def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    if conf.args.dataset in ['cifar10', 'cifar100']:
        img_shape = (32, 32, 3)
    else:
        img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        cotta_utils.Clip(0.0, 1.0),
        cotta_utils.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        cotta_utils.GaussianNoise(0, gaussian_std),
        cotta_utils.Clip(clip_min, clip_max)
    ])
    return tta_transforms


class CMT(DNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for param in self.net.parameters():
            param.requires_grad = True

        for module in self.net.modules():

            if isinstance(module,nn.BatchNorm1d) or isinstance(module,nn.BatchNorm2d):

                if conf.args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentun = conf.args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):
                module.weight.require_grad_(True)
                module.bias.requires_grad_(True)

        self.mt = conf.args.alpha #0.999 for every dataset
        self.rst = conf.args.restoration_factor_cmt #0.01 for all dataset
        self.ap = conf.args.aug_threshold_cmt #0.92 for CIFAR10, 0.72 for CIFAR100
        self.episodic = False


        self.mbs = conf.args.mask_block_size
        self.mr = conf.args.mask_ratio
        self.cjs = conf.args.mask_color_jitter_s
        self.cjp = conf.args.mask_color_jitter_p
        self.mb = conf.args.mask_blur
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.lamda = conf.args.lamda

        self.net_state = copy.deepcopy(self.net.state_dict())
        self.net_anchor = copy.deepcopy(self.net)
        self.net_ema = copy.deepcopy(self.net)
        self.transform = get_tta_transforms()

    def train_online(self, current_num_sample):
        """
        Train the model online
        """

        TRAINED = 0
        SKIPPED = 1
        FINISHED = 2

        if not hasattr(self, 'previous_train_loss'):
            self.previous_train_loss = 0

        if current_num_sample > len(self.target_train_set[0]):
            return FINISHED

        feats, cls, dls = self.target_train_set
        current_sample = feats[current_num_sample - 1], cls[current_num_sample - 1 ], dls[current_num_sample - 1]

        self.mem.add_instance(current_sample)

        if conf.args.use_learned_stats:
            self.evaluation_online(current_num_sample, '', self.mem.get_memory())
        if current_num_sample % conf.args.update_every_x != 0:
            if not (current_num_sample == len(self.target_train_set[0]) and conf.args.update_every_y >= current_num_sample):

                self.log_loss_results('train_online', epoch=current_num_sample, loss_avg = self.previous_train_loss)
                return SKIPPED

        if not conf.args.use_learned_stats:
            self.evaluation_online(current_num_sample, '', self.mem.get_memory())

        masking = Masking(
            block_size=self.mbs,
            ratio=self.mr,
            color_jitter_s=self.cjs,
            color_jitter_p=self.cjp,
            blur=self.mb,
            mean=self.mean,
            std=self.std
        )

        self.net.train()

        if len(feats) == 1:
            self.net.eval()

        feats, cls, dls = self.mem.get_memory()
        feats, cls, dls = torch.stack(feats), cls, torch.stack(dls)


        dataset = torch.utils.data.TensorDataset(feats)
        data_loader = DataLoader(dataset, batch_size=conf.args.opt['batch_size'],
                                 shuffle=True,
                                 drop_last=False,
                                 pin_memory=False)

        for e in range(conf.args.epoch):

            for batch_idx, (x,) in enumerate(data_loader):
                x = x.to(device)

                outputs = self.net(x)

                anchor_prob = torch.nn.functional.softmax(self.net_anchor(x), dim=1).max(1)[0]
                standard_ema = self.net_ema(x)

                N=32
                outputs_emas = []


                if anchor_prob.mean(0) < self.ap:
                    for i in range(N):
                        outputs_ = self.net_ema(self.transform(x)).detach()
                        outputs_emas.append(outputs_)
                    outputs_emas = torch.stack(outputs_emas).mean(0)
                else:
                    outputs_emas = standard_ema

                loss_T = (softmax_entropy(outputs, outputs_emas)).mean(0)


                x_masked = masking(x)
                outputs_masked = self.net(x_masked)
                loss_M = (softmax_entropy(outputs_masked, outputs_emas)).mean(0)



                loss = (1 - self.lamda) * loss_T + self.lamda * loss_M

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()


                self.net_ema = update_ema_variables(ema_model=self.net_ema, model=self.net, alpha_teacher=self.mt)


                if True:
                    for nm,m in self.net.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad:
                                mask = (torch.rand(p.shape) < self.rst).float().cuda()
                                with torch.no_grad():
                                    p.data = self.net_state[f"{nm}.{npp}"] * mask + p * (1. - mask)
        
        self.log_loss_results('train_online', epoch=current_num_sample, loss_avg=0)

        return TRAINED
        

                
