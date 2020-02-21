import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import models.module_util as module_util

from args import args as pargs

StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d


class NonAffineBN(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBN, self).__init__(dim, affine=False)


class MaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(module_util.mask_init(self))

        # Turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = module_util.GetSubnet.apply(self.scores.abs(), pargs.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class MultitaskMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mask_init = module_util.mask_init(self)
        # initialize the scores
        d = {set: nn.Parameter(mask_init.clone()) for set in pargs.set}
        d['INIT'] = nn.Parameter(mask_init.clone())
        self.scores = nn.ParameterDict(d)

        # Turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = module_util.GetSubnet.apply(
            self.scores[self.task].abs(), pargs.sparsity
        )
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
