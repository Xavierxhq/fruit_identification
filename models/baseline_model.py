# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import torch.nn.functional as F
from torch import nn

from .resnet import ResNet, resnet18, resnet34, resnet101, resnet152



class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, last_stride=1, model_path=None, layers=50):
        super().__init__()
        if layers == 50:
            self.base = ResNet(last_stride)
        if layers == 18:
            self.base = resnet18(last_stride=last_stride)
        if layers == 34:
            self.base = resnet34(last_stride=last_stride)
        if layers == 101:
            self.base = resnet101(last_stride=last_stride)
        if layers == 152:
            self.base = resnet152(last_stride=last_stride)
        if model_path:
            self.base.load_param(model_path)


    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()

        return [
            {'params': base_param_group}
        ]
