# encoding: utf-8
"""
@author: liaoxingyu
@contact: liaoxingyu@megvii.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import torch as th
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def load_param(self, model_path):
        param_dict = th.load(model_path)
        for i in param_dict:
            if 'fc' in i or 'epoch' in i or 'state_dict' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def resnet18(last_stride=2):
    model = ResNet(last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2])
    return model


def resnet34(last_stride=2):
    model = ResNet(last_stride=last_stride, block=BasicBlock, layers=[3, 4, 6, 3])
    return model


def resnet101(last_stride=2):
    model = ResNet(last_stride=last_stride, layers=[3, 4, 23, 3])
    return model


def resnet152(last_stride=2):
    model = ResNet(last_stride=last_stride, layers=[3, 8, 36, 3])
    return model


if __name__ == "__main__":
    net = ResNet(last_stride=2)
    net.load_param('../model/resnet50-19c8e357.pth')
    # param_dict = torch.load(model_path)
    # print (param_dict.keys())
    # odict_keys(['conv1.weight', 'bn1.running_mean', 'bn1.running_var', 'bn1.weight', 'bn1.bias',
    # 'layer1.0.conv1.weight', 'layer1.0.bn1.running_mean', 'layer1.0.bn1.running_var', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias',
    # 'layer1.0.conv2.weight', 'layer1.0.bn2.running_mean', 'layer1.0.bn2.running_var', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias',
    # 'layer1.0.conv3.weight', 'layer1.0.bn3.running_mean', 'layer1.0.bn3.running_var', 'layer1.0.bn3.weight', 'layer1.0.bn3.bias',
    # 'layer1.0.downsample.0.weight', 'layer1.0.downsample.1.running_mean', 'layer1.0.downsample.1.running_var', 'layer1.0.downsample.1.weight', 'layer1.0.downsample.1.bias',
    # 'layer1.1.conv1.weight', 'layer1.1.bn1.running_mean', 'layer1.1.bn1.running_var', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
    # 'layer1.1.conv2.weight', 'layer1.1.bn2.running_mean', 'layer1.1.bn2.running_var', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias',
    # 'layer1.1.conv3.weight', 'layer1.1.bn3.running_mean', 'layer1.1.bn3.running_var', 'layer1.1.bn3.weight', 'layer1.1.bn3.bias',
    # 'layer1.2.conv1.weight', 'layer1.2.bn1.running_mean', 'layer1.2.bn1.running_var', 'layer1.2.bn1.weight', 'layer1.2.bn1.bias',
    # 'layer1.2.conv2.weight', 'layer1.2.bn2.running_mean', 'layer1.2.bn2.running_var', 'layer1.2.bn2.weight', 'layer1.2.bn2.bias',
    # 'layer1.2.conv3.weight', 'layer1.2.bn3.running_mean', 'layer1.2.bn3.running_var', 'layer1.2.bn3.weight', 'layer1.2.bn3.bias',
    # 'layer2.0.conv1.weight', 'layer2.0.bn1.running_mean', 'layer2.0.bn1.running_var', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias',
    # 'layer2.0.conv2.weight', 'layer2.0.bn2.running_mean', 'layer2.0.bn2.running_var', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias',
    # 'layer2.0.conv3.weight', 'layer2.0.bn3.running_mean', 'layer2.0.bn3.running_var', 'layer2.0.bn3.weight', 'layer2.0.bn3.bias',
    # 'layer2.0.downsample.0.weight', 'layer2.0.downsample.1.running_mean', 'layer2.0.downsample.1.running_var', 'layer2.0.downsample.1.weight', 'layer2.0.downsample.1.bias',
    # 'layer2.1.conv1.weight', 'layer2.1.bn1.running_mean', 'layer2.1.bn1.running_var', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias',
    # 'layer2.1.conv2.weight', 'layer2.1.bn2.running_mean', 'layer2.1.bn2.running_var', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias',
    # 'layer2.1.conv3.weight', 'layer2.1.bn3.running_mean', 'layer2.1.bn3.running_var', 'layer2.1.bn3.weight', 'layer2.1.bn3.bias',
    # 'layer2.2.conv1.weight', 'layer2.2.bn1.running_mean', 'layer2.2.bn1.running_var', 'layer2.2.bn1.weight', 'layer2.2.bn1.bias',
    # 'layer2.2.conv2.weight', 'layer2.2.bn2.running_mean', 'layer2.2.bn2.running_var', 'layer2.2.bn2.weight', 'layer2.2.bn2.bias',
    # 'layer2.2.conv3.weight', 'layer2.2.bn3.running_mean', 'layer2.2.bn3.running_var', 'layer2.2.bn3.weight', 'layer2.2.bn3.bias',
    # 'layer2.3.conv1.weight', 'layer2.3.bn1.running_mean', 'layer2.3.bn1.running_var', 'layer2.3.bn1.weight', 'layer2.3.bn1.bias',
    # 'layer2.3.conv2.weight', 'layer2.3.bn2.running_mean', 'layer2.3.bn2.running_var', 'layer2.3.bn2.weight', 'layer2.3.bn2.bias',
    # 'layer2.3.conv3.weight', 'layer2.3.bn3.running_mean', 'layer2.3.bn3.running_var', 'layer2.3.bn3.weight', 'layer2.3.bn3.bias',
    # 'layer3.0.conv1.weight', 'layer3.0.bn1.running_mean', 'layer3.0.bn1.running_var', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias',
    # 'layer3.0.conv2.weight', 'layer3.0.bn2.running_mean', 'layer3.0.bn2.running_var', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias',
    # 'layer3.0.conv3.weight', 'layer3.0.bn3.running_mean', 'layer3.0.bn3.running_var', 'layer3.0.bn3.weight', 'layer3.0.bn3.bias',
    # 'layer3.0.downsample.0.weight', 'layer3.0.downsample.1.running_mean', 'layer3.0.downsample.1.running_var', 'layer3.0.downsample.1.weight', 'layer3.0.downsample.1.bias',
    # 'layer3.1.conv1.weight', 'layer3.1.bn1.running_mean', 'layer3.1.bn1.running_var', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias',
    # 'layer3.1.conv2.weight', 'layer3.1.bn2.running_mean', 'layer3.1.bn2.running_var', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias',
    # 'layer3.1.conv3.weight', 'layer3.1.bn3.running_mean', 'layer3.1.bn3.running_var', 'layer3.1.bn3.weight', 'layer3.1.bn3.bias',
    # 'layer3.2.conv1.weight', 'layer3.2.bn1.running_mean', 'layer3.2.bn1.running_var', 'layer3.2.bn1.weight', 'layer3.2.bn1.bias',
    # 'layer3.2.conv2.weight', 'layer3.2.bn2.running_mean', 'layer3.2.bn2.running_var', 'layer3.2.bn2.weight', 'layer3.2.bn2.bias',
    # 'layer3.2.conv3.weight', 'layer3.2.bn3.running_mean', 'layer3.2.bn3.running_var', 'layer3.2.bn3.weight', 'layer3.2.bn3.bias',
    # 'layer3.3.conv1.weight', 'layer3.3.bn1.running_mean', 'layer3.3.bn1.running_var', 'layer3.3.bn1.weight', 'layer3.3.bn1.bias',
    # 'layer3.3.conv2.weight', 'layer3.3.bn2.running_mean', 'layer3.3.bn2.running_var', 'layer3.3.bn2.weight', 'layer3.3.bn2.bias',
    # 'layer3.3.conv3.weight', 'layer3.3.bn3.running_mean', 'layer3.3.bn3.running_var', 'layer3.3.bn3.weight', 'layer3.3.bn3.bias',
    # 'layer3.4.conv1.weight', 'layer3.4.bn1.running_mean', 'layer3.4.bn1.running_var', 'layer3.4.bn1.weight', 'layer3.4.bn1.bias',
    # 'layer3.4.conv2.weight', 'layer3.4.bn2.running_mean', 'layer3.4.bn2.running_var', 'layer3.4.bn2.weight', 'layer3.4.bn2.bias',
    # 'layer3.4.conv3.weight', 'layer3.4.bn3.running_mean', 'layer3.4.bn3.running_var', 'layer3.4.bn3.weight', 'layer3.4.bn3.bias',
    # 'layer3.5.conv1.weight', 'layer3.5.bn1.running_mean', 'layer3.5.bn1.running_var', 'layer3.5.bn1.weight', 'layer3.5.bn1.bias',
    # 'layer3.5.conv2.weight', 'layer3.5.bn2.running_mean', 'layer3.5.bn2.running_var', 'layer3.5.bn2.weight', 'layer3.5.bn2.bias',
    # 'layer3.5.conv3.weight', 'layer3.5.bn3.running_mean', 'layer3.5.bn3.running_var', 'layer3.5.bn3.weight', 'layer3.5.bn3.bias',
    # 'layer4.0.conv1.weight', 'layer4.0.bn1.running_mean', 'layer4.0.bn1.running_var', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias',
    # 'layer4.0.conv2.weight', 'layer4.0.bn2.running_mean', 'layer4.0.bn2.running_var', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias',
    # 'layer4.0.conv3.weight', 'layer4.0.bn3.running_mean', 'layer4.0.bn3.running_var', 'layer4.0.bn3.weight', 'layer4.0.bn3.bias',
    # 'layer4.0.downsample.0.weight', 'layer4.0.downsample.1.running_mean', 'layer4.0.downsample.1.running_var', 'layer4.0.downsample.1.weight', 'layer4.0.downsample.1.bias',
    # 'layer4.1.conv1.weight', 'layer4.1.bn1.running_mean', 'layer4.1.bn1.running_var', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias',
    # 'layer4.1.conv2.weight', 'layer4.1.bn2.running_mean', 'layer4.1.bn2.running_var', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias',
    # 'layer4.1.conv3.weight', 'layer4.1.bn3.running_mean', 'layer4.1.bn3.running_var', 'layer4.1.bn3.weight', 'layer4.1.bn3.bias',
    # 'layer4.2.conv1.weight', 'layer4.2.bn1.running_mean', 'layer4.2.bn1.running_var', 'layer4.2.bn1.weight', 'layer4.2.bn1.bias',
    # 'layer4.2.conv2.weight', 'layer4.2.bn2.running_mean', 'layer4.2.bn2.running_var', 'layer4.2.bn2.weight', 'layer4.2.bn2.bias',
    # 'layer4.2.conv3.weight', 'layer4.2.bn3.running_mean', 'layer4.2.bn3.running_var', 'layer4.2.bn3.weight', 'layer4.2.bn3.bias', 'fc.weight', 'fc.bias'])

    import torch

    x = net(torch.zeros(1, 3, 256, 128))
    print(x.shape)
