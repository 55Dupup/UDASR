import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import numpy as np
import uuid
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from archs import common
from utils.registry import ARCH_REGISTRY


## Channel Attention (CA) Layer
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

@ARCH_REGISTRY.register()
class DegradationInfModule(nn.Module):
    def __init__(self, in_channels=64, num_residual_blocks=16, num_conv_layers=4):
        super(DegradationInfModule, self).__init__()
        self.in_channels = in_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_conv_layers = num_conv_layers

        layers = []
        layers += [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_residual_blocks):
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]

        for _ in range(num_conv_layers):
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]

        layers += [
            ChannelAttention(in_channels),
            nn.Conv2d(in_channels, int(in_channels/2), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/2), int(in_channels/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/4), int(in_channels/8), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/8), int(in_channels/16), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/16),1, kernel_size=3, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

def make_model(args, parent=False):
    return EDSR(args)#RCAN()


## Channel Attention (CA) Layer


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
 
    def __init__(self, nb, nf, res_scale=1.0, upscale=4, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = nb
        n_feats = nf
        kernel_size = 3
        scale = upscale
        act = nn.ReLU(True)

        self.sub_mean = MeanShift(255.0, sign=-1)
        self.add_mean = MeanShift(255.0, sign=1)
        self.degInf= DegradationInfModule(64,16,4)
        # define head module  
        m_head = [conv(4, n_feats, kernel_size)]
        # m_head = [conv(4, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
    def forward(self, x,deg_strength=None):
#----------update----------#
        x = self.sub_mean(x * 255.0)
        lr_x=x
        b, c, h, w = x.shape
        deg_strength_map=self.degInf(deg_strength)
        x = torch.cat((x, deg_strength_map), dim=1)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x) / 255.0
         
        return x
