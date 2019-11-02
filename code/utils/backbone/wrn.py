#!/usr/bin/env
# -*- coding: utf-8 -*-
'''
Script Name     : wrn
Author          : Charles Young
Python Version  : Python 3.6.2
Date            : 2018-12-01
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class WideDropoutBlock(nn.Module):
    """
    Class to contruct the wide-dropout residual block used in Wide Residual Network.

    Structure of wide-dropout residual block:
        x_(l)
        |    \
        |  conv3x3
        |     |
        |  dropout
        |     |
        |  conv3x3
        |    /
        x_(l+1)

    Batch normalization and ReLU precede each convolution
    """
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(WideDropoutBlock, self).__init__()

        self.equalInOut = (in_planes == out_planes)

        # First conv3x3 structure
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,  # downsample with first conv
                               padding=1, bias=False)
        # Dropout
        self.droprate = dropRate

        # Second conv3x3 structure
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,  # downsample 
                          padding=0, bias=False)
            )

    def forward(self, x):
        if self.equalInOut:  # shortcut after preactivation
            x = self.relu1(self.bn1(x))
            out = self.conv1(x)
        else:                # preactivation only for residual path
            out = self.relu1(self.bn1(x))
            out = self.conv1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        out += self.shortcut(x)
        return out

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers): # downsample at first conv
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):

    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0 # depth should be 6n+4
        n = (depth - 4) // 6
        block = WideDropoutBlock
        # First conv before any network block [3x3, 16]
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # First block [3x3, 16xk] x N
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # Second block [3x3, 32xk] x N
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # Third block [3x3, 64xk] x N
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Init all parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
