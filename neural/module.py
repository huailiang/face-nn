#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 10/22/19


import torch
import torch.nn as nn


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, mode=1):
        """
        mfm
        :param in_channels: in channel
        :param out_channels: out channel
        :param kernel_size: conv kernel size
        :param stride: conv stride
        :param padding: conv padding
        :param mode: 1: Conv2d  2: Linear
        """
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if mode == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2 * out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    """
    残差网络
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels)
        self.conv2 = mfm(in_channels, out_channels)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

    @staticmethod
    def make_layer(num_blocks, channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(ResidualBlock(channels, channels))
        return nn.Sequential(*layers)
