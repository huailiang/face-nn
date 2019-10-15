#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/15


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import util.logit as log
import utils

"""
imitator
用来模拟游戏引擎：由params生成图片(512x512)
"""


class Imitator(nn.Module):
    def __init__(self, name, lr=0.01, momentum=0.5):
        super(Imitator, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(95, 64, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(8, 8, kernel_size=3, stride=1)
        self.conv7 = nn.Conv2d(8, 4, kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(4, 3, kernel_size=3, stride=1)
        self.ref_pad129 = nn.ReflectionPad2d(129)
        self.ref_pad65 = nn.ReflectionPad2d(65)
        self.ref_pad33 = nn.ReflectionPad2d(33)
        self.rep_pad17 = nn.ReplicationPad2d(17)
        self.ref_pad9 = nn.ReflectionPad2d(9)
        self.rep_pad7 = nn.ReplicationPad2d(7)
        self.rep_pad5 = nn.ReflectionPad2d(5)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(3)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, x):
        batch = x.size(0)
        y1 = self.bn64(self.conv1(x))
        y1 = F.relu(y1)  # (batch, 64, 4, 4)
        y2 = self.bn32(self.conv2(self.rep_pad7(y1)))
        y2 = F.relu(y2)  # (batch, 32, 8, 8)
        y3 = self.bn32(self.conv3(self.rep_pad5(y2)))
        y3 = F.relu(y3)  # (batch, 32, 16, 16)
        y4 = self.bn16(self.conv4(self.ref_pad9(y3)))
        y4 = F.relu(y4)  # (batch, 16, 32, 32)
        y5 = self.bn8(self.conv5(self.rep_pad17(y4)))
        y5 = F.relu(y5)  # (batch, 8, 64, 64)
        y6 = self.bn8(self.conv6(self.ref_pad33(y5)))
        y6 = F.relu(y6)  # (batch, 8, 128, 128)
        y7 = self.bn4(self.conv7(self.ref_pad65(y6)))
        y7 = F.relu(y7)  # (batch, 4, 256, 256)
        y8 = self.bn3(self.conv8(self.ref_pad129(y7)))
        y8 = F.relu(y8)  # (batch, 3, 512, 512)
        return y8

    class ImitatorLoss(nn.Module):
        def __init__(self):
            log.info("imitator loss")

        def forward(self, y_, y, checkpoint):
            return utils.discriminative_loss(y, y_, checkpoint)

    def itr_train(self, x, y, checkpoint):
        pass


if __name__ == '__main__':
    log.init("FaceNeural", logging.DEBUG, log_path="output/log.txt")
    imitator = Imitator("neural_imitator")
    bn = nn.BatchNorm2d(95)
    y = imitator.forward(torch.randn(2, 95, 1, 1))
    log.info(y.size())


