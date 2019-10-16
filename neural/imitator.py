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


class ImitatorLoss(nn.Module):
    def __init__(self):
        super(ImitatorLoss, self).__init__()
        log.info("imitator loss __init__")

    def forward(self, y_, y, checkpoint):
        return utils.discriminative_loss(y, y_, checkpoint)


class Imitator(nn.Module):
    def __init__(self, name, lr=0.01, momentum=0.5):
        super(Imitator, self).__init__()
        self.name = name
        self.model = nn.Sequential(
            self.layer(95, 64, 4, 1, 3),  # 1. (batch, 64, 4, 4)
            nn.ReplicationPad2d(7),
            self.layer(64, 32, 4, 2),  # 2. (batch, 32, 8, 8)
            nn.ReflectionPad2d(5),
            self.layer(32, 32, 3, 1),  # 3. (batch, 32, 16, 16)
            nn.ReflectionPad2d(9),
            self.layer(32, 16, 3, 1),  # 4. (batch, 16, 32, 32)
            nn.ReplicationPad2d(17),
            self.layer(16, 8, 3, 1),  # 5. (batch, 8, 64, 64)
            nn.ReflectionPad2d(33),
            self.layer(8, 8, 3, 1),  # 6. (batch, 8, 128, 128)
            nn.ReflectionPad2d(65),
            self.layer(8, 4, 3, 1),  # 7. (batch, 4, 256, 256)
            nn.ReflectionPad2d(129),
            self.layer(4, 3, 3, 1),  # 8. (batch, 3, 512, 512)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def layer(self, in_chanel, out_chanel, kernel_size, stride, pad=0):
        return nn.Sequential(
            nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU()
        )

    def forward(self, x):
        batch = x.size(0)
        length = x.size(1)
        x = x.reshape((batch, length, 1, 1))
        return self.model(x)

    def itr_train(self, x, y, checkpoint):
        imitorloss = ImitatorLoss()
        self.optimizer.zero_grad()
        y_ = self.forward(x)
        print("shape: ", y.shape, y_.shape)
        loss = imitorloss(y_, y, checkpoint)
        self.optimizer.step()


if __name__ == '__main__':
    log.init("FaceNeural", logging.DEBUG, log_path="output/log.txt")
    imitator = Imitator("neural_imitator")
    lightcnn_checkpoint = torch.load("./dat/LightCNN_29Layers_V2_checkpoint.pth.tar", map_location="cpu")
    x = torch.randn(2, 95)
    y_ = torch.randn(2, 3, 512, 512)
    # utils.feature256(y_, lightcnn_checkpoint)
    imitator.itr_train(x, y_, lightcnn_checkpoint)
