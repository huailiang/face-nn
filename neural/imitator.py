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
        self.model = nn.Sequential(
            self.layer(95, 64, 4, 1, 3),  # 1. (batch, 64, 4, 4)
            nn.ReplicationPad2d(7),
            self.layer(64, 32, 4, 2),    # 2. (batch, 32, 8, 8)
            nn.ReflectionPad2d(5),
            self.layer(32, 32, 3, 1),    # 3. (batch, 32, 16, 16)
            nn.ReflectionPad2d(9),
            self.layer(32, 16, 3, 1),   # 4. (batch, 16, 32, 32)
            nn.ReplicationPad2d(17),
            self.layer(16, 8, 3, 1),    # 5. (batch, 8, 64, 64)
            nn.ReflectionPad2d(33),
            self.layer(8, 8, 3, 1),     # 6. (batch, 8, 128, 128)
            nn.ReflectionPad2d(65),
            self.layer(8, 4, 3, 1),     # 7. (batch, 4, 256, 256)
            nn.ReflectionPad2d(129),
            self.layer(4, 3, 3, 1),     # 8. (batch, 3, 512, 512)
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
        return self.model(x)

    class ImitatorLoss(nn.Module):
        def __init__(self):
            log.info("imitator loss")

        def forward(self, y_, y, checkpoint):
            return utils.discriminative_loss(y, y_, checkpoint)

    def itr_train(self, x, y, checkpoint):
        loss = F.nll_loss()
        loss.backward()


if __name__ == '__main__':
    log.init("FaceNeural", logging.DEBUG, log_path="output/log.txt")
    imitator = Imitator("neural_imitator")
    y = imitator.forward(torch.randn(2, 95, 1, 1))
    log.info(y.size())
    log.info(type(y))


