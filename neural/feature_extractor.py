#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/16

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import util.logit as log
import utils

"""
imitator
photo生成engine face's params
photo solution: 512x512
"""


class FeatureExtractor(nn.Module):
    def __init__(self, name, lr=0.01, momentum=0.5):
        super(FeatureExtractor, self).__init__()
        log.info("construct feature_extractor %s", name)
        self.name = name
        self.model = nn.Sequential(
            self.layer(3, 3, kernel_size=7, stride=2, pad=3),  # 1. (batch, 3, 256, 256)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 2. (batch, 3, 128, 128)
            self.layer(3, 8, kernel_size=3, stride=2, pad=1),  # 3. (batch, 8, 64, 64)
            self.layer(8, 16, kernel_size=3, stride=2, pad=1),  # 4. (batch, 16, 32, 32)
            self.layer(16, 32, kernel_size=3, stride=2, pad=1),  # 5. (batch, 32, 16, 16)
            self.layer(32, 64, kernel_size=3, stride=2, pad=1),  # 6. (batch, 64, 8, 8)
            self.layer(64, 95, kernel_size=7, stride=2),  # 7. (batch, 95, 1, 1)
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
        log.info("feature_extractor forward with batch: %d", batch)
        return self.model(x)


if __name__ == '__main__':
    log.init("FaceNeural", logging.DEBUG, log_path="output/log.txt")
    extractor = FeatureExtractor("neural_extractor")
    y = extractor.forward(torch.randn(2, 3, 512, 512))
    log.info(y.size())
