#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/15


import torch
import torch.nn as nn
import torch.optim as optim
import util.logit as log
import utils
import numpy as np
import ops
from tqdm import tqdm
from prepare_dataset import FaceDataset

"""
imitator
用来模拟游戏引擎：由params生成图片
network: 8 layer
input: params (batch,95)
output: tensor (batch, 3, 512, 512)
"""


class Imitator(nn.Module):
    def __init__(self, name, args, momentum=0.5):
        """
        imitator
        :param name: imitator name
        :param args: argparse options
        :param momentum: momentum for optimizer
        """
        super(Imitator, self).__init__()
        self.name = name
        self.args = args
        self.initial_step = 0
        self.prev_path = "./output/preview"
        self.model_path = "./output/imitator"
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
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=momentum)

    @staticmethod
    def layer(in_chanel, out_chanel, kernel_size, stride, pad=0):
        """
        imitator convolution layer
        """
        return nn.Sequential(
            nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU()
        )

    def forward(self, params):
        """
        construct network
        :param params: [batch, 95]
        :return: (batch, 3, 512, 512)
        """
        batch = params.size(0)
        length = params.size(1)
        _params = params.reshape((batch, length, 1, 1))
        _params.requires_grad_(True)
        return self.model(_params)

    def itr_train(self, params, referimage, checkpoint):
        """
        iterator training
        :param params:  [batch, 95]
        :param referimage: reference photo [batch, 3, 512, 512]
        :param checkpoint: light cnn's model
        :return loss: [batch]
        """
        self.optimizer.zero_grad()
        y_ = self.forward(params)
        loss = utils.discriminative_loss(referimage, y_, checkpoint)
        loss.backward()  # 求导  loss: [batch]
        self.optimizer.step()  # 更新网络参数权重
        return loss, y_

    def batch_train(self):
        """
        step training, cpu default
        """
        location = self.args.lightcnn
        checkpoint = torch.load(location, map_location="cpu")
        dataset = FaceDataset(self.args)
        initial_step = self.initial_step
        total_steps = self.args.total_steps
        self.clean()
        progress = tqdm(range(initial_step, total_steps + 1), initial=initial_step, total=total_steps)
        for step in progress:
            names, params, images = dataset.get_batch(batch_size=1)
            loss, y_ = self.itr_train(params, images, checkpoint)
            loss_ = loss.detach().numpy()
            progress.set_description("loss:"+"{:.3f}".format(loss_))
            if (step + 1) % 20 == 0:
                path = "{2}/imitator_{0}_step{1}.jpg".format(names[0][2:-4], step, self.prev_path)
                utils.save_img(path, y_)
            if (step + 1) % self.args.save_freq == 0:
                state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': step}
                torch.save(state, '{1}/model_imitator_{0}.pth'.format(step + 1, self.model_path))

    def load_checkpoint(self, path):
        """
        从checkpoint 中恢复net
        :param path: checkpoint's path
        """
        self.clean()
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.initial_step = checkpoint['epoch']
        log.info("recovery imitator from %s", path)

    def inference(self, path, params):
        """
        imitator生成图片
        :param path: checkpoint's path
        :param params: engine's params
        :return: images [batch, 3, 512, 512]
        """
        self.load_checkpoint(path)
        _, images = self.forward(params)
        return images

    def clean(self):
        """
        清空前记得备份
        :return:
        """
        ops.clear_files(self.prev_path)
        ops.clear_files(self.model_path)
