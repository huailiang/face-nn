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
from dataset import FaceDataset
from tensorboardX import SummaryWriter

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
        self.clean()
        self.writer = SummaryWriter(comment='imitator', log_dir=args.path_tensor_log)
        self.model = nn.Sequential(
            self.layer(95, 64, 4, 1, 3),  # 1. (batch, 64, 4, 4)
            nn.ReplicationPad2d(7),
            self.layer(64, 32, 4, 2),  # 2. (batch, 32, 8, 8)
            nn.ReplicationPad2d(5),
            self.layer(32, 32, 3, 1),  # 3. (batch, 32, 16, 16)
            nn.ReplicationPad2d(9),
            self.layer(32, 16, 3, 1),  # 4. (batch, 16, 32, 32)
            nn.ReplicationPad2d(17),
            self.layer(16, 8, 3, 1),  # 5. (batch, 8, 64, 64)
            nn.ZeroPad2d(33),
            self.layer(8, 8, 3, 1),  # 6. (batch, 8, 128, 128)
            nn.ZeroPad2d(65),
            self.layer(8, 8, 3, 1),  # 7. (batch, 8, 256, 256)
            nn.ZeroPad2d(129),
            self.layer(8, 3, 3, 1),  # 8. (batch, 3, 512, 512)
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

    def batch_train(self, cuda=False):
        """
        batch training
        :param cuda: 是否开启gpu加速运算， cpu default
        """
        location = self.args.lightcnn
        rnd_input = torch.randn(self.args.batch_size, self.args.params_cnt)
        if cuda:
            checkpoint = torch.load(location)
            rnd_input = rnd_input.cuda()
        else:
            checkpoint = torch.load(location, map_location="cpu")
        self.writer.add_graph(self, input_to_model=rnd_input)
        dataset = FaceDataset(self.args, mode="train")
        initial_step = self.initial_step
        total_steps = self.args.total_steps

        progress = tqdm(range(initial_step, total_steps + 1), initial=initial_step, total=total_steps)
        for step in progress:
            names, params, images = dataset.get_batch(batch_size=self.args.batch_size)
            if cuda:
                params = params.cuda()
                images = images.cuda()
            loss, y_ = self.itr_train(params, images, checkpoint)
            loss_ = loss.detach().numpy()
            progress.set_description("loss:" + "{:.3f}".format(loss_))
            self.writer.add_scalar('imitator/loss', loss_, step)
            if (step + 1) % self.args.prev_freq == 0:
                path = "{1}/imit_{0}.jpg".format(step + 1, self.prev_path)
                ops.save_img(path, images, y_)
                lr = self.args.learning_rate * loss_
                utils.update_optimizer_lr(self.optimizer, lr)
                self.writer.add_scalar('imitator/learning rate', lr, step)
            if (step + 1) % self.args.save_freq == 0:
                state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': step}
                torch.save(state, '{1}/model_imitator_{0}.pth'.format(step + 1, self.model_path))
        self.writer.close()

    def load_checkpoint(self, path, training=False):
        """
        从checkpoint 中恢复net
        :param training: 恢复之后 是否接着train
        :param path: checkpoint's path
        """
        checkpoint = torch.load(self.args.path_to_inference + "/" + path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.initial_step = checkpoint['epoch']
        log.info("recovery imitator from %s", path)
        if training:
            self.batch_train()

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

    def evaluate(self):
        """
        评估准确率
        :return: accuracy rate
        """
        dataset = FaceDataset(self.args, mode="test")
        steps = 100
        accuracy = 0.0
        location = self.args.lightcnn
        checkpoint = torch.load(location, map_location="cpu")
        for step in range(steps):
            log.info("step: %d", step)
            names, params, images = dataset.get_batch(batch_size=self.args.batch_size)
            loss, _ = self.itr_train(params, images, checkpoint)
            accuracy += 1.0 - loss
        accuracy = accuracy / steps
        log.info("accuracy rate is %f", accuracy)
        return accuracy

    def clean(self):
        """
        清空前记得手动备份
        :return:
        """
        ops.clear_files(self.prev_path)
        ops.clear_files(self.model_path)
