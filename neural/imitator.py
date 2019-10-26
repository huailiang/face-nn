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
from module import ResidualBlock

"""
imitator
用来模拟游戏引擎：由params生成图片/灰度图
network: 8 layer
input: params (batch, 95)
output: tensor (batch, 3, 512, 512)
"""


class Imitator(nn.Module):
    def __init__(self, name, args, momentum=0.8, clean=True):
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
        if clean:
            self.clean()
        self.writer = SummaryWriter(comment='imitator', log_dir=args.path_tensor_log)
        self.model = nn.Sequential(
            nn.ConstantPad2d(3, 0.5),
            utils.conv_layer(95, 64, 4, 1),  # 1. (batch, 64, 4, 4)
            ResidualBlock(64, 64),
            nn.ReplicationPad2d(9),
            utils.conv_layer(64, 32, 7, 2),  # 2. (batch, 32, 8, 8)
            nn.ReflectionPad2d(5),
            utils.conv_layer(32, 32, 3, 1),  # 3. (batch, 32, 16, 16)
            ResidualBlock(32, 32),
            nn.ReplicationPad2d(9),
            utils.conv_layer(32, 16, 3, 1),  # 4. (batch, 16, 32, 32)
            nn.ReflectionPad2d(17),
            utils.conv_layer(16, 8, 3, 1),  # 5. (batch, 8, 64, 64)
            ResidualBlock(8, 8),
            nn.ReplicationPad2d(33),
            utils.conv_layer(8, 8, 3, 1),  # 6. (batch, 8, 128, 128)
            nn.ReflectionPad2d(65),
            utils.conv_layer(8, 8, 3, 1),  # 7. (batch, 8, 256, 256)
            nn.ReflectionPad2d(129),
            utils.conv_layer(8, 1, 3, 1),  # 8. (batch, 1, 512, 512) grey
        )
        self.model.apply(utils.init_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def forward(self, params):
        """
        construct network
        :param params: [batch, 95]
        :return: (batch, 1, 512, 512)
        """
        batch = params.size(0)
        length = params.size(1)
        _params = params.reshape((batch, length, 1, 1))
        _params = (_params * 2) - 1
        _params.requires_grad_(True)
        y = self.model(_params)
        return (y + 1) * 0.5

    def itr_train(self, params, reference, lightcnn_inst):
        """
        iterator training
        :param params:  [batch, 95]
        :param reference: reference photo [batch, 1, 512, 512]
        :param lightcnn_inst: light cnn's model
        :return loss: [batch]
        """
        self.optimizer.zero_grad()
        y_ = self.forward(params)
        loss = utils.discriminative_loss(reference, y_, lightcnn_inst)
        # utils.net_parameters(self.model, "imitator")

        loss.backward()  # 求导  loss: [1] scalar
        self.optimizer.step()  # 更新网络参数权重
        return loss, y_

    def batch_train(self, cuda=False):
        """
        batch training
        :param cuda: 是否开启gpu加速运算
        """
        location = self.args.lightcnn
        lightcnn_inst = utils.load_lightcnn(location)
        rnd_input = torch.randn(self.args.batch_size, self.args.params_cnt)
        if cuda:
            rnd_input = rnd_input.cuda()
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

            loss, y_ = self.itr_train(params, images, lightcnn_inst)
            loss_ = loss.detach().numpy()
            progress.set_description("loss:" + "{:.3f}".format(loss_))
            self.writer.add_scalar('imitator/loss', loss_, step)
            self.upload_weights(step)

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

    def upload_weights(self, step):
        """
        把neural net的权重以图片的方式上传到tensorboard
        :param step: train step
        """
        if self.args.open_tensorboard_image:
            for module in self.model._modules.values():
                if isinstance(module, nn.Sequential):
                    for it in module._modules.values():
                        if isinstance(it, nn.Conv2d):
                            name = "weight_{0}_{1}".format(it.in_channels, it.out_channels)
                            if it.in_channels == 32 and it.out_channels == 32:
                                weights = it.weight.reshape(3, 48, -1)
                                self.writer.add_image(name, weights, step)
                            if it.in_channels == 16:
                                weights = it.weight.reshape(3, 24, -1)
                                self.writer.add_image(name, weights, step)
                            break

    def load_checkpoint(self, path, training=False, cuda=False):
        """
        从checkpoint 中恢复net
        :param training: 恢复之后 是否接着train
        :param path: checkpoint's path
        :param cuda: gpu speedup
        """
        checkpoint = torch.load(self.args.path_to_inference + "/" + path)
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.initial_step = checkpoint['epoch']
        log.info("recovery imitator from %s", path)
        if training:
            self.batch_train(cuda)

    def inference(self, path, params, cuda=False):
        """
        imitator生成图片
        :param path: checkpoint's path
        :param params: engine's params
        :param cuda: gpu speedup
        :return: images [batch, 1, 512, 512]
        """
        self.load_checkpoint(path, cuda=cuda)
        _, images = self.forward(params)
        return images

    def evaluate(self):
        """
        评估准确率
        :return: accuracy rate
        """
        self.model.eval()
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
        ops.clear_files(self.args.path_tensor_log)
        ops.clear_files(self.prev_path)
        ops.clear_files(self.model_path)
