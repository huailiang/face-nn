#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/16

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util.logit as log
import utils
import ops
import os
from tqdm import tqdm
import align
from tensorboardX import SummaryWriter

"""
feature extractor
photo生成engine face's params
input: photo solution: 512x512
output: engine params [95]
"""


class FeatureExtractor(nn.Module):
    def __init__(self, name, args, imitator, momentum=0.5):
        """
        feature extractor
        :param name: model name
        :param args: argparse options
        :param imitator: imitate engine's behaviour
        :param momentum:  momentum for optimizer
        """
        super(FeatureExtractor, self).__init__()
        log.info("construct feature_extractor %s", name)
        self.name = name
        self.imitator = imitator
        self.initial_step = 0
        self.args = args
        self.model_path = "./output/imitator"
        self.params_path = "./output/params"
        self.writer = SummaryWriter(comment="feature extractor", log_dir=args.path_tensor_log)
        self.model = nn.Sequential(
            self.layer(3, 3, kernel_size=7, stride=2, pad=3),  # 1. (batch, 3, 256, 256)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 2. (batch, 3, 128, 128)
            self.layer(3, 8, kernel_size=3, stride=2, pad=1),  # 3. (batch, 8, 64, 64)
            self.layer(8, 16, kernel_size=3, stride=2, pad=1),  # 4. (batch, 16, 32, 32)
            self.layer(16, 32, kernel_size=3, stride=2, pad=1),  # 5. (batch, 32, 16, 16)
            self.layer(32, 64, kernel_size=3, stride=2, pad=1),  # 6. (batch, 64, 8, 8)
            self.layer(64, 95, kernel_size=7, stride=2),  # 7. (batch, 95, 1, 1)
        )
        self.optimizer = optim.SGD(self.parameters(),
                                   lr=args.extractor_learning_rate,
                                   momentum=momentum)

    @staticmethod
    def layer(in_chanel, out_chanel, kernel_size, stride, pad=0):
        return nn.Sequential(
            nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.BatchNorm2d(out_chanel),
            nn.ReLU()
        )

    def forward(self, x):
        batch = x.size(0)
        log.info("feature_extractor forward with batch: %d", batch)
        return self.model(x)

    def itr_train(self, image):
        """
        这里train的方式使用的是imitator
        第二种方法是 通过net把params发生引擎生成image
        (这种方法需要保证同步，但效果肯定比imitator效果好)
        :param image: [batch, 3, 512, 512]
        :return: loss scalar
        """
        self.optimizer.zero_grad()
        param_ = self.forward(image)
        img_ = self.imitator.forward(param_)
        loss = utils.content_loss(image, img_)
        loss.backward()
        self.optimizer.step()
        return loss, param_

    def batch_train(self, cuda):
        log.info("feature extractor train")
        initial_step = self.initial_step
        total_steps = self.args.total_extractor_steps
        progress = tqdm(range(initial_step, total_steps + 1), initial=initial_step, total=total_steps)
        for step in progress:
            log.info("current step: %d", step)
            if (step + 1) % 20 == 0:
                log.info("step {0}", step)
                loss, params = self.itr_train(None)
                loss_ = loss.detach().numpy()
                progress.set_description("loss:" + "{:.3f}".format(loss_))
                self.writer.add_scalar('feature extractor/loss', loss_, step)
                path = os.path.join(self.params_path, "step" + str(step))
                ops.generate_file(path, params)
                utils.update_optimizer_lr(self.optimizer, loss_)
            if (step + 1) % self.args.extractor_save_freq == 0:
                state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': step}
                torch.save(state, '{1}/model_imitator_{0}.pth'.format(step + 1, self.model_path))
        self.writer.close()

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

    def clean(self):
        """
        清空前记得备份
        :return:
        """
        ops.clear_folder(self.model_path)

    def inference(self, path, photo):
        """
        feature extractor: 由图片生成捏脸参数
        :param path: checkpoint's path
        :param photo: input photo
        :return: params [batch, 95]
        """
        align.align_face()
        self.load_checkpoint(path)
        _, params_ = self.forward(path)
        return params_
