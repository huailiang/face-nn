#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/16

import align
import utils
import ops
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util.logit as log
import numpy as np
from tqdm import tqdm
from module import ResidualBlock, group
from tensorboardX import SummaryWriter

"""
feature extractor
photo生成engine face's params
input: photo solution: 64x64
output: engine params [95]
"""


class Extractor(nn.Module):
    def __init__(self, name, args, imitator=None, momentum=0.5):
        """
        feature extractor
        :param name: model name
        :param args: argparse options
        :param imitator: imitate engine's behaviour
        :param momentum:  momentum for optimizer
        """
        super(Extractor, self).__init__()
        log.info("construct feature_extractor %s", name)
        self.name = name
        self.imitator = imitator
        self.initial_step = 0
        self.args = args
        self.model_path = "./output/extractor"
        self.params_path = "./output/params"
        self.training = False
        self.params_cnt = self.args.params_cnt
        self.writer = SummaryWriter(comment="feature extractor", log_dir=args.path_tensor_log)
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3),  # 1. (batch, 4, 32, 32)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 2. (batch, 4, 16, 16)
            group(4, 8, kernel_size=3, stride=1, padding=1),  # 3. (batch, 8, 16, 16)
            ResidualBlock.make_layer(8, channels=8),  # 4. (batch, 8, 16, 16)
            group(8, 16, kernel_size=3, stride=1, padding=1),  # 5. (batch, 16, 16, 16)
            ResidualBlock.make_layer(8, channels=16),  # 6. (batch, 16, 16, 16)
            group(16, 64, kernel_size=3, stride=1, padding=1),  # 7. (batch, 64, 16, 16)
            ResidualBlock.make_layer(8, channels=64),  # 8. (batch, 64, 16, 16)
            group(64, self.params_cnt, kernel_size=3, stride=1, padding=1),  # 9. (batch, params_cnt, 16, 16)
            ResidualBlock.make_layer(4, channels=self.params_cnt),  # 10. (batch, params_cnt, 16, 16)
        )
        self.fc = nn.Linear(95 * 16 * 16, 95)
        self.optimizer = optim.SGD(self.parameters(),
                                   lr=args.extractor_learning_rate,
                                   momentum=momentum)

    def forward(self, input):
        batch = input.size(0)
        log.info("feature extractor forward with batch: %d", batch)
        output = self.model(input)
        output = output.view(x.size(0), -1)
        output = self.fc(output)
        output = F.dropout(output, training=self.training)
        output = torch.sigmoid(output)
        return output

    def itr_train(self, image):
        """
        这里train的方式使用的是imitator （同步）
        第二种方法是 通过net把params发生引擎生成image (异步)
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
        self.training = True
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
                self.save(step)
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

    def save(self, step):
        state = {'net': self.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': step}
        torch.save(state, '{1}/model_extractor_{0}.pth'.format(step + 1, self.model_path))

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


if __name__ == '__main__':
    from parse import parser
    import logging

    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/ex_log.txt")
    extractor = Extractor("neural extractor", args)
    x = np.random.rand(512, 512, 1).astype(np.float32)
    log.info(x.shape)
    x = cv2.resize(x, (64, 64), interpolation=cv2.INTER_LINEAR)
    log.info(x.shape)
    x = torch.from_numpy(x)
    log.info(x.size())
    x = x.view(-1, 1, 64, 64)
    log.info(x.size())
    y = extractor.forward(x)
    log.info(y.size())
    log.info(y)
    extractor.save(99)
