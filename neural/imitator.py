#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/15


import utils
import ops
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import util.logit as log
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FaceDataset
from util.exception import NeuralException
from tensorboardX import SummaryWriter

"""
imitator
用来模拟游戏引擎：由params生成图片/灰度图
network: 8 layer
input: params (batch, 95)
output: tensor (batch, 3, 512, 512)
"""


class Imitator(nn.Module):
    def __init__(self, name, args, clean=True):
        """
        imitator
        :param name: imitator name
        :param args: argparse options
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
            utils.deconv_layer(args.params_cnt, 512, kernel_size=4),  # 1. (batch, 512, 4, 4)
            utils.deconv_layer(512, 512, kernel_size=4, stride=2, pad=1),  # 2. (batch, 512, 8, 8)
            utils.deconv_layer(512, 512, kernel_size=4, stride=2, pad=1),  # 3. (batch, 512, 16, 16)
            utils.deconv_layer(512, 256, kernel_size=4, stride=2, pad=1),  # 4. (batch, 256, 32, 32)
            utils.deconv_layer(256, 128, kernel_size=4, stride=2, pad=1),  # 5. (batch, 128, 64, 64)
            utils.deconv_layer(128, 64, kernel_size=4, stride=2, pad=1),  # 6. (batch, 64, 128, 128)
            utils.deconv_layer(64, 64, kernel_size=4, stride=2, pad=1),  # 7. (batch, 64, 256, 256)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 8. (batch, 3, 512, 512)
            nn.Sigmoid(),
        )
        self.model.apply(utils.init_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def forward(self, params):
        """
        forward module
        :param params: [batch, params_cnt]
        :return: (batch, 3, 512, 512)
        """
        batch = params.size(0)
        length = params.size(1)
        _params = params.reshape((batch, length, 1, 1))
        return self.model(_params)

    def itr_train(self, params, reference):
        """
        iterator training
        :param params:  [batch, params_cnt]
        :param reference: reference photo [batch, 3, 512, 512]
        :return loss: [batch], y_: generated picture
        """
        self.optimizer.zero_grad()
        y_ = self.forward(params)
        loss = F.l1_loss(reference, y_)
        loss.backward()  # 求导  loss: [1] scalar
        self.optimizer.step()  # 更新网络参数权重
        return loss, y_

    def batch_train(self, cuda=False):
        """
        batch training
        :param cuda: 是否开启gpu加速运算
        """
        rnd_input = torch.randn(self.args.batch_size, self.args.params_cnt)
        if cuda:
            rnd_input = rnd_input.cuda()
        self.writer.add_graph(self, input_to_model=rnd_input)

        self.model.train()
        dataset = FaceDataset(self.args, mode="train")
        initial_step = self.initial_step
        total_steps = self.args.total_steps
        progress = tqdm(range(initial_step, total_steps + 1), initial=initial_step, total=total_steps)
        for step in progress:
            names, params, images = dataset.get_batch(batch_size=self.args.batch_size, edge=False)
            if cuda:
                params = params.cuda()
                images = images.cuda()

            loss, y_ = self.itr_train(params, images)
            loss_ = loss.cpu().detach().numpy()
            progress.set_description("loss: {:.3f}".format(loss_))
            self.writer.add_scalar('imitator/loss', loss_, step)

            if (step + 1) % self.args.prev_freq == 0:
                path = "{1}/imit_{0}.jpg".format(step + 1, self.prev_path)
                self.capture(path, images, y_, self.args.parsing_checkpoint, cuda)
                x = step / float(total_steps)
                lr = self.args.learning_rate * (x ** 2 - 2 * x + 1) + 2e-3
                utils.update_optimizer_lr(self.optimizer, lr)
                self.writer.add_scalar('imitator/learning rate', lr, step)
                self.upload_weights(step)
            if (step + 1) % self.args.save_freq == 0:
                self.save(step)
        self.writer.close()

    def upload_weights(self, step):
        """
        把neural net的权重以图片的方式上传到tensorboard
        :param step: train step
        :return weights picture
        """
        for module in self.model._modules.values():
            if isinstance(module, nn.Sequential):
                for it in module._modules.values():
                    if isinstance(it, nn.ConvTranspose2d):
                        if it.in_channels == 64 and it.out_channels == 64:
                            name = "weight_{0}_{1}".format(it.in_channels, it.out_channels)
                            weights = it.weight.reshape(4, 64, -1)
                            self.writer.add_image(name, weights, step)
                            return weights

    def load_checkpoint(self, path, training=False, cuda=False):
        """
        从checkpoint 中恢复net
        :param training: 恢复之后 是否接着train
        :param path: checkpoint's path
        :param cuda: gpu speedup
        """
        path_ = self.args.path_to_inference + "/" + path
        if not os.path.exists(path_):
            raise NeuralException("not exist checkpoint of imitator with path " + path_)
        if cuda:
            checkpoint = torch.load(path_)
        else:
            checkpoint = torch.load(path_, map_location='cpu')
        self.model.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.initial_step = checkpoint['epoch']
        log.info("recovery imitator from %s", path)
        if training:
            self.batch_train(cuda)

    def evaluate(self):
        """
        评估准确率
        方法是loss取反，只是一个相对值
        :return: accuracy rate
        """
        self.model.eval()
        dataset = FaceDataset(self.args, mode="test")
        steps = 100
        accuracy = 0.0
        losses = []
        for step in range(steps):
            log.info("step: %d", step)
            names, params, images = dataset.get_batch(batch_size=self.args.batch_size, edge=False)
            loss, _ = self.itr_train(params, images)
            accuracy += 1.0 - loss
            losses.append(loss.item())
        self.plot(losses)
        accuracy = accuracy / steps
        log.info("accuracy rate is %f", accuracy)
        return accuracy

    def plot(self, losses):
        plt.style.use('seaborn-whitegrid')
        steps = len(losses)
        x = range(steps)
        plt.plot(x, losses)
        plt.xlabel('step')
        plt.ylabel('loss')
        path = os.path.join(self.prev_path, "imitator.png")
        plt.savefig(path)

    def clean(self):
        """
        清空前记得手动备份
        """
        ops.clear_files(self.args.path_tensor_log)
        ops.clear_files(self.prev_path)
        ops.clear_files(self.model_path)

    def save(self, step):
        """
       save checkpoint
       :param step: train step
       """
        state = {'net': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': step}
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        ext = "cuda" if self.cuda() else "cpu"
        torch.save(state, '{1}/imitator_{0}_{2}.pth'.format(step + 1, self.model_path, ext))

    @staticmethod
    def capture(path, tensor1, tensor2, parse, cuda):
        """
        imitator 快照
        :param cuda: use gpu
        :param path: save path
        :param tensor1: input photo
        :param tensor2: generated image
        :param parse: parse checkpoint's path
        """
        img1 = ops.tensor_2_image(tensor1)[0].swapaxes(0, 1).astype(np.uint8)
        img2 = ops.tensor_2_image(tensor2)[0].swapaxes(0, 1).astype(np.uint8)
        img1 = cv2.resize(img1, (512, 512), interpolation=cv2.INTER_LINEAR)
        img3 = utils.faceparsing_ndarray(img1, parse, cuda)
        img4 = utils.img_edge(img3)
        img4 = 255 - ops.fill_gray(img4)
        image = ops.merge_4image(img1, img2, img3, img4, transpose=False)
        cv2.imwrite(path, image)
