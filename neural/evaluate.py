#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/31

import utils
import util.logit as log
from tqdm import tqdm
from imitator import Imitator
from faceparsing.evaluate import *

"""
面部相似性度量
    Ls = alpha * L1 + L2
"""


class Evaluate:
    def __init__(self, name, args, cuda=False):
        """
        Evaluate
        :param name: evaluate name
        :param args: argparse options
        :param cuda: gpu speed up
        """
        self.name = name
        self.args = args
        location = self.args.lightcnn
        self.lightcnn_inst = utils.load_lightcnn(location)
        self.cuda = cuda
        self.parsing = self.args.parsing_checkpoint
        self.max_itr = 4
        self.imitator = Imitator("neural imitator", args, clean=False)
        if cuda:
            self.imitator.cuda()
        self.imitator.load_checkpoint("model_imitator_15000.pth", False, cuda=cuda)

    def discrim_l1(self, y, y_):
        """
        content loss evaluated by lightcnn
        :param y: input photo, numpy array [H, W, C]
        :param y_: generated image, torch tensor [B, C, W, H]
        :return: l1 loss
        """
        y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        y = np.swapaxes(y, 0, 2)
        y = np.mean(y, axis=0)[np.newaxis, np.newaxis, :, :]
        y = torch.from_numpy(y)
        y_ = y_.cpu().detach().numpy()
        y_ = y_.reshape(512, 512, 3)
        y_ = cv2.resize(y_, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        y_ = np.mean(y_, axis=2)[np.newaxis, np.newaxis, :, :]
        y_ = torch.from_numpy(y_)
        return utils.discriminative_loss(y, y_, self.lightcnn_inst)

    def discrim_l2(self, y, y_):
        """
        facial semantic feature loss
        evaluate loss use l1 at pixel space
        :param y: input photo, numpy array  [H, W, C]
        :param y_: generated image, tensor  [B, C, W, H]
        :return: l1 loss in pixel space
        """
        img1 = parse_evaluate(y, cp=self.parsing, cuda=self.cuda)
        img2 = y_.cpu().detach().numpy()
        shape = img2.shape
        img2 = img2.reshape(shape[1], shape[2], shape[3])
        img2 = np.swapaxes(img2, 0, 2)
        img2 = parse_evaluate(img2, cp=self.parsing, cuda=self.cuda)
        img1 = utils.img_edge(img1).astype(np.float32)
        img2 = utils.img_edge(img2).astype(np.float32)
        return np.mean(img1 - img2)

    def evaluate_ls(self, y, y_, alpha):
        """
        评估损失Ls
        :param y: input photo, numpy array
        :param y_:  generated image, tensor [b,c,w,h]
        :param alpha: 权重
        :return: ls
        """
        l1 = self.discrim_l1(y, y_)
        l2 = self.discrim_l2(y, y_)
        log.debug("l1:{0:.3f} l2:{1:.3f}".format(l1, l2))
        return alpha * l1 + l2

    def itr_train(self, y):
        """
        iterator train
        :param y: numpy array
        :return:
        """
        param_cnt = self.args.params_cnt
        np_params = 0.5 * np.ones((1, param_cnt), dtype=np.float32)
        x_ = torch.from_numpy(np_params)
        alpha = 0.1
        learning_rate = 0.01
        idx = 0  # batch index
        steps = param_cnt
        progress = tqdm(range(0, steps), initial=0, total=steps)  # type: tqdm
        for j in progress:
            loss_ = 0
            for i in range(self.max_itr):
                y_ = self.imitator(x_)
                loss = self.evaluate_ls(y, y_, alpha)
                delta = loss - loss_
                x_[idx][j] = self.update_x(x_[idx][j], learning_rate * delta)
                loss_ = loss
                progress.set_description("loss: {0:.3f} loss_: {1:.3f} delta: {2:.3f}".format(loss, loss_, delta))
        return x_

    @staticmethod
    def update_x(x, loss):
        """
        更新梯度
        :param x: input scalar
        :param loss: gradient loss
        :return: updated value, scalar
        """
        x -= loss
        if x < 0:
            x = 0
        elif x > 1:
            x = 1
        return x


if __name__ == '__main__':
    import logging
    from parse import parser

    log.info("evaluation mode start")
    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/neural_log.txt")
    evl = Evaluate("test", args)
    img = cv2.imread("../export/testset_female2/db_0000_4.jpg").astype(np.float32)
    evl.itr_train(img)
