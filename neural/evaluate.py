#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/31

import utils
import util.logit as log
import torch.nn.functional as F
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
        self.max_itr = 1
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
        :return: l2 loss
        """
        img1 = parse_evaluate(y, cp=self.parsing, cuda=self.cuda)
        img2 = y_.cpu().detach().numpy()
        shape = img2.shape
        img2 = img2.reshape(shape[1], shape[2], shape[3])
        img2 = np.swapaxes(img2, 0, 2)
        img2 = parse_evaluate(img2, cp=self.parsing, cuda=self.cuda)
        img1 = utils.img_edge(img1)
        img2 = utils.img_edge(img2)
        # downsample
        img1 = cv2.resize(img1, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        img1_ = torch.from_numpy(img1)
        img2_ = torch.from_numpy(img2)
        return F.l1_loss(img1_, img2_)

    def evaluate_ls(self, y, y_, alpha):
        """
        评估损失Ls
        :param y: input photo, numpy array
        :param y_:  generated image
        :param alpha: 权重
        :return: ls
        """
        l1 = self.discrim_l1(y, y_)
        l2 = self.discrim_l2(y, y_)
        return alpha * l1 + l2

    def itr_train(self, y):
        """
        iterator train
        :param y: numpy array
        :return:
        """
        param_cnt = self.args.params_cnt
        x = utils.random_params(param_cnt)
        np_params = np.zeros((1, param_cnt), dtype=np.float32)
        x_ = torch.from_numpy(np_params)
        alpha = 0.1
        learning_rate = 0.01
        loss_ = 100
        idx_ = 0  # batch index
        for i in range(self.max_itr):
            log.info("step {0}".format(i))
            for j in range(1):
                y_ = self.imitator(x_)
                loss = self.evaluate_ls(y, y_, alpha)
                x_[idx_][j] += learning_rate * loss
                if x_[idx_][j] < 0:
                    x_[idx_][j] = 0
                if x_[idx_][j] > 1:
                    x_[idx_][j] = 1
                log.info("current loss: {0} min loss: {1}".format(loss, loss_))
        return x_


if __name__ == '__main__':
    import logging
    from parse import parser
    import cv2

    log.info("evaluation")
    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/neural_log.txt")
    evl = Evaluate("test", args)
    img = cv2.imread("../export/testset_female2/db_0000_4.jpg").astype(np.float32)
    evl.itr_train(img)
