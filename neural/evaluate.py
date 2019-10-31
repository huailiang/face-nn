#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/31

import utils
import torch.nn.functional as F
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

    def discrim_l1(self, y, y_):
        """
        content loss evaluated by lightcnn
        :param y: input photo
        :param y_: reference image
        :return: l1 loss
        """
        loss = utils.discriminative_loss(y, y_, self.lightcnn_inst)
        return loss

    def discrim_l2(self, y, y_):
        """
        facial semantic feature loss
        evaluate loss use l1 at pixel space
        :param y: input photo
        :param y_: reference image
        :return: l2 loss
        """
        img1 = out_evaluate(y, cp=self.parsing, cuda=self.cuda)
        img2 = out_evaluate(y_, cp=self.parsing, cuda=self.cuda)
        img1 = utils.img_edge(img1)
        img2 = utils.img_edge(img2)
        # down sample
        img1 = cv2.resize(img1, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, dsize=(64, 64), interpolation=cv2.INTER_AREA)
        return F.l1_loss(img1, img2)

    def evaluate_ls(self, y, y_, alpha):
        l1 = self.discrim_l1(y, y_)
        l2 = self.discrim_l2(y, y_)
        return alpha * l1 + l2


if __name__ == '__main__':
    pass
