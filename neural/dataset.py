#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

import numpy as np
import torch
import os
import cv2
import random
import struct
import util.logit as log
from util.exception import NeuralException


class FaceDataset:
    """
    由Unity引擎生成的dataset
    """

    def __init__(self, args, mode="train"):
        """
        Dataset construction
        :param args: argparse options
        :param mode: "train": 训练集, "test": 测试集
        """
        self.names = []
        self.params = []
        if mode == "train":
            self.path = args.path_to_dataset
        elif mode == "test":
            self.path = args.path_to_dataset
        else:
            raise NeuralException("not such mode for dataset")
        self.args = args
        if os.path.exists(self.path):
            name = "db_description"
            path = os.path.join(self.path, name)
            log.info(path)
            f = open(path, "rb")
            self.cnt = struct.unpack("i", f.read(4))[0]
            for it in range(self.cnt):
                kk = f.read(10)[1:]  # 第一个是c#字符串的长度
                self.names.append(str(kk, encoding='utf-8'))
                v = []
                for i in range(args.params_cnt):
                    v.append(struct.unpack("f", f.read(4))[0])
                self.params.append(v)
            f.close()
        else:
            log.info("can't be found path %s. Skip it.", self.path)

    def get_batch(self, batch_size):
        """
        以<name, params, image>的形式返回
        formatter: [batch, ?]
        """
        names = []
        cnt = self.cnt
        param_cnt = self.args.params_cnt
        np_params = np.zeros((batch_size, param_cnt), dtype=np.float32)
        np_images = np.zeros((batch_size, 1, 512, 512), dtype=np.float32)
        for i in range(batch_size):
            ind = random.randint(0, cnt - 1)
            name = self.names[ind]
            np_params[i] = self.params[ind]
            name = name + ".jpg"
            names.append(name)
            path = os.path.join(self.path, name)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            np_images[i] = image[np.newaxis, :, :] / 255.0
        params = torch.from_numpy(np_params)
        params.requires_grad = True
        images = torch.from_numpy(np_images)
        log.debug("\nbatch leaf:{0}  grad:{1} type:{2}".format(params.is_leaf, params.requires_grad, params.dtype))
        log.debug("numpy params type:{0}".format(np_params.dtype))
        return names, params, images
