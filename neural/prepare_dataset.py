#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

from __future__ import print_function
import numpy as np
import torch
import os
import scipy.misc
import random
import struct
import util.logit as log


class FaceDataset:
    """
    由Unity引擎生成的dataset
    """

    def __init__(self, args):
        self.names = []
        self.params = []
        self.path_to_dataset = args.path_to_dataset
        cnt = args.db_item_cnt
        self.args = args
        if os.path.exists(self.path_to_dataset):
            name = "db_description"
            path = os.path.join(self.path_to_dataset, name)
            log.info(path)
            f = open(path, "rb")
            for it in range(cnt):
                kk = f.read(10)[1:]  # 第一个是c#字符串的长度
                self.names.append(str(kk, encoding='utf-8'))
                v = []
                for i in range(args.params_cnt):
                    v.append(struct.unpack("f", f.read(4))[0])
                self.params.append(v)
            f.close()
        else:
            print("can't be found path %s. Skip it." % self.path_to_dataset)

    def get_batch(self, batch_size):
        """
        以<name, params, image>的形式返回
        formatter: [batch, ?]
        """
        names = []
        cnt = self.args.db_item_cnt
        params = torch.rand([batch_size, self.args.params_cnt])
        images = torch.rand([batch_size, 3, 512, 512])
        for i in range(batch_size):
            ind = random.randint(0, cnt-1)
            name = self.names[ind]
            val = self.params[ind]
            name = name + ".jpg"
            path = os.path.join(self.path_to_dataset, name)
            image = scipy.misc.imread(name=path, mode='RGB')
            names.append(name)
            params[i] = torch.Tensor(val)
            image = np.swapaxes(image, 1, 0)
            image = np.swapaxes(image, 0, 2)
            image = image/255.0
            images[i] = torch.Tensor(image)
        return names, params, images
