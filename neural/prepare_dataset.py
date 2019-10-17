#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

from __future__ import print_function
import numpy as np
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
        self.dataset = {}
        self.path_to_dataset = args.path_to_dataset
        cnt = args.db_item_cnt
        self.args = args
        if os.path.exists(self.path_to_dataset):
            name = "db_description"
            path = os.path.join(self.path_to_dataset, name)
            log.info(path)
            f = open(path, "rb")
            for it in range(cnt):
                kk = f.read(9)[1:]  # 第一个是c#字符串的长度
                k = struct.unpack("8s", kk)[0]
                v = []
                for i in range(args.params_cnt):
                    v.append(struct.unpack("f", f.read(4))[0])
                    self.dataset[k] = v
            f.close()
        else:
            print("can't be found path %s. Skip it." % self.path_to_dataset)

    def get_batch(self, batch_size):
        """
        以<name, (params, image)>的形式返回
        """
        batch_rst = {}
        for _ in range(batch_size):
            cnt = self.args.db_item_cnt
            ind = random.randint(0, cnt)
            keys = self.dataset.keys()
            key = keys[ind]
            val = self.dataset[key]
            name = key + ".jpg"
            path = os.path.join(self.path_to_dataset, name)
            image = scipy.misc.imread(name=path, mode='RGB')
            # log.info("info: ", key, len(val), path, image.shape)
            batch_rst[key] = (val, image)
            return batch_rst
