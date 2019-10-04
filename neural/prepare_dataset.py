#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

from __future__ import print_function
import numpy as np
import os
from tqdm import tqdm
import scipy.misc
import random
import struct


class ArtDataset():
    def __init__(self, path_to_art_dataset):

        self.dataset = [os.path.join(path_to_art_dataset, x) for x in os.listdir(path_to_art_dataset)]
        print("Art dataset contains %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Reads data from dataframe data containing path to images in column 'path' and, in case of dataframe,
         also containing artist name, technique name, and period of creation for given artist.
         In case of content images we have only the 'path' column.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch
        Returns:
            dictionary with fields: image
        """

        batch_image = []

        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')

            if max(image.shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image.shape))
            if max(image.shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image.shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            if augmentor:
                batch_image.append(augmentor(image).astype(np.float32))
            else:
                batch_image.append((image).astype(np.float32))
        # Now return a batch in correct form
        batch_image = np.asarray(batch_image)
        return {"image": batch_image}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)


class CocoDataset():

    def __init__(self, path_to_dataset):
        self.dataset = []
        print(path_to_dataset)
        if os.path.exists(path_to_dataset):
            for file_name in tqdm(os.listdir(path_to_dataset)):
                self.dataset.append(os.path.join(path_to_dataset, file_name))
        else:
            print("can't be found path %s. Skip it." % path_to_dataset)

        print("Finished. Constructed Places2 dataset of %d images." % len(self.dataset))

    def get_batch(self, augmentor, batch_size=1):
        """
        Generate bathes of images with attached labels(place category) in two different formats:
        textual and one-hot-encoded.
        Args:
            augmentor: Augmentor object responsible for augmentation pipeline
            batch_size: size of batch we return
        Returns:
            dictionary with fields: image
        """

        batch_image = []
        for _ in range(batch_size):
            image = scipy.misc.imread(name=random.choice(self.dataset), mode='RGB')
            image = scipy.misc.imresize(image, size=2.)
            image_shape = image.shape

            if max(image_shape) > 1800.:
                image = scipy.misc.imresize(image, size=1800. / max(image_shape))
            if max(image_shape) < 800:
                # Resize the smallest side of the image to 800px
                alpha = 800. / float(min(image_shape))
                if alpha < 4.:
                    image = scipy.misc.imresize(image, size=alpha)
                    image = np.expand_dims(image, axis=0)
                else:
                    image = scipy.misc.imresize(image, size=[800, 800])

            batch_image.append(augmentor(image).astype(np.float32))

        return {"image": np.asarray(batch_image)}

    def initialize_batch_worker(self, queue, augmentor, batch_size=1, seed=228):
        np.random.seed(seed)
        while True:
            batch = self.get_batch(augmentor=augmentor, batch_size=batch_size)
            queue.put(batch)


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
            print(path)
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
            # print("info: ", key, len(val), path, image.shape)
            batch_rst[key] = (val, image)
            return batch_rst
