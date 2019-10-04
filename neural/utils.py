#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

from __future__ import division
import math
import scipy.misc
import torch
import numpy as np
from ops import *
import random
from lightcnn.extract_features import *


def random_params(cnt):
    """
    随机生成捏脸参数
    """
    params = []
    for i in range(cnt):
        params.append(random.randint(0, 1000) / 1000.0)
    return params


def param_2_arr(params):
    """
    捏脸参数转numpy array
    """
    cnt = len(params)
    array = np.array(params)
    array = array.reshape([1, 1, 1, cnt])
    return array


def feature256(img):
    """
    使用light cnn提取256维特征参数
    :param img: 输入图片
    :return: 256维特征参数
    """
    model = LightCNN_29Layers_v2(num_classes=79077)
    model.eval()
    model = torch.nn.DataParallel(model).cuda()
    transform = transforms.Compose([transforms.ToTensor()])
    img = np.reshape(img, (128, 128, 1))
    img = transform(img)
    input[0, :, :, :] = img
    input_var = torch.autograd.Variable(input, volatile=True)
    _, features = model(input_var)
    return features


def save_batch(input_painting_batch, input_photo_batch, output_painting_batch, output_photo_batch, filepath):
    """
    Concatenates, processes and stores batches as image 'filepath'.
    Args:
        input_painting_batch: numpy array of size [B x H x W x C]
        input_photo_batch: numpy array of size [B x H x W x C]
        output_painting_batch: numpy array of size [B x H x W x C]
        output_photo_batch: numpy array of size [B x H x W x C]
        filepath: full name with path of file that we save

    Returns:

    """

    def batch_to_img(batch):
        return np.reshape(batch, newshape=(batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3]))

    inputs = np.concatenate([batch_to_img(input_painting_batch), batch_to_img(input_photo_batch)], axis=0)
    outputs = np.concatenate([batch_to_img(output_painting_batch), batch_to_img(output_photo_batch)], axis=0)
    to_save = np.concatenate([inputs, outputs], axis=1)
    to_save = np.clip(to_save, a_min=0., a_max=255.).astype(np.uint8)
    scipy.misc.imsave(filepath, arr=to_save)


def normalize_arr_of_imgs(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return arr / 127.5 - 1.  # return (arr - np.mean(arr)) / np.std(arr)


def denormalize_arr_of_imgs(arr):
    """
    Inverse of the normalize_arr_of_imgs function.
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return (arr + 1.) * 127.5
