#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-27

import cv2
import os
import shutil
import numpy as np
import util.logit as log
from util.exception import NeuralException


def rm_dir(path):
    """
    清空文件夹 包含子文件夹
    :param path: 文件夹路径
    """
    try:
        if os.path.exists(path):
            log.warn("rm directory %s", path)
            shutil.rmtree(path)
        else:
            log.warn("not exist directory: %s", path)
    except IOError as e:
        log.error("io error, load imitator failed ", e)


def clear_folder(dir):
    """
    清空文件夹 包含子文件夹
    :param dir: 文件夹路径
    """
    try:
        if os.path.exists(dir):
            for root, dirs, files in os.walk(dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        else:
            log.warn("not exist directory: %s", dir)
    except IOError as e:
        log.error("io error, load imitator failed ", e)


def clear_files(dir):
    """
    只清空文件 不清空子文件夹
    如果文件夹不存在 则创建一个新的文件夹
    :param dir: 文件夹路径
    """
    try:
        if os.path.exists(dir):
            for root, dirs, files in os.walk(dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
        else:
            log.warn("not exist directory: %s, create new", dir)
            os.mkdir(dir)
    except IOError as e:
        log.error("io error, load imitator failed {0}".format(e))


def get_imit_cp(dir, ext=None):
    """
    匹配查找最近生成的checkpoint
    :param dir: search directory
    :param ext: 文件后缀
    """
    try:
        m_time = 0
        rst = None
        for file in os.listdir(dir):
            path = os.path.join(dir, file)
            time = os.path.getmtime(path)
            match = True
            if ext is not None:
                match = file.endswith(ext)
            if match and (m_time == 0 or time > m_time):
                m_time = time
                rst = path
        return rst
    except IOError as e:
        log.error('get_imit_cp, io error')


def generate_file(path, content):
    """
    生成文件
    :param path: file path
    :param content: file content
    :return:
    """
    try:
        dir = os.path.pardir(path)
        if not os.path.exists(path):
            os.mkdir(dir)
        f = open(path, 'bw')
        f.write(content)
        f.close()
    except IOError as e:
        log.error("io error, load imitator failed {0}".format(e))


def to_gray(rgb):
    """
    灰度处理
    :param rgb: Tensor(RGB)
    :return: Tensor(Gray)
    """
    if len(rgb.shape) >= 3:
        arr = np.mean(rgb, axis=2)
        return arr[:, :, np.newaxis]
    else:
        raise NeuralException("to gray error")


def fill_gray(image):
    """
    [W, H, 1] -> [W, H, 3] or [W, H]->[W, H, 3]
    :param image: input image
    :return: transfer image
    """
    shape = image.shape
    if len(shape) == 2:
        image = image[:, :, np.newaxis]
        shape = image.shape
    if shape[2] == 1:
        return np.pad(image, ((0, 0), (0, 0), (1, 1)), 'edge')
    elif shape[2] == 3:
        return np.mean(image, axis=2)
    return image


def tensor_2_image(tensor):
    """
    将tensor转numpy array 给cv2使用
    :param tensor: [batch, c, w, h]
    :return: [batch, h, w, c]
    """
    batch = tensor.size(0)
    images = []
    for i in range(batch):
        img = tensor[i].cpu().detach().numpy()
        img = np.swapaxes(img, 0, 2)  # [h, w, c]
        img = np.swapaxes(img, 0, 1)  # [w, h, c]
        images.append(img * 255)
    return images


def save_img(path, tensor1, tensor2):
    """
    save first image of batch to disk
    :param path: save path
    :param tensor1: shape: [Batch, C, W, H)
    :param tensor2: shape: [Batch, C, W, H)
    """
    image1 = tensor_2_image(tensor1)
    image2 = tensor_2_image(tensor2)
    if len(image1) > 1:
        img = merge_4image(image1[0], image2[0], image1[1], image2[1])
    elif len(image1) > 0:
        img = merge_image(image1[0], image2[0], mode='h')
    else:
        raise NeuralException("tensor error")
    cv2.imwrite(path, img)


def merge_image(image1, image2, mode="h", size=512, show=False, transpose=True):
    """
    拼接图片
    :param image1: numpy array
    :param image2: numpy array
    :param mode: 'h': 横向拼接 'v': 纵向拼接
    :param size: 输出分辨率
    :param show: 窗口显示
    :param transpose: 转置长和宽 cv2顺序[H, W, C]
    :return: numpy array
    """
    size_ = (int(size / 2), int(size / 2))
    img1_ = cv2.resize(image1, size_)
    img2_ = cv2.resize(image2, size_)
    if mode == 'h':
        image = np.append(img1_, img2_, axis=1)  # (256, 512, 3)
    elif mode == 'v':
        image = np.append(img1_, img2_, axis=0)
    else:
        log.warn("not implements mode: %s".format(mode))
        return
    if transpose:
        image = image.swapaxes(0, 1)
    if show:
        cv2.imshow("contact", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image


def merge_4image(image1, image2, image3, image4, size=512, show=False, transpose=True):
    """
    拼接图片
    :param image1: input image1, numpy array
    :param image2: input image2, numpy array
    :param image3: input image3, numpy array
    :param image4: input image4, numpy array
    :param size: 输出分辨率
    :param show: 窗口显示
    :param transpose: 转置长和宽 cv2顺序[H, W, C]
    :return: merged image
    """
    size_ = (int(size / 2), int(size / 2))
    img_1 = cv2.resize(image1, size_)
    img_2 = cv2.resize(image2, size_)
    img_3 = cv2.resize(image3, size_)
    img_4 = cv2.resize(image4, size_)
    image1_ = np.append(img_1, img_2, axis=1)
    image2_ = np.append(img_3, img_4, axis=1)
    image = np.append(image1_, image2_, axis=0)
    if transpose:
        image = image.swapaxes(0, 1)
    if show:
        cv2.imshow("contact", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image
