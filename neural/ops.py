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
    :param dir: 文件夹路径
    """
    try:
        if os.path.exists(dir):
            for root, dirs, files in os.walk(dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
        else:
            log.warn("not exist directory: %s", dir)
    except IOError as e:
        log.error("io error, load imitator failed ", e)


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
        log.error("io error, load imitator failed ", e)


def normal_2_image(tensor):
    """
    将tensor转numpy array 给cv2使用
    :param tensor: [batch, c, w, h]
    :return: [batch, h, w, c]
    """
    batch = tensor.size(0)
    images = []
    for i in range(batch):
        img = tensor[i].detach().numpy()
        img = np.swapaxes(img, 0, 2)  # [h, w, c]
        img = np.swapaxes(img, 0, 1)  # [w, h, c]
        images.append(img * 256)
    return images


def save_img(path, tensor1, tensor2):
    """
    save first image of batch to disk
    :param path: save path
    :param tensor1: shape: [Batch, C, W, H)
    :param tensor2: shape: [Batch, C, W, H)
    """
    image1 = normal_2_image(tensor1)
    image2 = normal_2_image(tensor2)
    if len(image1) > 1:
        img = merge_4image(image1[0], image2[0], image1[1], image2[1])
    elif len(image1) > 0:
        img = merge_image(image1[0], image2[0], mode='h')
    else:
        raise NeuralException("tensor error")
    cv2.imwrite(path, img)


def merge_image(image1, image2, mode="h", show=False):
    """
    拼接图片
    :param image1: numpy array
    :param image2: numpy array
    :param mode: 'h': 横向拼接 'v': 纵向拼接
    :param show: 窗口显示
    :return: numpy array
    """
    img1_ = cv2.resize(image1, (256, 256))
    img2_ = cv2.resize(image2, (256, 256))
    if mode == 'h':
        image = np.append(img1_, img2_, axis=1)  # (256, 512, 3)
    elif mode == 'v':
        image2 = np.append(img1_, img2_, axis=0)
    else:
        log.warn("not implements mode: %s", mode)
        return
    if show:
        cv2.imshow("contact", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image


def merge_4image(image1, image2, image3, image4, show=False):
    """
    拼接图片 512x512
    :param image1: input image1
    :param image2: input image2
    :param image3: input image3
    :param image4: input image4
    :param show: 窗口显示
    :return:
    """
    img1 = cv2.resize(image1, (256, 256))
    img2 = cv2.resize(image2, (256, 256))
    img3 = cv2.resize(image3, (256, 256))
    img4 = cv2.resize(image4, (256, 256))
    image1_ = np.append(img1, img2, axis=1)
    image2_ = np.append(img3, img4, axis=1)
    image = np.append(image1_, image2_, axis=0)
    if show:
        cv2.imshow("contact", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return image


if __name__ == '__main__':
    log.init("")
    img1 = cv2.imread("./output/db_3.jpg")
    img2 = cv2.imread("./output/db_4.jpg")
    img3 = cv2.imread("./output/db_1.jpg")
    img4 = cv2.imread("./output/db_2.jpg")
    merge_4image(img1, img2, img3, img4, show=True)

""" 
# tensorflow implement, not use again

def instance_norm(input, name="instance_norm", is_training=True):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def deconv2d(input_, output_dim, ks=4, s=2, name="deconv2d"):
    # Upsampling procedure, like suggested in this article:
    # https://distill.pub/2016/deconv-checkerboard/. At first upsample
    # tensor like an image and then apply convolutions.
    with tf.variable_scope(name):
        input_ = tf.image.resize_images(images=input_, size=tf.shape(input_)[1:3] * s,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # That is optional
        return conv2d(input_=input_, output_dim=output_dim, ks=ks, s=1, padding='SAME')


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
"""
