#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-27

import math
import numpy as np
import os
import shutil
import util.logit as log


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
    try:
        dir = os.path.pardir(path)
        if not os.path.exists(path):
            os.mkdir(dir)
        f = open(path, 'bw')
        f.write(content)
        f.close()
    except IOError as e:
        log.error("io error, load imitator failed ", e)


""" 
# tensorflow implement
# not use again

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


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", activation_fn=None):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=activation_fn,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=None)


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
