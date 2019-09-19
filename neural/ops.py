#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-04-27

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import cv2

import tensorflow.contrib.layers as tflayers

from utils import *


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


def printh(tensor, name, idz=-1):
    shape = tensor.shape
    if idz == -1:
        idz = shape[3] / 2
    print("\n%s shape: %dx%dx%d  indz:%d" % (name, shape[1], shape[2], shape[3], idz))
    max_x = min(20, shape[1])
    max_y = min(14, shape[2])
    o_str = ""
    for i in range(0, max_x):
        o_str += "[" + str(i) + "] "
        for j in range(0, max_y):
            o_str += "\t" + str("%.4f" % tensor[0][i][j][idz])
        o_str += "\n"
    print(o_str)


def printf(tensor, name, idx=-1):
    shape = tensor.shape
    if len(shape) == 4:
        if idx == -1:
            idx = int(shape[1] / 2)
        print("\n%s shape: %dx%dx%d  indx:%d" % (name, shape[1], shape[2], shape[3], idx))
        max_y = min(80, shape[2])
        max_z = min(14, shape[3])
        if shape[3] > 8:
            max_y = min(20, shape[2])
        o_str = ""
        for i in range(0, max_y):
            o_str += "[" + str(i) + "] "
            for j in range(0, max_z):
                o_str += "\t" + str("%.4f" % tensor[0][idx][i][j])
            o_str += "\n"
        print(o_str)
    if len(shape) == 3:
        if idx == -1:
            idx = int(shape[0] / 2)
        print("\n%s shape: %dx%dx%d  indx:%d" % (name, shape[0], shape[1], shape[2], idx))
        max_y = min(80, shape[1])
        max_z = min(14, shape[2])
        if shape[2] > 8:
            max_y = min(20, shape[1])
        o_str = ""
        for i in range(0, max_y):
            o_str += "[" + str(i) + "] "
            for j in range(0, max_z):
                o_str += "\t" + str("%.4f" % tensor[idx][i][j])
            o_str += "\n"
        print(o_str)


def checkzero(tensor, name):
    shape = tensor.shape
    if len(shape) == 4:
        print("\n%s shape: %dx%dx%d  indx" % (name, shape[1], shape[2], shape[3]))
        o_str = ""
        # for k in range(0, shape[1]):
        for i in range(0, shape[2]):
            for j in range(0, shape[3]):
                check = abs(tensor[0][8][i][j]) > 1e-50
                if check:
                    o_str += " \t" + str(8) + "\t" + str(i) + "\t" + str(j) + "\t\t" + str(
                        '%.4f' % (tensor[0][8][i][j])) + "\n"

        print(o_str)
