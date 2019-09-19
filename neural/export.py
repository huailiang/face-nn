#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-04-2

import os
import shutil
import struct

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

tf.set_random_seed(228)


def write(f, key, tensor):
    """
        write tensor to file stream
        :param f: 	writable file handler
        :param key: tensor name
        """
    shape = tensor.shape
    f.write(struct.pack('h', len(shape)))
    f.write(struct.pack('h' + str(len(key)) + 's', len(key), key.encode('utf-8')))
    if len(shape) == 1:
        f.write(struct.pack('h', shape[0]))
        for x in range(0, shape[0]):
            byte = struct.pack('f', tensor[x])
            f.write(byte)
    elif len(shape) == 4:
        f.write(struct.pack('h', shape[2]))
        f.write(struct.pack('h', shape[3]))
        f.write(struct.pack('h', shape[0]))
        f.write(struct.pack('h', shape[1]))

        for i in range(0, shape[2]):  # input count
            for j in range(0, shape[3]):  # output count
                for k in range(0, shape[0]):  # kernel height
                    for l in range(0, shape[1]):  # kernel width
                        byte = struct.pack('f', tensor[k, l, i, j])
                        f.write(byte)
    else:
        print("not handle shape: " + shape)


def warite_layer(f, tensor):
    shape = tensor.shape
    f.write(struct.pack('h', len(shape)))
    f.write(struct.pack('h', shape[1]))
    f.write(struct.pack('h', shape[2]))
    f.write(struct.pack('h', shape[3]))
    for i in range(0, shape[1]):
        for j in range(0, shape[2]):
            for k in range(0, shape[3]):
                byte = struct.pack('f', tensor[0, i, j, k])
                f.write(byte)


def movefile(srcfile, dstfile):
    """
     move file from source to destination
    :param srcfile:  source path
    :param dstfile:  destination path
    """
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        print("move %s -> %s" % (srcfile, dstfile))


def move2unity(name):
    pwd = os.getcwd()
    project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    source = project_path + "/python/" + name
    destination = project_path + "/unity/Assets/Resources/" + name
    movefile(source, destination)


def print_tensor(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("len: ", len(var_to_shape_map))
    llist = []
    for key in var_to_shape_map:
        shape = reader.get_tensor(key).shape
        if not key.endswith("Adam_1") and not key.endswith("Adam") and not key.startswith("discriminator") and len(
                shape) > 0:
            print(key.replace("/", "_"), shape)
            llist.append(key.replace("/", "_") + "  " + str(shape))  # tensor = reader.get_tensor(key)
    llist.sort()
    for x in llist:
        print(x)


def export_args(checkpoint_path):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print("len: ", len(var_to_shape_map))
    counter = 0
    name = "args.bytes"
    f = open(name, 'wb')
    for key in var_to_shape_map:
        shape = reader.get_tensor(key).shape
        if not key.endswith("Adam_1") and not key.endswith("Adam") \
                and not key.startswith("discriminator") and len(shape) > 0 \
                and key.find("_r2_") < 0 and key.find("_r3_") < 0 and key.find("_r4_") < 0 \
                and key.find("_r5_") < 0 and key.find("_r6_") < 0 and key.find("_r7_") < 0 \
                and key.find("_r8_") < 0 and key.find("_r9_") < 0:
            print(key.replace("/", "_"), shape)
            for x in shape:
                counter += x
            write(f, key.replace("/", "_"), reader.get_tensor(key))
    f.write(struct.pack('h', 0))  # write 0 stands for eof
    print("done, network arg memory: %dMB" % ((counter * 4) / 1024))
    f.close()
    move2unity(name)


def export_layer(tensor, name):
    shape = tensor.shape
    if len(shape) == 4:
        print("\n export %s shape: %dx%dx%d" % (name, shape[1], shape[2], shape[3]))
        f = open(name + ".bytes", 'w')
        warite_layer(f, tensor)
        f.close()
        print("write layer finish")
        move2unity(name + ".bytes")


if __name__ == '__main__':
    checkpoint_path = "./models/model_van-gogh/checkpoint_long/model16_van-gogh_bks10_flw100_300000.ckpt-300000"
    export_args(checkpoint_path)
