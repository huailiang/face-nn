#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-19

import os
import shutil
import struct
import cv2
import sys
import utils
import util.logit as log

"""
此工具
1. 用来生成一些规则的model（.bytes）
    可以在引擎里加载 显示对应的脸型
    Unity 菜单选择Tools->SelectModel
2. 将引擎生成的图片转换为edge图片
"""


def write_layer(f, shape, args):
    f.write(struct.pack('i', shape))
    for i in range(0, 95):
        byte = struct.pack('f', args[i])
        f.write(byte)


def move_file(srcfile, dstfile):
    """
     move file from source to destination
    :param srcfile:  source path
    :param dstfile:  destination path
    """
    if not os.path.isfile(srcfile):
        log.info("%s not exist!" % srcfile)
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        log.info("move %s -> %s" % (srcfile, dstfile))


def move2unity(name):
    current_path = os.getcwd()
    proj_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    source = proj_path + "/python/" + name
    destination = proj_path + "/unity/Assets/Resources/" + name
    move_file(source, destination)


def export_layer(path, shape, weight):
    shape = shape
    args = []
    for _ in range(0, 95):
        args.append(weight / 10.0)
    name = os.path.join(path, str(shape) + "-" + str(weight) + ".bytes")
    f = open(name, 'wb')
    write_layer(f, shape, args)
    f.close()


def batch_transfer(dir):
    """
    批量转换edge图片
    :param dir: 转换目录
    :return:
    """
    if os.path.exists(dir):
        dir_2 = dir + "_2"
        if os.path.exists(dir_2):
            shutil.rmtree(dir_2)
        os.mkdir(dir_2)
        for root, dirs, files in os.walk(dir, topdown=False):
            for name in files:
                path1 = os.path.join(root, name)
                path2 = os.path.join(dir_2, name)
                if not os.path.exists(path2):
                    shutil.copy(path1, path2)
                    image_transfer(path2)
    else:
        print("there is not dir ", dir)


def image_transfer(im_path):
    print(im_path)
    if im_path.find('.jpg') >= 0:
        img = utils.evalute_face(im_path, "./dat/79999_iter.pth", False)
        img = utils.img_edge(img)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        cv2.imwrite(im_path, img)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        batch_transfer(path)
    else:
        pwd = os.getcwd()
        project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        model_path = os.path.join(project_path, "unity/models/")
        log.info(model_path)
        shapes = [3, 4]
        for i in shapes:
            for j in range(0, 10):
                export_layer(model_path, i, j)
