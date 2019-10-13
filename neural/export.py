#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-19

import os
import shutil
import struct
import util.logit as log

"""
此工具用来生成一些规则的model（.bytes）
可以在引擎里加载 显示对应的脸型
Unity 菜单选择Tools->SelectModel
"""


def write_layer(f, shape, args):
    f.write(struct.pack('i', shape))
    for i in range(0, 95):
        byte = struct.pack('f', args[i])
        f.write(byte)


def movefile(srcfile, dstfile):
    """
     move file from source to destination
    :param srcfile:  source path
    :param dstfile:  destination path
    """
    if not os.path.isfile(srcfile):
        log.info("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.move(srcfile, dstfile)  # 移动文件
        log.info("move %s -> %s" % (srcfile, dstfile))


def move2unity(name):
    pwd = os.getcwd()
    project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    source = project_path + "/python/" + name
    destination = project_path + "/unity/Assets/Resources/" + name
    movefile(source, destination)


def export_layer(path, shape, weight):
    shape = shape
    args = []
    for i in range(0, 95):
        args.append(weight / 10.0)
    name = os.path.join(path, str(shape) + "-" + str(weight) + ".bytes")
    f = open(name, 'wb')
    write_layer(f, shape, args)
    f.close()


if __name__ == '__main__':
    pwd = os.getcwd()
    project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    model_path = os.path.join(project_path, "unity/models/")
    log.info(model_path)

    shapes = [3, 4]
    for i in shapes:
        for j in range(0, 10):
            export_layer(model_path, i, j)
