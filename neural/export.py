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
import threading
import util.logit as log

"""
此工具
1. 用来生成一些规则的model（.bytes）
    可以在引擎里加载 显示对应的脸型
    Unity 菜单选择Tools->SelectModel
2. 将引擎生成的图片转换为edge图片
    多线程里实现
"""


def write_layer(f, shape, args):
    f.write(struct.pack('i', shape))
    for it in args:
        byte = struct.pack('f', it)
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
        file_root, file_name = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(file_root):
            os.makedirs(file_root)  # 创建路径
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
    for _ in range(103):
        args.append(weight / 10.0)
    name = os.path.join(path, str(shape) + "-" + str(weight) + ".bytes")
    f = open(name, 'wb')
    write_layer(f, shape, args)
    f.close()


class Thread_Transfer(threading.Thread):
    """
    批量转换edge图片
    """

    def __init__(self, threadID, root, dir_2, cp, files):
        threading.Thread.__init__(self)
        self.root = root
        self.dir2 = dir_2
        self.files = files
        self.cp = cp
        print("thread {0} files count {1}".format(threadID, len(files)))

    def run(self):
        for name in self.files:
            path1 = os.path.join(self.root, name)
            path2 = os.path.join(self.dir2, name)
            if not os.path.exists(path2):
                shutil.copy(path1, path2)
                self.image_transfer(path2)

    def image_transfer(self, im_path):
        if im_path.find('.jpg') >= 0:
            img = utils.evalute_face(im_path, self.cp, False)
            img = utils.img_edge(img)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            cv2.imwrite(im_path, img)


def batch_transfer(curr_path, export_path):
    """
    批量转换edge图片
    默认开启16个线程， 根据自己电脑cpu核数去自定义此数值 thread_cnt=16
    :param curr_path: current script's path
    :param export_path: 转换目录
    :return:
    """
    if os.path.exists(export_path):
        root, _ = os.path.split(curr_path)
        print("root", root)
        cp = os.path.join(root, "dat/79999_iter.pth")
        print(cp)
        dir2 = export_path + "2"
        if os.path.exists(dir2):
            shutil.rmtree(dir2)
        os.mkdir(dir2)
        thread_cnt = 16
        for root, dirs, files in os.walk(export_path, topdown=False):
            count = len(files)
            split = int(count / thread_cnt)
            thread_pool = []
            for i in range(thread_cnt):
                start = split * i
                end = split * (i + 1)
                end = end if (i < thread_cnt - 1) else count
                files_ = files[start: end]
                thread = Thread_Transfer(i, root, dir2, cp, files_)
                thread.start()
                thread_pool.append(thread)
            for thread in thread_pool:
                thread.join()
    else:
        print("there is not dir ", export_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        work_path = os.path.join(os.getcwd(), sys.argv[0])
        exp_path = sys.argv[1]
        batch_transfer(work_path, exp_path)
    else:
        pwd = os.getcwd()
        project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        model_path = os.path.join(project_path, "unity/models/")
        log.info(model_path)
        shapes = [3, 4]
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        for i in shapes:
            for j in range(0, 10):
                export_layer(model_path, i, j)
