# encoding=utf8

import os
import tensorflow as tf
from ops import *
import imageio
import numpy as np

from tqdm import tqdm
import multiprocessing

# encoding:utf-8
from multiprocessing import Process
import os, time, random


def r():
    print 'run method'


import time
from tqdm import tqdm

for i in tqdm(range(10, 20)):
    # print(i)
    time.sleep(1)

# if __name__ == "__main__":
#         print "main process run..."
#         #没有指定Process的targt
#         p1 = Process()
#         p2 = Process()
#         #如果在创建Process时不指定target，那么执行时没有任何效果。
#         #因为默认的run方法是判断如果不指定target，那就什么都不做
#         #所以这里手动改变了run方法
#         p1.run = r
#         p2.run = r

#         p1.start()
#         p2.start()
#         print("********")
#         p1.join()
#         p2.join()
#         time.sleep(0.001)
#         print "main process runned all lines..."

#
# tf.set_random_seed(228)
#
# # img = tf.Variable(tf.ones([1,80,80,256]))
# # print img.get_shape()
# # print(img.get_shape().as_list()[-1])
#
#
# # 用来测试tf.nn.moments & keep_dims
#
# # img = tf.Variable(tf.ones([32,32,4]))
# # axis=[0,1]
# # mean,variance = tf.nn.moments(img,axis,keep_dims=False)
# # print(img)
# # print(mean)
# # print(variance)
#
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     resultMean = sess.run(mean)
# #     print(resultMean)
# #     resultMean = sess.run(variance)
# #     print(resultMean)
#
#
# '''
# # 用来测试不同维度的tensor相加的结果
# a1 = tf.constant([1,2,3])
# a2 = tf.constant([1])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     v = sess.run(a1+a2)
#     print(v)
# '''
#
# '''
#
# # 用来测试tf.pad
# a1 = tf.constant([[2,3,4],[5,6,7]])
# print(a1.get_shape())
# pad = tf.pad(a1, paddings=[[1,1],[2,2]],mode="CONSTANT");
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     v = sess.run(pad)
#     print(pad.get_shape())
#
# '''
#

# img_path = "./RT.jpg"
# img = imageio.imread(img_path)
# printf(img, "img")  # tensor = tf.convert_to_tensor(img)
# tensor = tf.image.resize_images(images=tensor, size=tf.shape(tensor)[0:2] * 2,
#                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
# with tf.Session() as sess:
#     img = sess.run(tensor)
#     imageio.imwrite("./data/app1_stylized.jpg", img)

#
# newimg = np.zeros([512,  512, 3])
#
# shape = img.shape
#
# x = shape[0]
# y = shape[1]
# z = shape[2]
#
# xstr = ""
# for i in xrange(0, x):
#     for j in xrange(0, y):
#         for k in xrange(0, z):
#             v = img[i, j, k]
#             xstr += str('%.4f\t' % v)
#             newimg[2*i, 2*j, k] = v
#             newimg[2 * i+1, 2 * j, k] = v
#             newimg[2 * i, 2 * j + 1, k] = v
#             newimg[2 * i +1, 2 * j + 1, k] = v
#         xstr += "\n"
# print(xstr)
# imageio.imwrite("./data/app2_stylized.jpg", newimg)
#
#
#
