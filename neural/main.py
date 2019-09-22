#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-20

import tensorflow as tf
from parse import parser
from model import Artgan
from net import Net

import logging
logger = logging.getLogger("nn-face")


tf.set_random_seed(228)


def main(_):
    net = Net(5010, 5011)

    while (True):
        input = raw_input("command: \n")
        if input == "s":
            msg = raw_input("input: ")
            net.onlySend(msg)
        elif input == 'r':
            msg = raw_input("input: ")
            net.sendRcv(msg)
        elif input == "q":
            net.onlySend("quit")
            net.close()
            break
        else:
            logger.info("unknown code, quit")
            net.close()
            break

    # args = parser.parse_args()
    # tfconfig = tf.ConfigProto(allow_soft_placement=False)
    # tfconfig.gpu_options.allow_growth = True
    # with tf.Session(config=tfconfig) as sess:
    #     model = Artgan(sess, args)

    #     if args.phase == 'train':
    #         print("Train.")
    #         model.train(args, ckpt_nmbr=args.ckpt_nmbr)



if __name__ == '__main__':
    tf.app.run()
