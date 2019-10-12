#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-20

import tensorflow as tf
from model import Face
from net import Net
from module import *
from parse import parser
import util.logit as log
import logging

tf.set_random_seed(228)


def main(_):
    args = parser.parse_args()
    log.init("FaceNeural", logging.DEBUG, log_path="output/log.txt")

    with tf.Session() as sess:
        if args.phase == "train":
            model = Face(sess, args)
            model.train(args)
            log.info('train mode')
        elif args.phase == "inference":
            log.info("inference")
            model = Face(sess, args)
            model.inference(args)
        elif args.phase == "lightcnn":
            log.info("light cnn test")
        elif args.phase == "faceparsing":
            log.info("faceparsing")
        elif args.phase == "net":
            log.info("net start with ports (%d, %d)", 5010, 5011)
            net = Net(5010, 5011)
            while True:
                r_input = raw_input("command: \n")
                if r_input == "s":
                    msg = raw_input("input: ")
                    net.only_send(msg)
                elif r_input == 'r':
                    msg = raw_input("input: ")
                    net.send_recv(msg)
                elif r_input == "q":
                    net.only_send("quit")
                    net.close()
                    break
                else:
                    log.error("unknown code, quit")
                    net.close()
                    break


if __name__ == '__main__':
    tf.app.run()
