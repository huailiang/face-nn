#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-20


import utils
from imitator import Imitator
from feature_extractor import FeatureExtractor
from net import Net
from parse import parser
import logging
import torch
import align
import util.logit as log


def ex_net():
    """
    建立和引擎的通信
    python中启动之后， unity菜单栏选中Tools->Connect
    """
    net = Net(5010, 5011)
    while True:
        r_input = input("command: \n")
        if r_input == "s":
            msg = input("input: ")
            net.only_send(msg)
        elif r_input == 'r':
            msg = input("input: ")
            net.send_recv(msg)
        elif r_input == "q":
            net.only_send("quit")
            net.close()
            break
        else:
            log.error("unknown code, quit")
            net.close()
            break


def init_device(args):
    """
    检查配置和硬件是否支持gpu
    :param args: 配置
    :return: 返回True 则支持gpu
    """
    support_gpu = torch.cuda.is_available()
    log.info("neural face network support gpu %s, use gpu %s", support_gpu, args.use_gpu)
    if support_gpu and args.use_gpu:
        if not args.gpuid:
            args.gpuid = 0
        device = torch.device("cuda:%d" % args.gpuid)
        return True, device
    else:
        device = torch.device("cpu")
        return False, device


if __name__ == '__main__':
    """
    程序入口函数
    """
    args = parser.parse_args()
    log.init("FaceNeural", logging.DEBUG, log_path="./output/neural_log.txt")
    init_device(args)

    if args.phase == "train_imitator":
        log.info('imitator train mode')
        imitator = Imitator("neural imitator", args)
        imitator.batch_train()
    elif args.phase == "train_extractor":
        log.info('feature extractor train mode')
        extractor = FeatureExtractor("neural extractor", args)
        extractor.batch_train()
    elif args.phase == "inference_imitator":
        log.info("inference imitator")
        imitator = Imitator("neural imitator", args)
        imitator.load_checkpoint("./output/imitator/")
    elif args.phase == "lightcnn":
        log.info("light cnn test")
        checkpoint = torch.load("./dat/LightCNN_29Layers_V2_checkpoint.pth.tar", map_location="cpu")
        img = torch.randn(1, 3, 512, 512)
        features = utils.feature256(img, checkpoint)
        log.info(features.size())
    elif args.phase == "faceparsing":
        log.info("faceparsing")
    elif args.phase == "net":
        log.info("net start with ports (%d, %d)", 5010, 5011)
        ex_net()
    elif args.phase == "align":
        align.face_features("./output/image/timg.jpeg", "test.jpg")
    else:
        log.error("not known phase %s", args.phase)

