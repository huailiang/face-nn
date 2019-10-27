#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-20


import utils
import logging
import torch
import align
import cv2
import util.logit as log
from dataset import FaceDataset
from imitator import Imitator
from extractor import Extractor
from parse import parser


def init_device(args):
    """
    检查配置和硬件是否支持gpu
    :param args: 配置
    :return: 返回True 则支持gpu
    """
    support_gpu = torch.cuda.is_available()
    log.info("neural face network use gpu: %s", support_gpu and args.use_gpu)
    if support_gpu and args.use_gpu:
        if not args.gpuid:
            args.gpuid = 0
        dev = torch.device("cuda:%d" % args.gpuid)
        return True, dev
    else:
        dev = torch.device("cpu")
        return False, dev


if __name__ == '__main__':
    """
    程序入口函数
    """
    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/neural_log.txt")
    cuda, device = init_device(args)

    if args.phase == "train_imitator":
        log.info('imitator train mode')
        imitator = Imitator("neural imitator", args)
        if cuda:
            imitator.cuda()
        imitator.batch_train(cuda)
    elif args.phase == "train_extractor":
        log.info('feature extractor train mode')
        extractor = Extractor("neural extractor", args)
        if cuda:
            extractor.cuda()
        extractor.batch_train(cuda)
    elif args.phase == "inference_imitator":
        log.info("inference imitator")
        imitator = Imitator("neural imitator", args, clean=False)
        if cuda:
            imitator.cuda()
        imitator.load_checkpoint("model_imitator_14000.pth", training=True, cuda=cuda)
    elif args.phase == "lightcnn":
        log.info("light cnn test")
        checkpoint = torch.load("./dat/LightCNN_29Layers_V2_checkpoint.pth.tar", map_location="cpu")
        img = torch.randn(1, 3, 512, 512)
        features = utils.feature256(img, checkpoint)
        log.info(features.size())
    elif args.phase == "faceparsing":
        log.info("faceparsing")
        im = utils.evalute_face("./output/face/db_0000_3.jpg", args.extractor_checkpoint, True)
        cv2.imwrite("./output/eval.jpg", im)
    elif args.phase == "align":
        align.face_features("./output/image/timg.jpeg", "test.jpg")
    elif args.phase == "dataset":
        dataset = FaceDataset(args, "test")
        dataset.pre_process(cuda)
    else:
        log.error("not known phase %s", args.phase)
