#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-20


import utils
import ops
import logging
import torch
import align
import cv2
import os
import util.logit as log
from dataset import FaceDataset
from imitator import Imitator
from extractor import Extractor
from parse import parser


def init_device(arguments):
    """
    检查配置和硬件是否支持gpu
    :param arguments: 配置
    :return: 返回True 则支持gpu
    """
    support_gpu = torch.cuda.is_available()
    log.info("neural face network use gpu: %s", support_gpu and arguments.use_gpu)
    if support_gpu and arguments.use_gpu:
        if not arguments.gpuid:
            arguments.gpuid = 0
        dev = torch.device("cuda:%d" % arguments.gpuid)
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
        imitator.load_checkpoint("model_imitator_130000.pth", True, cuda=cuda)
    elif args.phase == "inference_extractor":
        log.info("inference extractor")
        extractor = Extractor("neural extractor", args)
        if cuda:
            extractor.cuda()
        extractor.load_checkpoint("model_extractor_845000.pth", True, cuda)
    elif args.phase == "lightcnn":
        log.info("light cnn test")
        checkpoint = torch.load(args.lightcnn, map_location="cpu")
        img = torch.randn(1, 3, 512, 512)
        features = utils.feature256(img, checkpoint)
        log.info(features.size())
    elif args.phase == "faceparsing":
        log.info("faceparsing")
        im = utils.evalute_face("./output/face/db_0000_3.jpg", args.parsing_checkpoint, cuda)
        cv2.imwrite("./output/eval.jpg", im)
    elif args.phase == "align":
        align.face_features("../export/regular/model.jpg", "../export/regular/out.jpg")
    elif args.phase == "dataset":
        dataset = FaceDataset(args, "test")
        dataset.pre_process(cuda)
    elif args.phase == "preview":
        log.info(" preview picture ")
        path = "../export/regular/model.jpg"
        img = cv2.imread(path)
        img2 = utils.parse_evaluate(img, args.parsing_checkpoint, cuda)
        img3 = utils.img_edge(img2)
        img3_ = ops.fill_grey(img3)
        img4 = align.face_features(path)
        log.info("{0} {1} {2} {3}".format(img.shape, img2.shape, img3_.shape, img4.shape))
        ops.merge_4image(img, img2, img3_, img4, show=True)
    elif args.phase == "cache":
        dataset = FaceDataset(args, "train")
        image1, image2, name = dataset.get_cache(cuda)
        orig_path = os.path.join(args.path_to_dataset + "2", name)
        orig_img = cv2.imread(orig_path)
        parse_img = utils.parse_evaluate(orig_img, args.parsing_checkpoint, cuda)
        ops.save_extractor('../export/test.jpg', image1, image2, orig_img, parse_img)
    else:
        log.error("not known phase %s", args.phase)
