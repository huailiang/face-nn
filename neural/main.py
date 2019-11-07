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
import util.logit as log
import numpy as np
from dataset import FaceDataset
from imitator import Imitator
from extractor import Extractor
from evaluate import Evaluate
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
        imitator.load_checkpoint(args.imitator_model, True, cuda=cuda)
    elif args.phase == "prev_imitator":
        log.info("preview imitator")
        imitator = Imitator("neural imitator", args, clean=False)
        imitator.load_checkpoint(args.imitator_model, False, cuda=False)
        dataset = FaceDataset(args)
        name, param, img = dataset.get_picture()
        param = np.array(param, dtype=np.float32)
        b_param = param[np.newaxis, :]
        log.info(b_param.shape)
        t_param = torch.from_numpy(b_param)
        output = imitator(t_param)
        output = output.cpu().detach().numpy()
        output = np.squeeze(output, axis=0)
        output = output.swapaxes(0, 2) * 255
        cv2.imwrite('./output/{0}.jpg'.format(name), output)
    elif args.phase == "inference_extractor":
        log.info("inference extractor")
        extractor = Extractor("neural extractor", args)
        if cuda:
            extractor.cuda()
        extractor.load_checkpoint("model_extractor_845000.pth", True, cuda)
    elif args.phase == "lightcnn":
        log.info("light cnn test")
        lightcnn_inst = utils.load_lightcnn(args.lightcnn, cuda)
        img = torch.ones(1, 1, 512, 512)
        features = utils.feature256(img, lightcnn_inst)
        log.info(features.size())
        log.info("features: {0} {1}".format(features[0][1], features[0][0]))
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
    elif args.phase == "evaluate":
        log.info("evaluation mode start")
        evl = Evaluate(args, cuda=cuda)
        img = cv2.imread(args.eval_image).astype(np.float32)
        x_ = evl.itr_train(img)
        evl.output(x_, img)
    else:
        log.error("not known phase %s", args.phase)
