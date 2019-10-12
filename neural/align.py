#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-01

import os
import shutil
import struct
import dlib
import cv2
import numpy as np
import util.logit as log


def generate_detector():
    predictor = dlib.shape_predictor('dat/shape_predictor_68_face_landmarks.dat')
    facerec = dlib.face_recognition_model_v1('dat/dlib_face_recognition_resnet_model_v1.dat')
    detector = dlib.get_frontal_face_detector()
    return detector, predictor, facerec


def face_features(path_img, path_save):
    try:
        img = cv2.imread(path_img)
        if img.shape[0] * img.shape[1] > 512 * 512:
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        detector, predictor, facerec = generate_detector()
        dets = detector(img, 1)  # 使用检测算子检测人脸，返回的是所有的检测到的人脸区域
        print("检测的人脸图像：", path_img, "\n")
        d = dets[0]  # 默认处理第一个检测到的人脸区域
        bb = np.zeros(4, dtype=np.int32)

        ext = 0
        bb[0] = np.maximum(d.left() - ext, 0)
        bb[1] = np.maximum(d.top() - ext, 0)
        bb[2] = np.minimum(d.right() + ext, img.shape[1])
        bb[3] = np.minimum(d.bottom() + ext, img.shape[0])

        rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        shape = predictor(img, rec)  # 获取landmark
        face_descriptor = facerec.compute_face_descriptor(img, shape)  # 使用resNet获取128维的人脸特征向量
        face_array = np.array(face_descriptor).reshape((1, 128))  # 转换成numpy中的数据结构

        # 显示人脸区域
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 255, 255), 1)
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        scaled = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(path_save, img)
        cv2.imwrite(path_save.replace("align_", "align2_"), scaled)
    except Exception as e:
        log.error(e.message)


def clean(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith("align"):
                path = os.path.join(root, file)
                os.remove(path)


def export(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            path1 = os.path.join(root, file)
            path2 = os.path.join(root, "align_" + file)
            face_features(path1, path2)


if __name__ == '__main__':
    pwd = os.getcwd()
    project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    model_path = os.path.join(project_path, "export/image/")
    log.info(model_path)
    clean(model_path)
    export(model_path)
