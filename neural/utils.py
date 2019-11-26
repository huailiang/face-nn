#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

from __future__ import division

import random
from lightcnn.extract_features import *
import torch.nn as nn
import torch.nn.functional as F
import util.logit as log
from faceparsing.evaluate import *


def random_params(cnt=99):
    """
    随机生成捏脸参数
    :param cnt: param count
    """
    params = []
    for i in range(cnt):
        params.append(random.randint(0, 1000) / 1000.0)
    return params


def init_params(cnt=99):
    """
    初始捏脸参数
    :param cnt: param count
    """
    params = []
    for i in range(cnt):
        params.append(0.5)
    params[96] = 1
    return params


def param_2_arr(params):
    """
    捏脸参数转numpy array
    """
    cnt = len(params)
    array = np.array(params)
    array = array.reshape([1, 1, 1, cnt])
    return array


def init_weights(m):
    """
    使用正太分布初始化网络权重
    :param m: model
    """
    classcache = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.4, 0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 0.0, 0.2)
        nn.init.constant_(m.bias.data, 0.0)


def deconv_layer(in_chanel, out_chanel, kernel_size, stride=1, pad=0):
    """
    反卷积layer, CT->BN->Relu 主要用于上采样
    公式： output = (input - 1) * stride + outputpadding - 2 * padding + kernel_size
    :param in_chanel: chanel in, int
    :param out_chanel: chanel out, int
    :param kernel_size: triple or int
    :param stride: conv stride
    :param pad: pad
    :return: nn.Sequential
    """
    return nn.Sequential(nn.ConvTranspose2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
                         nn.BatchNorm2d(out_chanel), nn.ReLU())


def conv_layer(in_chanel, out_chanel, kernel_size, stride, pad=0):
    """
    实现一个通用的卷积layer, 卷积->BN->Relu
    公式: output = [input + 2 *padding - (kernel_size - 1)] / stride
    :param in_chanel: chanel in, int
    :param out_chanel: chanel out, int
    :param kernel_size: triple or int
    :param stride: conv stride
    :param pad: pad
    :return: nn.Sequential
    """
    return nn.Sequential(nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
                         nn.BatchNorm2d(out_chanel), nn.Sigmoid())


def load_lightcnn(location, cuda=False):
    """
    load light cnn to memory
    :param location: lightcnn path
    :param cuda: gpu speed up
    :return: 29-layer-v2 light cnn model
    """
    model = LightCNN_29Layers_v2(num_classes=80013)
    # lock_net(model)
    model.eval()
    if cuda:
        checkpoint = torch.load(location)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(location, map_location="cpu")
        new_state_dict = model.state_dict()
        for k, v in checkpoint['state_dict'].items():
            _name = k[7:]  # remove `module.`
            new_state_dict[_name] = v
        model.load_state_dict(new_state_dict)
    return model


def debug_parameters(model, tag="_model_"):
    """
    debug parameters
    :param tag: debug tag
    :param model: net model
    :return:
    """
    log.debug("\n **** %s ****", tag)
    for index, (name, param) in enumerate(model.named_parameters()):
        log.debug("{0}\t{1}\tgrad:{2}\tshape:{3}".format(index, name, param.requires_grad, param.size()))


def lock_net(model, opening=False):
    """
    是否锁住某个model, 锁住之后不再更新权重梯度
    :param model: net model
    :param opening: True 会更新梯度， False 则不会
    :return:
    """
    for param in model.parameters():
        param.requires_grad = opening


def feature256(img, lightcnn_inst):
    """
    使用light cnn提取256维特征参数
    :param lightcnn_inst: lightcnn model instance
    :param img: tensor 输入图片 shape:(batch, 1, 512, 512)
    :return: 256维特征参数 tensor [batch, 256]
    """
    transform = transforms.Compose([transforms.ToTensor()])
    batch = img.size(0)
    feature_tensor = torch.empty(batch, 256)
    for i in range(batch):
        _img = img[i].cpu().detach().numpy()
        _img = _img.reshape((_img.shape[1], _img.shape[2]))
        _img = cv2.resize(_img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        _img = transform(_img)
        _img = _img.view(1, 1, 128, 128)
        _, features = lightcnn_inst(_img)
        feature_tensor[i] = features
    return feature_tensor


def batch_feature256(img, lightcnn_inst):
    """
       使用light cnn提取256维特征参数
       :param lightcnn_inst: lightcnn model instance
       :param img: tensor 输入图片 shape:(batch, 1, 512, 512)
       :return: 256维特征参数 tensor [batch, 256]
       """
    _, features = lightcnn_inst(img)
    # log.debug("features shape:{0} {1} {2}".format(features.size(), features.requires_grad, img.requires_grad))
    return features


def discriminative_loss(img1, img2, lightcnn_inst):
    """
    论文里的判别损失, 判断真实照片和由模拟器生成的图像是否属于同一个身份
    Discriminative Loss 使用余弦距离
    https://www.cnblogs.com/dsgcBlogs/p/8619566.html
    :param lightcnn_inst: lightcnn model instance
    :param img1: generated by engine, type: list of Tensor
    :param img2: generated by imitator, type: list of Tensor
    :return tensor scalar
    """
    x1 = batch_feature256(img1, lightcnn_inst)
    x2 = batch_feature256(img2, lightcnn_inst)
    distance = torch.cosine_similarity(x1, x2)
    return torch.mean(distance)


def evalute_face(img_path, cp, cuda):
    """
    face segmentation model
    :param img_path: cv read image path
    :param cp:  face parsing checkpoint name
    :param cuda: gpu speed up
    :return: face-parsing image
    """
    image = cv2.imread(img_path)
    return faceparsing_ndarray(image, cp=cp, cuda=cuda)


def content_loss(img1, img2):
    """
    change resolution to 1/8, 512/8 = 64
    :param img1: numpy array 64x64
    :param img2: numpy array
    :return: tensor
    """
    image1 = torch.from_numpy(img1)
    image2 = torch.from_numpy(img2)
    image2 = image2.view(64, 64, 1)
    log.info("img1 size {0} img2 size: {1}".format(image1.size(), image2.size()))
    return F.mse_loss(image1, image2)


def img_edge(img):
    """
    提取原始图像的边缘
    :param img: input image
    :return: edge image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x_grad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    y_grad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    return cv2.Canny(x_grad, y_grad, 40, 130)


def save_batch(input_painting_batch, input_photo_batch, output_painting_batch, output_photo_batch, filepath):
    """
    Concatenates, processes and stores batches as image 'filepath'.
    Args:
        input_painting_batch: numpy array of size [B x H x W x C]
        input_photo_batch: numpy array of size [B x H x W x C]
        output_painting_batch: numpy array of size [B x H x W x C]
        output_photo_batch: numpy array of size [B x H x W x C]
        filepath: full name with path of file that we save
    """

    def batch_to_img(batch):
        return np.reshape(batch, newshape=(batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3]))

    inputs = np.concatenate([batch_to_img(input_painting_batch), batch_to_img(input_photo_batch)], axis=0)
    outputs = np.concatenate([batch_to_img(output_painting_batch), batch_to_img(output_photo_batch)], axis=0)
    to_save = np.concatenate([inputs, outputs], axis=1)
    to_save = np.clip(to_save, a_min=0., a_max=255.).astype(np.uint8)
    cv2.imwrite(filepath, to_save)


def normalize_arr_of_imgs(arr):
    """
    Normalizes an array so that the result lies in [-1; 1].
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return arr / 127.5 - 1.  # return (arr - np.mean(arr)) / np.std(arr)


def denormalize_arr_of_imgs(arr):
    """
    Inverse of the normalize_arr_of_imgs function.
    Args:
        arr: numpy array of arbitrary shape and dimensions.
    Returns:
    """
    return (arr + 1.) * 127.5


def update_optimizer_lr(optimizer, lr):
    """
    为了动态更新learning rate， 加快训练速度
    :param optimizer: torch.optim type
    :param lr: learning rate
    :return:
    """
    for group in optimizer.param_groups:
        group['lr'] = lr


def curr_roleshape(dataset):
    """
    判断当前运行的是roleshape (c# RoleShape)
    :param dataset: args path_to_dataset
    :return: RoleShape
    """
    if dataset.find("female") >= 0:
        return 4
    else:
        return 3
