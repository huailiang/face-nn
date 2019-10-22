#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

from __future__ import division

import random
import scipy.misc
from util.exception import NeuralException
from lightcnn.extract_features import *
import torch.nn as nn
import util.logit as log
from faceparsing.evaluate import *


def random_params(cnt):
    """
    随机生成捏脸参数
    """
    params = []
    for i in range(cnt):
        params.append(random.randint(0, 1000) / 1000.0)
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
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # nn.init.uniform(m.weight, a=0., b=1.)
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.2)
        nn.init.constant_(m.bias.data, 0.0)


def to_gray(rgb):
    """
    灰度处理
    :param rgb: Tensor(RGB)
    :return: Tensor(Grey)
    """
    if len(rgb.shape) >= 3:
        arr = np.mean(rgb, axis=2)
        return arr[:, :, np.newaxis]
    else:
        raise NeuralException("to gray error")


def residule_layer(in_chanel, out_chanel, kernel_size=3, stride=1, pad=1):
    nn.Conv2d(in_chanel, out_chanel, kernel_size=3, stride=stride, padding=pad)


def conv_layer(in_chanel, out_chanel, kernel_size, stride, pad=0):
    """
    实现一个通用的卷积layer, 卷积->BN->Relu
    :param in_chanel: chanel in, int
    :param out_chanel: chanel out, int
    :param kernel_size: triple or int
    :param stride: conv stride
    :param pad: pad
    :return: nn.Sequential
    """
    return nn.Sequential(nn.Conv2d(in_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=pad),
                         nn.BatchNorm2d(out_chanel), nn.ReLU())


def load_lightcnn(location, cuda=False):
    """
    load light cnn to memory
    :param location: lightcnn path
    :param cuda: gpu speed up
    :return: 29-layer light cnn model
    """
    model = LightCNN_29Layers_v2(num_classes=80013)
    lock_net(model)
    # net_parameters(model, "light cnn")
    model.eval()
    if cuda:
        checkpoint = torch.load(location)
        model = torch.nn.DataParallel(model).cuda()
    else:
        checkpoint = torch.load(location, map_location="cpu")
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
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
        _img = scipy.misc.imresize(arr=_img, size=(128, 128), interp='bilinear')
        _img = transform(_img)
        _img = _img.view(1, 1, 128, 128)
        input_var = torch.autograd.Variable(_img)
        _, features = lightcnn_inst(input_var)
        feature_tensor[i] = features
    return feature_tensor


def batch_feature256(img, lightcnn_inst):
    """
       使用light cnn提取256维特征参数
       :param lightcnn_inst: lightcnn model instance
       :param img: tensor 输入图片 shape:(batch, 1, 512, 512)
       :return: 256维特征参数 tensor [batch, 256]
       """
    transform = transforms.Compose([transforms.ToTensor()])
    img = F.max_pool2d(img, (4, 4))
    # input_var = torch.autograd.Variable(img)
    _, features = lightcnn_inst(img)
    log.debug("features shape:{0} {1} {2}".format(features.size(), features.requires_grad, img.requires_grad))
    return features


def get_cos_distance(x1, x2):
    """
    calculate cos distance between two sets
    tensorflow: https://blog.csdn.net/liuchonge/article/details/70049413
    :param x1: [batch, 256] dimensions vector
    :param x2: [batch, 256 dimensions vector
    """
    batch = x1.size(0)
    result = torch.Tensor(batch)
    for i in range(batch):
        """
        # implement with tensorflow
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1)))
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2)))
        x1_x2 = tf.reduce_sum(tf.multiply(x1, x2))
        result[i] = x1_x2 / (x1_norm * x2_norm)
        """
        x1_norm = torch.sqrt(torch.sum(x1.mul(x1)))
        x2_norm = torch.sqrt(torch.sum(x2.mul(x2)))
        x1_x2 = torch.sum(x1.mul(x2))
        result[i] = x1_x2 / (x1_norm * x2_norm)
    return result


def discriminative_loss(img1, img2, lightcnn_inst):
    """
    论文里的判别损失
    Discriminative Loss
    :param lightcnn_inst: lightcnn model instance
    :param img1: generated by engine, type: list of Tensor
    :param img2: generated by imitator, type: list of Tensor
    :return tensor scalar
    """
    # with torch.no_grad():
    batch_size = img1.size(0)
    x1 = batch_feature256(img1, lightcnn_inst)
    x2 = batch_feature256(img2, lightcnn_inst)
    cos_t = get_cos_distance(x1, x2)
    return torch.mean(torch.ones(batch_size) - cos_t)


def evalute_face(img):
    """
    face segmentation model
    :return: face-parsing image
    """
    return out_evaluate(img)


def content_loss(img1, img2):
    """
    change resolution to 1/8, 512/8 = 64
    :param img1 str image1's path
    :param img2 str image2's path
    :return: tensor
    """
    image1 = scipy.misc.imresize(arr=img1, size=(64, 64))
    image2 = scipy.misc.imresize(arr=img2, size=(64, 64))
    entroy = nn.CrossEntropyLoss()
    cross = entroy(image1, image2)
    return cross


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
    scipy.misc.imsave(filepath, arr=to_save)


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
