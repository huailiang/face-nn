#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-04-27

from __future__ import division
from ops import *


def feature_extractor(x, reuse=True, name="extractor"):
    """
    此网络用来生成engine face的params
    :param x: reference image (batch, 512, 512, 3)
    :param reuse:
    :param name: scope
    :return: engine params
    """
    y1 = conv2d(x, 3, 2, name="ex_e1_c")  # (1, 256, 256, 3)
    y1 = tf.nn.relu(instance_norm(y1, name="ex_e1_bn"))  # (1, 256, 256, 3)
    y1 = tf.nn.max_pool(y1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="VALID", name="ex_pool")  # (1, 254,
    # 254, 3)
    y2 = conv2d(y1, 8, 2, name="ex_e2_c")
    y2 = tf.nn.relu(instance_norm(y2, name="ex_e2_bn"))  # (1, 127, 127, 8)
    y3 = conv2d(y2, 16, 2, name="ex_e3_c")
    y3 = tf.nn.relu(instance_norm(y3, name="ex_e3_bn"))  # (1, 64, 64, 16)
    y4 = conv2d(y3, 32, 1, 4, name="ex_e4_c")
    y4 = tf.nn.relu(instance_norm(y4, name="ex_e4_bn"))  # (1, 16, 16, 32)
    y5 = conv2d(y4, 64, 1, 4, name="ex_e5_c")
    y5 = tf.nn.relu(instance_norm(y5, name="ex_e5_bn"))  # (1, 4, 4, 64)
    y6 = conv2d(y5, 95, 1, 8, name="ex_e6_c")
    y6 = tf.nn.relu(instance_norm(y6, name="ex_e6_bn"))  # (1, 1, 1, 95)
    return y6


def imitator(x, reuse=True, name="imitator"):
    """
    这里建立八层imitator网络， 用来拟合引擎生成mesh的过程
    由于引擎中捏脸使用的参数跟论文《逆水寒》引擎中使用的参数不相同，所以每一个layer的depth不一样
    :param x: 捏脸参数
    :param reuse:
    :param name: scope
    """
    with tf.variable_scope(name):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # else:
        #     print("encoder reuse", tf.get_variable_scope().reuse)
        #     assert tf.get_variable_scope().reuse is False
        y1 = tf.pad(x, [[0, 0], [1, 2], [1, 2], [0, 0]], "CONSTANT")  # (1, 4, 4, 95)
        y1 = tf.nn.relu(instance_norm(conv2d(y1, 64, 4, 1, name='i_e1_c'), name='i_e1_bn'))  # (1,4,4,64)
        y2 = tf.pad(y1, [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")  # (1, 8, 8, 64)
        y2 = tf.nn.relu(instance_norm(conv2d(y2, 32, 4, 1, name='i_e2_c'), name='i_e2_bn'))  # (1, 8, 8, 32)
        y3 = tf.pad(y2, [[0, 0], [4, 4], [4, 4], [0, 0]], "REFLECT")
        y3 = tf.nn.relu(instance_norm(conv2d(y3, 32, 4, 1, name='i_e3_c'), name='i_e3_bn'))  # (1, 16, 16, 32)
        y4 = tf.pad(y3, [[0, 0], [8, 8], [8, 8], [0, 0]], "REFLECT")
        y4 = tf.nn.relu(instance_norm(conv2d(y4, 16, 4, 1, name='i_e4_c'), name='i_e4_bn'))  # (1, 32, 32, 16)
        y5 = tf.pad(y4, [[0, 0], [16, 16], [16, 16], [0, 0]], "REFLECT")
        y5 = tf.nn.relu(instance_norm(conv2d(y5, 16, 4, 1, name='i_e5_c'), name='i_e5_bn'))  # (1, 64, 64, 16)
        y6 = tf.pad(y5, [[0, 0], [32, 32], [32, 32], [0, 0]], "REFLECT")
        y6 = tf.nn.relu(instance_norm(conv2d(y6, 16, 4, 1, name='i_e6_c'), name='i_e6_bn'))  # (1, 128, 128, 16)
        y7 = tf.pad(y6, [[0, 0], [64, 64], [64, 64], [0, 0]], "REFLECT")
        y7 = tf.nn.relu(instance_norm(conv2d(y7, 8, 4, 1, name='i_e7_c'), name='i_e7_bn'))  # (1, 256, 256, 16)
        y8 = tf.pad(y7, [[0, 0], [128, 128], [128, 128], [0, 0]], "REFLECT")
        y8 = tf.nn.relu(instance_norm(conv2d(y8, 3, 4, 1, name='i_e8_c'), name='i_e8_bn'))  # (1, 512, 512, 3)
        return y8


def encoder(image, options, reuse=True, name="encoder"):
    """
    Args:
        image: input tensor, must have
        options: options defining number of kernels in conv layers
        reuse: to create new encoder or use existing
        name: name of the encoder

    Returns: Encoded image.
    """

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            print("encoder reuse", tf.get_variable_scope().reuse)
            assert tf.get_variable_scope().reuse is False
        image = instance_norm(input=image, is_training=options.is_training, name='g_e0_bn')
        c0 = tf.pad(image, [[0, 0], [15, 15], [15, 15], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(input=conv2d(c0, options.gf_dim, 3, 1, padding='VALID', name='g_e1_c'),
                                      is_training=options.is_training, name='g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(input=conv2d(c1, options.gf_dim, 3, 2, padding='VALID', name='g_e2_c'),
                                      is_training=options.is_training, name='g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim * 2, 3, 2, padding='VALID', name='g_e3_c'),
                                      is_training=options.is_training, name='g_e3_bn'))
        c4 = tf.nn.relu(instance_norm(conv2d(c3, options.gf_dim * 4, 3, 2, padding='VALID', name='g_e4_c'),
                                      is_training=options.is_training, name='g_e4_bn'))
        c5 = tf.nn.relu(instance_norm(conv2d(c4, options.gf_dim * 8, 3, 2, padding='VALID', name='g_e5_c'),
                                      is_training=options.is_training, name='g_e5_bn'))
        return c5


def decoder(features, options, reuse=True, name="decoder"):
    """
    Args:
        features: input tensor, must have
        options: options defining number of kernels in conv layers
        reuse: to create new decoder or use existing
        name: name of the encoder

    Returns: Decoded image.
    """

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c1'), name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name + '_c2'), name + '_bn2')
            return y + x

        # Now stack 9 residual blocks
        num_kernels = features.get_shape().as_list()[-1]
        r1 = residule_block(features, num_kernels, name='g_r1')
        r2 = residule_block(r1, num_kernels, name='g_r2')
        r3 = residule_block(r2, num_kernels, name='g_r3')
        r4 = residule_block(r3, num_kernels, name='g_r4')
        r5 = residule_block(r4, num_kernels, name='g_r5')
        r6 = residule_block(r5, num_kernels, name='g_r6')
        r7 = residule_block(r6, num_kernels, name='g_r7')
        r8 = residule_block(r7, num_kernels, name='g_r8')
        r9 = residule_block(r8, num_kernels, name='g_r9')

        # Decode image.
        d1 = tf.nn.relu(instance_norm(input=deconv2d(r1, options.gf_dim * 8, 3, 2, name='g_d1_dc'), name='g_d1_bn',
                                      is_training=options.is_training))
        d2 = tf.nn.relu(instance_norm(input=deconv2d(d1, options.gf_dim * 4, 3, 2, name='g_d2_dc'), name='g_d2_bn',
                                      is_training=options.is_training))
        d3 = tf.nn.relu(instance_norm(input=deconv2d(d2, options.gf_dim * 2, 3, 2, name='g_d3_dc'), name='g_d3_bn',
                                      is_training=options.is_training))
        d4 = tf.nn.relu(instance_norm(input=deconv2d(d3, options.gf_dim, 3, 2, name='g_d4_dc'), name='g_d4_bn',
                                      is_training=options.is_training))

        d4 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.sigmoid(conv2d(d4, 3, 7, 1, padding='VALID', name='g_pred_c')) * 2. - 1.
        return pred


def discriminator(image, options, reuse=True, name="discriminator"):
    """
    Discriminator agent, that provides us with information about image plausibility at different scales.
    Args:
        image: input tensor
        options: options defining number of kernels in conv layers
        reuse: to create new discriminator or use existing
        name: name of the discriminator

    Returns:
        Image estimates at different scales.
    """
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(instance_norm(conv2d(image, options.df_dim * 2, ks=5, name='d_h0_conv'), name='d_bn0'))
        h0_pred = conv2d(h0, 1, ks=5, s=1, name='d_h0_pred', activation_fn=None)

        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim * 2, ks=5, name='d_h1_conv'), name='d_bn1'))
        h1_pred = conv2d(h1, 1, ks=10, s=1, name='d_h1_pred', activation_fn=None)

        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim * 4, ks=5, name='d_h2_conv'), name='d_bn2'))

        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim * 8, ks=5, name='d_h3_conv'), name='d_bn3'))
        h3_pred = conv2d(h3, 1, ks=10, s=1, name='d_h3_pred', activation_fn=None)

        h4 = lrelu(instance_norm(conv2d(h3, options.df_dim * 8, ks=5, name='d_h4_conv'), name='d_bn4'))

        h5 = lrelu(instance_norm(conv2d(h4, options.df_dim * 16, ks=5, name='d_h5_conv'), name='d_bn5'))
        h5_pred = conv2d(h5, 1, ks=6, s=1, name='d_h5_pred', activation_fn=None)

        h6 = lrelu(instance_norm(conv2d(h5, options.df_dim * 16, ks=5, name='d_h6_conv'), name='d_bn6'))
        h6_pred = conv2d(h6, 1, ks=3, s=1, name='d_h6_pred', activation_fn=None)

        return {"scale_0": h0_pred, "scale_1": h1_pred, "scale_3": h3_pred, "scale_5": h5_pred, "scale_6": h6_pred}


# ====== Define different types of losses applied to discriminator's output. ====== #

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mse_criterion(in_, target):
    return tf.reduce_mean((in_ - target) ** 2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def reduce_spatial_dim(input_tensor):
    """
    Since labels and discriminator outputs are of different shapes (and even ranks)
    we should write a routine to deal with that.
    Args:
        input: tensor of shape [batch_size, spatial_resol_1, spatial_resol_2, depth]
    Returns:
        tensor of shape [batch_size, depth]
    """
    input_tensor = tf.reduce_mean(input_tensor=input_tensor, axis=1)
    input_tensor = tf.reduce_mean(input_tensor=input_tensor, axis=1)
    return input_tensor


def add_spatial_dim(input_tensor, dims_list, resol_list):
    """
        Appends dimensions mentioned in dims_list resol_list times. S
        Args:
            input: tensor of shape [batch_size, depth0]
            dims_list: list of integers with position of new  dimensions to append.
            resol_list: list of integers with corresponding new dimensionalities for each dimension.
        Returns:
            tensor of new shape
        """
    for dim, res in zip(dims_list, resol_list):
        input_tensor = tf.expand_dims(input=input_tensor, axis=dim)
        input_tensor = tf.concat(values=[input_tensor] * res, axis=dim)
    return input_tensor


def repeat_scalar(input_tensor, shape):
    """
    Repeat scalar values.
    :param input_tensor: tensor of shape [batch_size, 1]
    :param shape: new_shape of the element of the tensor
    :return: tensor of the shape [batch_size, *shape] with elements repeated.
    """
    with tf.control_dependencies([tf.assert_equal(tf.shape(input_tensor)[1], 1)]):
        batch_size = tf.shape(input_tensor)[0]
    input_tensor = tf.tile(input_tensor, tf.stack(values=[1, tf.reduce_prod(shape)], axis=0))
    input_tensor = tf.reshape(input_tensor, tf.concat(values=[[batch_size], shape, [1]], axis=0))
    return input_tensor


def transformer_block(input_tensor, kernel_size=10):
    """
    This is a simplified version of transformer block described in our paper
    https://arxiv.org/abs/1807.10201.
    Args:
        input_tensor: Image(or tensor of rank 4) we want to transform.
        kernel_size: Size of kernel we apply to the input_tensor.
    Returns:
        Transformed tensor
    """
    return slim.avg_pool2d(inputs=input_tensor, kernel_size=kernel_size, stride=1, padding='SAME')
