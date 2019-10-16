#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-04-29

from __future__ import division
from __future__ import print_function

import time
from glob import glob
from collections import namedtuple
from tqdm import tqdm
import multiprocessing
import util.logit as log
from prepare_dataset import FaceDataset

from module import *
from utils import *
from export import *
import prepare_dataset
import img_augm


class Face(object):
    def __init__(self, sess, args):
        self.args = args
        self.model_name = args.model_name
        self.root_dir = './models'
        self.checkpoint_dir = os.path.join(self.root_dir, self.model_name, 'checkpoint')
        self.logs_dir = os.path.join(self.root_dir, self.model_name, 'logs')
        self.batch_size = args.batch_size
        self.param_cnt = args.params_cnt
        self.learning_rate = args.learning_rate
        self.dataset = FaceDataset(args)
        self.sess = sess
        self.initial_step = 0
        self.total_steps = args.db_item_cnt
        self.sess.run(tf.global_variables_initializer())
        self.lightcnn_path = args.lightcnn
        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=2)

    def _build_model(self):
        with tf.name_scope('placeholder'):
            self.input_x = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, None, None, self.param_cnt],
                                          name="params_x")
            self.refer_img = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 512, 512, 3],
                                            name="reinference_img")
            self.lightcnn_checkpoint = torch.load(self.lightcnn_path, map_location="cpu")
            self.imitator = imitator(self.input_x)
            self.extractor = feature_extractor(self.imitator)

            # ================== Define optimization steps. =============== #
            t_vars = tf.trainable_variables()
            imitator_vars = [var for var in t_vars if 'imitator' in var.name]
            loss1 = discriminative_loss(self.refer_img, self.imitator, self.lightcnn_checkpoint)
            self.i_optim_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=loss1, var_list=imitator_vars)
            extractor_vars = [var for var in t_vars if 'extractor' in var.name]
            loss2 = content_loss(imitator(self.extractor), self.refer_img)
            self.e_optim_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=loss2, var_list=extractor_vars)

            summary_i = tf.summary.scalar("face/loss1", loss1)
            summary_e = tf.summary.scalar("face/loss2", loss2)
            self.summary_feature_loss = tf.summary.merge([summary_i + summary_e])
            self.summary_merged_all = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)

    def train(self):
        for step in tqdm(range(self.initial_step, self.total_steps + 1), initial=self.initial_step,
                         total=self.total_steps):
            # print(step)
            batch_rst = self.dataset.get_batch(1)
            key = batch_rst.keys()[0]
            val = batch_rst[key][0]
            img = batch_rst[key][1]
            feed = {self.input_x: param_2_arr(val), self.refer_img: img}
            l1, l2, summary_all = self.sess.run(self.i_optim_step, self.e_optim_step, self.summary_merged_all,
                                                feed_dict=feed)
            self.writer.add_summary(summary_all, step * self.batch_size)

            if step % 500 == 0 and step > self.initial_step:
                self.save(step)

    def save(self, step):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name + '_%d.ckpt' % step),
                        global_step=step)

    def loadckpt(self, checkpoint_dir=None):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        log.info("Start inference.")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            log.info("checkpoint path: ", ckpt.model_checkpoint_path)
            self.initial_step = int(ckpt_name.split("_")[-1].split(".")[0])
            log.info("Load checkpoint %s. Initial step: %s." % (ckpt_name, self.initial_step))
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def inference(self, args, to_save_dir, img_path):
        loaded = self.loadckpt(to_save_dir)
        if loaded:
            img = scipy.misc.imread(img_path, mode='RGB')
            img = scipy.misc.imresize(img, size=512)
            param = self.sess.run(self.extractor(img))
            log.info("params:", param)
        else:
            log.error("error, loaded failed")

