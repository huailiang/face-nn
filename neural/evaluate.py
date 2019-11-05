#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/31

import utils
import ops
import util.logit as log
from imitator import Imitator
from faceparsing.evaluate import *

"""
面部相似性度量
    Ls = alpha * L1 + L2
"""


class Evaluate:
    def __init__(self, name, args, cuda=False):
        """
        Evaluate
        :param name: evaluate name
        :param args: argparse options
        :param cuda: gpu speed up
        """
        self.name = name
        self.args = args
        location = self.args.lightcnn
        self.lightcnn_inst = utils.load_lightcnn(location, cuda)
        self.cuda = cuda
        self.parsing = self.args.parsing_checkpoint
        self.max_itr = 8
        self.prev_path = "./output/eval"
        self.clean()
        self.imitator = Imitator("neural imitator", args, clean=False)
        if cuda:
            self.imitator.cuda()
        self.imitator.eval()
        self.imitator.load_checkpoint("model_imitator_40000.pth", False, cuda=cuda)

    def discrim_l1(self, y, y_):
        """
        content loss evaluated by lightcnn
        :param y: input photo, numpy array [H, W, C]
        :param y_: generated image, torch tensor [B, C, W, H]
        :return: l1 loss
        """
        y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        y = np.swapaxes(y, 0, 2)
        y = np.mean(y, axis=0)[np.newaxis, np.newaxis, :, :]
        y = torch.from_numpy(y)
        y_ = y_.cpu().detach().numpy()
        y_ = y_.reshape(512, 512, 3)
        y_ = cv2.resize(y_, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        y_ = np.mean(y_, axis=2)[np.newaxis, np.newaxis, :, :]
        y_ = torch.from_numpy(y_)
        return utils.discriminative_loss(y, y_, self.lightcnn_inst)

    def discrim_l2(self, y, y_, export, step):
        """
        facial semantic feature loss
        evaluate loss use l1 at pixel space
        :param y: input photo, numpy array  [H, W, C]
        :param y_: generated image, tensor  [B, C, W, H]
        :param export: export for preview in train
        :param step: train step
        :return: l1 loss in pixel space
        """
        img1 = parse_evaluate(y.astype(np.uint8), cp=self.parsing, cuda=self.cuda)
        y_ = y_.cpu().detach().numpy()
        y_ = np.squeeze(y_, axis=0)
        y_ = np.swapaxes(y_, 0, 2) * 255
        img2 = parse_evaluate(y_.astype(np.uint8), cp=self.parsing, cuda=self.cuda)
        edge_img1 = utils.img_edge(img1).astype(np.float32)
        edge_img2 = utils.img_edge(img2).astype(np.float32)
        if export:
            path = os.path.join(self.prev_path, "l2_{0}.jpg".format(step))
            edge1_v3 = 255. - ops.fill_grey(edge_img1)
            edge2_v3 = 255. - ops.fill_grey(edge_img2)
            image = ops.merge_4image(y, y_, edge1_v3, edge2_v3, transpose=False)
            cv2.imwrite(path, image)
        return np.mean(np.abs(edge_img1 - edge_img2))

    def evaluate_ls(self, y, y_, export, step):
        """
        评估损失Ls
        :param y: input photo, numpy array
        :param y_:  generated image, tensor [b,c,w,h]
        :param step: train step
        :param export: export for preview in train
        :return: ls
        """
        l1 = self.discrim_l1(y, y_)
        l2 = self.discrim_l2(y, y_, export, step)
        alpha = 200  # weight balance
        # log.info("l1:{0:.5f} l2:{1:.5f}".format(l1, l2))
        return alpha * l1 + l2

    def itr_train(self, y):
        """
        iterator train
        :param y: numpy array
        """
        param_cnt = self.args.params_cnt
        np_params = 0.5 * np.ones((1, param_cnt), dtype=np.float32)
        x_ = torch.from_numpy(np_params)
        if self.cuda:
            x_ = x_.cuda()
        learning_rate = 0.01
        ix = 0  # batch index
        steps = 1  # param_cnt
        loss_ = 0
        # progress = tqdm(range(0, steps), initial=0, total=steps)  # type: # tqdm
        with torch.no_grad():
            for j in range(steps):
                # for j in progress:
                for i in range(self.max_itr):
                    y_ = self.imitator(x_)
                    export = i == self.max_itr - 1
                    loss = self.evaluate_ls(y, y_, export, j)
                    delta = loss - loss_
                    loss_ = loss
                    x_[ix][j] = self.update_x(x_[ix][j], learning_rate * delta)
                    description = "loss: {0:.5f} delta:{1:.5} x:{2:.5}".format(loss, delta, x_[ix][j])
                    # progress.set_description(description)
                    log.info(description)
                loss_ = loss + learning_rate
        return x_

    def for_train(self, y):
        param_cnt = self.args.params_cnt
        np_params = 0.5 * np.ones((1, param_cnt), dtype=np.float32)
        param_cnt = self.args.params_cnt
        x_ = torch.from_numpy(np_params)
        if self.cuda:
            x_ = x_.cuda()
        m_progress = tqdm(range(0, param_cnt), initial=0, total=param_cnt)
        with torch.no_grad():
            loop = 4
            for _ in range(loop):
                m_progress.pos = 0
                for p in m_progress:
                    mini_loss = 1e5
                    tmp_x = 0
                    for i in range(10):
                        x_[0][p] = i * 0.1 + 0.025 * _
                        y_ = self.imitator(x_)
                        loss = self.evaluate_ls(y, y_, False, i).item()
                        if loss < mini_loss:
                            tmp_x = x_[0][p].item()
                            mini_loss = loss
                    x_[0][p] = tmp_x
                    m_progress.set_description("{0}/{1} {2:.4f}".format(_, loop, tmp_x))
        return x_

    @staticmethod
    def update_x(x, delta_loss):
        """
        更新梯度
        :param x: input scalar
        :param delta_loss: gradient loss, delta * lr
        :return: updated value, scalar
        """
        delta = delta_loss.item()
        # 避免更新幅度过大或者过小
        dir = -1 if delta < 0 else 1
        if abs(delta) < 1e-3:
            delta = dir * 1e-3
        elif abs(delta) > 0.4:
            delta = dir * 0.4

        delta_x = -delta
        new_x = x + delta_x
        if new_x < 0:
            new_x = 0
        elif new_x > 1:
            new_x = 1
        return new_x

    def output(self, x, refer):
        """
        capture for result
        :param refer: reference picture
        :param x: generated image with grad, torch tensor [b,params]
        """
        y_ = self.imitator.forward(x)
        y_ = y_.cpu().detach().numpy()
        y_ = np.squeeze(y_, axis=0)

        y_ = np.swapaxes(y_, 0, 2) * 255.
        y_ = y_.astype(np.uint8)
        image_ = ops.merge_image(refer, y_, transpose=False)
        path = os.path.join(self.prev_path, "eval.jpg")
        cv2.imwrite(path, image_)

    def clean(self):
        """
        clean for new iter
        """
        ops.clear_files(self.prev_path)


if __name__ == '__main__':
    import logging
    from parse import parser

    log.info("evaluation mode start")
    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/neural_log.txt")
    evl = Evaluate("test", args, cuda=True)
    img = cv2.imread("../export/testset_female/db_0000_4.jpg").astype(np.float32)
    x_ = evl.for_train(img)
    # x_ = evl.itr_train(img)
    evl.output(x_, img)
