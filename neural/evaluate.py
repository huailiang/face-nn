#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/31

import utils
import ops
import util.logit as log
from imitator import Imitator
from faceparsing.evaluate import *
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
        utils.lock_net(self.lightcnn_inst)
        self.cuda = cuda
        self.parsing = self.args.parsing_checkpoint
        self.max_itr = 256
        self.losses = []
        self.prev_path = "./output/eval"
        self.clean()
        self.imitator = Imitator("neural imitator", args, clean=False)
        if cuda:
            self.imitator.cuda()
        self.imitator.eval()
        utils.lock_net(self.imitator)
        self.imitator.load_checkpoint("model_imitator_40000.pth", False, cuda=cuda)

    def discrim_l1(self, y, y_):
        """
        content loss evaluated by lightcnn
        :param y: input photo, numpy array [H, W, C]
        :param y_: generated image, torch tensor [B, C, W, H]
        :return: l1 loss
        """
        y = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        y = np.swapaxes(y, 0, 2).astype(np.float32)
        y = np.mean(y, axis=0)[np.newaxis, np.newaxis, :, :]
        y = torch.from_numpy(y)
        if self.cuda:
            y = y.cuda()
        y_ = F.max_pool2d(y_, kernel_size=(4, 4), stride=4)  # 512->128
        y_ = torch.mean(y_, dim=1).view(1, 1, 128, 128)  # gray
        return utils.discriminative_loss(y, y_, self.lightcnn_inst)

    def discrim_l2(self, y, y_, step):
        """
        facial semantic feature loss
        evaluate loss use l1 at pixel space
        :param y: input photo, numpy array  [H, W, C]
        :param y_: generated image, tensor  [B, C, W, H]
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

        if step % 50 == 0:
            path = os.path.join(self.prev_path, "l2_{0}.jpg".format(step))
            edge1_v3 = 255. - ops.fill_grey(edge_img1)
            edge2_v3 = 255. - ops.fill_grey(edge_img2)
            image = ops.merge_4image(y, y_, edge1_v3, edge2_v3, transpose=False)
            cv2.imwrite(path, image)
        return np.mean(np.abs(edge_img1 - edge_img2))

    def evaluate_ls(self, y, y_, step):
        """
        评估损失Ls
        :param y: input photo, numpy array
        :param y_:  generated image, tensor [b,c,w,h]
        :param step: train step
        :return: ls
        """
        l1 = self.discrim_l1(y, y_)
        l2 = self.discrim_l2(y, y_, step)
        alpha = 2  # weight balance
        ls = alpha * l1 + l2
        log.info("{0:3} l1:{1:.4f} l2:{2:.4f} ls:{3:.4f}".format(step + 1, l1, l2, ls))
        self.losses.append((l1.item(), l2.item(), ls.item()))
        return ls

    def itr_train(self, y):
        """
        iterator train
        :param y: numpy array, image [H, W, C]
        """
        param_cnt = self.args.params_cnt
        np_params = 0.5 * np.ones((1, param_cnt), dtype=np.float32)
        t_params = torch.from_numpy(np_params)
        if self.cuda:
            t_params = t_params.cuda()
        t_params.requires_grad = True
        self.losses.clear()
        optimize = optim.Adam([t_params], lr=0.01)
        for i in range(self.max_itr):
            y_ = self.imitator.forward(t_params)
            loss = self.evaluate_ls(y, y_, i)
            loss.backward()
            optimize.step()
            optimize.zero_grad()
        self.plot()
        return t_params

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

    def plot(self):
        plt.style.use('seaborn-whitegrid')
        x = range(self.max_itr)
        y1 = []
        y2 = []
        y3 = []
        for it in self.losses:
            y1.append(it[0])
            y2.append(it[1])
            y3.append(it[2])
        plt.plot(x, y1, color='r')
        plt.plot(x, y2, color='g')
        plt.plot(x, y3, color='b')
        path = os.path.join(self.prev_path, "curve.png")
        plt.savefig(path)


if __name__ == '__main__':
    import logging
    from parse import parser

    log.info("evaluation mode start")
    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/neural_log.txt")
    evl = Evaluate("test", args, cuda=True)
    img = cv2.imread("../export/testset_female/db_0000_4.jpg").astype(np.float32)
    x_ = evl.itr_train(img)
    evl.output(x_, img)
