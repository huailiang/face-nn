#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/31

import utils
import ops
import util.logit as log
from imitator import Imitator
from export import write_layer
from faceparsing.evaluate import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
面部相似性度量
 Ls = alpha * L1 + L2
"""


class Evaluate:
    def __init__(self, arguments, cuda=False):
        """
        Evaluate
        :param arguments: argparse options
        :param cuda: gpu speed up
        """
        self.args = arguments
        location = self.args.lightcnn
        self.lightcnn_inst = utils.load_lightcnn(location, cuda)
        self.cuda = cuda
        self.parsing = self.args.parsing_checkpoint
        self.max_itr = arguments.total_eval_steps
        self.learning_rate = arguments.eval_learning_rate
        self.losses = []
        self.prev_path = "./output/eval"
        self.model_path = "../unity/models"
        self.clean()
        self.imitator = Imitator("neural imitator", arguments, clean=False)
        self.l2_c = (torch.ones((512, 512)), torch.ones((512, 512)))
        if cuda:
            self.imitator.cuda()
        self.imitator.eval()
        self.imitator.load_checkpoint(args.imitator_model, False, cuda=cuda)

    def _init_l1_l2(self, y):
        """
        init reference photo l1 & l2
        :param y: input photo, numpy array [H, W, C]
        """
        y_ = cv2.resize(y, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
        y_ = np.swapaxes(y_, 0, 2).astype(np.float32)
        y_ = np.mean(y_, axis=0)[np.newaxis, np.newaxis, :, :]
        y_ = torch.from_numpy(y_)
        if self.cuda:
            y_ = y_.cuda()
        self.l1_y = y_
        y = y[np.newaxis, :, :, ]
        y = np.swapaxes(y, 1, 2)
        y = np.swapaxes(y, 1, 3)
        y = torch.from_numpy(y)
        if self.cuda:
            y = y.cuda()
        self.l2_y = y / 255.

    def discrim_l1(self, y_):
        """
        content loss evaluated by lightcnn
        :param y_: generated image, torch tensor [B, C, W, H]
        :return: l1 loss
        """
        y_ = F.max_pool2d(y_, kernel_size=(4, 4), stride=4)  # 512->128
        y_ = torch.mean(y_, dim=1).view(1, 1, 128, 128)  # gray
        return utils.discriminative_loss(self.l1_y, y_, self.lightcnn_inst)

    def discrim_l2(self, y_):
        """
        facial semantic feature loss
        evaluate loss use l1 at pixel space
        :param y_: generated image, tensor  [B, C, W, H]
        :return: l1 loss in pixel space
        """
        # [eyebrow，eye，nose，teeth，up lip，lower lip]
        w_r = [1.1, 1., 1., 0.7, 1., 1.]
        w_g = [1.1, 1., 1., 0.7, 1., 1.]
        part1, _ = faceparsing_tensor(self.l2_y, self.parsing, w_r, cuda=self.cuda)
        y_ = y_.transpose(2, 3)
        part2, _ = faceparsing_tensor(y_, self.parsing, w_g, cuda=self.cuda)
        self.l2_c = (part1 * 10, part2 * 10)
        return F.l1_loss(part1, part2)

    def evaluate_ls(self, y_):
        """
        评估损失Ls
        由于l1表示的是余弦距离的损失， 其范围又在0-1之间 所以这里使用1-l1
        (余弦距离越大 表示越接近)
        :param y_:  generated image, tensor [b,c,w,h]
        :return: ls, description
        """
        l1 = self.discrim_l1(y_)
        l2 = self.discrim_l2(y_)
        alpha = self.args.eval_alpha
        ls = alpha * (1 - l1) + l2
        info = "l1:{0:.3f} l2:{1:.3f} ls:{2:.3f}".format(l1, l2, ls)
        self.losses.append((l1.item(), l2.item() / 3, ls.item()))
        return ls, info

    def argmax_params(self, params, start, count):
        """
        One-hot编码 argmax 处理
        :param params: 处理params
        :param start: One-hot 偏移起始地址
        :param count: One-hot 编码长度
        """
        dims = params.size()[0]
        for dim in range(dims):
            tmp = params[dim, start]
            mx = start
            for idx in range(start + 1, start + count):
                if params[dim, idx] > tmp:
                    mx = idx
                    tmp = params[dim, idx]
            for idx in range(start, start + count):
                params[dim, idx] = 1. if idx == mx else 0

    def itr_train(self, y):
        """
        iterator train
        :param y: numpy array, image [H, W, C]
        """
        param_cnt = self.args.params_cnt
        t_params = 0.5 * torch.ones((1, param_cnt), dtype=torch.float32)
        if self.cuda:
            t_params = t_params.cuda()
        t_params.requires_grad = True
        self.losses.clear()
        lr = self.learning_rate
        self._init_l1_l2(y)
        m_progress = tqdm(range(1, self.max_itr + 1))
        for i in m_progress:
            y_ = self.imitator(t_params)
            loss, info = self.evaluate_ls(y_)
            loss.backward()
            if i == 1:
                self.output(t_params, y, 0)
            t_params.data = t_params.data - lr * t_params.grad.data
            t_params.data = t_params.data.clamp(0., 1.)
            self.argmax_params(t_params.data, 96, 3)
            t_params.grad.zero_()
            m_progress.set_description(info)
            if i % self.args.eval_prev_freq == 0:
                x = i / float(self.max_itr)
                lr = self.learning_rate * (1 - x) + 1e-2
                self.output(t_params, y, i)
                self.plot()
        self.plot()
        log.info("steps:{0} params:{1}".format(self.max_itr, t_params.data))
        return t_params

    def output(self, x, refer, step):
        """
        capture for result
        :param x: generated image with grad, torch tensor [b,params]
        :param refer: reference picture
        :param step: train step
        """
        self.write(x)
        y_ = self.imitator(x)
        y_ = y_.cpu().detach().numpy()
        y_ = np.squeeze(y_, axis=0)
        y_ = np.swapaxes(y_, 0, 2) * 255
        y_ = y_.astype(np.uint8)
        im1 = self.l2_c[0]
        im2 = self.l2_c[1]
        np_im1 = im1.cpu().detach().numpy()
        np_im2 = im2.cpu().detach().numpy()
        f_im1 = ops.fill_gray(np_im1)
        f_im2 = ops.fill_gray(np_im2)
        image_ = ops.merge_4image(refer, y_, f_im1, f_im2, transpose=False)
        path = os.path.join(self.prev_path, "eval_{0}.jpg".format(step))
        cv2.imwrite(path, image_)

    def write(self, params):
        """
        生成二进制文件 能够在unity里还原出来
        :param params: 捏脸参数 tensor [batch, params_cnt]
        """
        np_param = params.cpu().detach().numpy()
        np_param = np_param[0]
        list_param = np_param.tolist()
        dataset = self.args.path_to_dataset
        shape = utils.curr_roleshape(dataset)
        path = os.path.join(self.model_path, "eval.bytes")
        f = open(path, 'wb')
        write_layer(f, shape, list_param)
        f.close()

    def clean(self):
        """
        clean for new iter
        """
        ops.clear_files(self.prev_path)
        ops.clear_files(self.model_path)

    def plot(self):
        """
        plot loss
        """
        count = len(self.losses)
        if count > 0:
            plt.style.use('seaborn-whitegrid')
            x = range(count)
            y1 = []
            y2 = []
            for it in self.losses:
                y1.append(it[0])
                y2.append(it[1])
            plt.plot(x, y1, color='r', label='l1')
            plt.plot(x, y2, color='g', label='l2')
            plt.ylabel("loss")
            plt.xlabel('step')
            plt.legend()
            path = os.path.join(self.prev_path, "loss.png")
            plt.savefig(path)
            plt.close('all')


if __name__ == '__main__':
    import logging
    from parse import parser

    log.info("evaluation mode start")
    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/evaluate.txt")
    evl = Evaluate(args, cuda=torch.cuda.is_available())
    img = cv2.imread(args.eval_image).astype(np.float32)
    evl.itr_train(img)
