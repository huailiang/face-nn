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
import torch.optim as optim
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
        utils.lock_net(self.lightcnn_inst)
        self.cuda = cuda
        self.parsing = self.args.parsing_checkpoint
        self.max_itr = arguments.total_eval_steps
        self.learning_rate = arguments.eval_learning_rate
        self.losses = []
        self.prev_path = "./output/eval"
        self.clean()
        self.imitator = Imitator("neural imitator", arguments, clean=False)
        if cuda:
            self.imitator.cuda()
        self.imitator.eval()
        utils.lock_net(self.imitator)
        self.imitator.load_checkpoint(args.imitator_model, False, cuda=cuda)

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
        w_g = 1.0
        w_r = 1.0

        if step % self.args.eval_prev_freq == 0:
            path = os.path.join(self.prev_path, "l2_{0}.jpg".format(step))
            edge1_v3 = 255. - ops.fill_grey(edge_img1)
            edge2_v3 = 255. - ops.fill_grey(edge_img2)
            merge = ops.merge_4image(y, y_, edge1_v3, edge2_v3, transpose=False)
            cv2.imwrite(path, merge)
        return np.mean(np.abs(w_r * edge_img1 - w_g * edge_img2))

    def evaluate_ls(self, y, y_, step):
        """
        评估损失Ls
        :param y: input photo, numpy array
        :param y_:  generated image, tensor [b,c,w,h]
        :param step: train step
        :return: ls, description
        """
        l1 = self.discrim_l1(y, y_)
        l2 = self.discrim_l2(y, y_, step)
        alpha = self.args.eval_alpha
        ls = alpha * l1 + l2
        info = "l1:{0:.3f} l2:{1:.3f} ls:{2:.3f}".format(alpha * l1, l2, ls)
        self.losses.append((l1.item() * alpha, l2.item(), ls.item()))
        return ls, info

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
        optimizer = optim.Adam([t_params], lr=self.learning_rate)
        progress = tqdm(range(self.max_itr), initial=0, total=self.max_itr)
        for i in progress:
            y_ = self.imitator.forward(t_params)
            loss, info = self.evaluate_ls(y, y_, i)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            log.info(t_params)
            log.info(t_params.requires_grad)
            t_params = t_params.clamp(0., 1.)
            log.info("op grad:{0}".format(t_params.requires_grad))
            progress.set_description(info)
            if self.max_itr % 100 == 0:
                x = i / float(self.max_itr)
                lr = self.learning_rate * (x ** 2 - 2 * x + 1) + 1e-4
                utils.update_optimizer_lr(optimizer, lr)
        self.plot()
        log.info(t_params)
        return t_params.clamp(0., 1.)

    def output(self, x, refer):
        """
        capture for result
        :param x: generated image with grad, torch tensor [b,params]
        :param refer: reference picture
        """
        self.write(x)
        y_ = self.imitator.forward(x)
        y_ = y_.cpu().detach().numpy()
        y_ = np.squeeze(y_, axis=0)
        y_ = np.swapaxes(y_, 0, 2) * 255.
        y_ = y_.astype(np.uint8)
        image_ = ops.merge_image(refer, y_, transpose=False)
        path = os.path.join(self.prev_path, "eval.jpg")
        cv2.imwrite(path, image_)

    def write(self, params):
        """
        生成二进制文件 能够在unity里还原出来
        :param params: 捏脸参数 tensor [batch, 95]
        """
        np_param = params.cpu().detach().numpy()
        np_param = np_param[0]
        list_param = np_param.tolist()
        dataset = self.args.path_to_dataset
        shape = utils.curr_roleshape(dataset)
        path = os.path.join(self.prev_path, "eval.bytes")
        f = open(path, 'wb')
        write_layer(f, shape, list_param)
        f.close()

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
        for it in self.losses:
            y1.append(it[0])
            y2.append(it[1])
        plt.plot(x, y1, color='r', label='l1')
        plt.plot(x, y2, color='g', label='l2')
        plt.ylabel("loss")
        plt.xlabel('step')
        plt.legend()
        path = os.path.join(self.prev_path, "eval.png")
        plt.savefig(path)


if __name__ == '__main__':
    import logging
    from parse import parser

    log.info("evaluation mode start")
    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/evaluate.txt")
    evl = Evaluate(args, cuda=True)
    img = cv2.imread(args.eval_image).astype(np.float32)
    x_ = evl.itr_train(img)
    evl.output(x_, img)
