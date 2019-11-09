#!/usr/bin/python
# -*- encoding: utf-8 -*-


from faceparsing.model import BiSeNet
import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import os.path as osp
import numpy as np
import torchvision.transforms as transforms


def vis_parsing_maps(im, parsing_anno, stride):
    """
    # 显示所有部位
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    # 不显示头发 脖子等
    part_colors = [[255, 255, 255], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [255, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [255, 255, 255], [255, 255, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    """
    # 只显示鼻子 眼睛 眉毛 嘴巴
    part_colors = [[255, 255, 255], [255, 255, 255], [255, 170, 0], [255, 170, 0], [255, 0, 170], [0, 255, 0],
                   [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    """
    part_colors = [[255, 255, 255], [脸], [左眉], [右眉], [左眼], [右眼],
                   [255, 255, 255],
                   [左耳], [右耳], [255, 255, 255], [鼻子], [中唇], [上唇], 
                   [下唇],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    """

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.01, vis_parsing_anno_color, 0.99, 0)
    return vis_im


def _build_net(cp, cuda=False):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if cuda:
        net.cuda()
        net.load_state_dict(torch.load(cp))
    else:
        net.load_state_dict(torch.load(cp, map_location="cpu"))
    net.eval()
    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),  # [H, W, C]->[C, H, W]
            # 这里是用来增强数据
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return net, to_tensor


_net_ = None
_to_tensor_ = None


def build_net(cp, cuda=False):
    """
    global _net_, _to_tensor_ for performance
    :param cuda: use gpu to speedup
    :param cp: args.parsing_checkpoint, str
    """
    global _net_
    global _to_tensor_
    if _net_ is None or _to_tensor_ is None:
        _net_, _to_tensor_ = _build_net(cp, cuda)


def faceparsing_ndarray(input, cp, cuda=False):
    """
    evaluate with numpy array
    :param input: numpy array, 注意一定要是np.uint8, 而不是np.float32 [H, W, C]
    :param cp: args.parsing_checkpoint, str
    :param cuda: use gpu to speedup
    """
    build_net(cp, cuda)
    tensor = _to_tensor_(input)
    tensor = torch.unsqueeze(tensor, 0)
    if cuda:
        tensor = tensor.cuda()
    out = _net_(tensor)[0]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    return vis_parsing_maps(input, parsing, stride=1)


def faceparsing_tensor(tensor, cp, cuda=False):
    """
    evaluate with torch tensor
    :param tensor: torch tensor [B, H, W, C]
    :param cp: args.parsing_checkpoint, str
    :param cuda: use gpu to speedup
    :return  tensor: [H, W]
    """
    build_net(cp, cuda)
    out = _net_(tensor)[0]
    out = out.squeeze()
    out = torch.argmax(out, dim=0)
    return out


def _img_edge(input):
    """
    提取原始图像的边缘
    :param input: input image
    :return: edge image
    """
    gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    x_grad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    y_grad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    return cv2.Canny(x_grad, y_grad, 20, 40)


if __name__ == '__main__':
    src_path = "../../export/faceparsing/src"
    root, name = os.path.split(src_path)
    dst_pth = os.path.join(root, 'dst')
    edge_pth = os.path.join(root, 'edge')
    if not os.path.exists(dst_pth):
        os.mkdir(dst_pth)
    if not os.path.exists(edge_pth):
        os.mkdir(edge_pth)
    with torch.no_grad():
        list_image = os.listdir(src_path)
        total = len(list_image)
        progress = tqdm(range(0, total), initial=0, total=total)
        for step in progress:
            img = cv2.imread(osp.join(src_path, list_image[step]))
            image = cv2.resize(img, (512, 512), cv2.INTER_LINEAR)
            cuda = torch.cuda.is_available()
            img = faceparsing_ndarray(image, '../dat/79999_iter.pth', cuda)
            save_path = osp.join(dst_pth, list_image[step])
            cv2.imwrite(save_path, img)
            save_path = osp.join(edge_pth, list_image[step])
            edge = 255 - _img_edge(img)
            cv2.imwrite(save_path, edge)
