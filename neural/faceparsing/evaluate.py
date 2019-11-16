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

_net_ = None
_to_tensor_ = None


def vis_parsing_maps(im, parsing, stride):
    """
    # 显示所有部位
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    """
    # 只显示鼻子 眼睛 眉毛 嘴巴
    part_colors = [[255, 255, 255], [255, 255, 255], [25, 170, 0], [255, 170, 0], [254, 0, 170], [254, 0, 170],
                   [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [0, 0, 254], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    """
    part_colors = [[255, 255, 255], [脸], [左眉], [右眉], [左眼], [右眼],
                   [255, 255, 255],
                   [左耳], [右耳], [255, 255, 255], [鼻子], [牙齿], [上唇], 
                   [下唇],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255],
                   [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]
    """

    im = np.array(im)
    vis_parsing = parsing.copy().astype(np.uint8)
    vis_parsing_anno_color = np.zeros((vis_parsing.shape[0], vis_parsing.shape[1], 3)) + 255
    num_of_class = np.max(vis_parsing)
    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    return vis_parsing_anno_color


def _build_net(cp, cuda=False):
    net = BiSeNet(n_classes=19)
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
    input_ = _to_tensor_(input)
    input_ = torch.unsqueeze(input_, 0)
    if cuda:
        input_ = input_.cuda()
    out = _net_(input_)
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0)
    return vis_parsing_maps(input, parsing, stride=1)


def faceparsing_tensor(input, cp, w, cuda=False):
    """
    evaluate with torch tensor
    :param input: torch tensor [B, H, W, C] rang: [0-1], not [0-255]
    :param cp: args.parsing_checkpoint, str
    :param w: tuple len=6 [eyebrow，eye，nose，teeth，up lip，lower lip]
    :param cuda: use gpu to speedup
    :return  tensor, shape:[H, W]
    """
    build_net(cp, cuda)
    out = _net_(input)
    out = out.squeeze()
    return w[0] * out[3] + w[1] * out[4] + w[2] * out[10] + out[11] + out[12] + out[13], out[1]


if __name__ == '__main__':
    src_path = "../../export/faceparsing/src"
    root, name = os.path.split(src_path)
    dst_pth = os.path.join(root, 'dst')
    if not os.path.exists(dst_pth):
        os.mkdir(dst_pth)
    with torch.no_grad():
        list_image = os.listdir(src_path)
        total = len(list_image)
        cuda = torch.cuda.is_available()
        progress = tqdm(range(0, total), initial=0, total=total)
        for step in progress:
            img = cv2.imread(osp.join(src_path, list_image[step]))
            image = cv2.resize(img, (512, 512), cv2.INTER_LINEAR)
            img_1 = faceparsing_ndarray(image, '../dat/79999_iter.pth', cuda)
            save_path = os.path.join(dst_pth, "c-" + list_image[step])
            cv2.imwrite(save_path, img_1)
            image = image[np.newaxis, :, :, :]
            image = np.swapaxes(image, 1, 3)
            image = np.swapaxes(image, 2, 3)

            tensor = torch.from_numpy(image).float() / 255.
            if cuda:
                tensor = tensor.cuda()
            weight = [1.2, 1.4, 1.1, .7, 1., 1.]
            lsi = faceparsing_tensor(tensor, '../dat/79999_iter.pth', weight, cuda)
            for idx, tensor in enumerate(lsi):
                save_path = osp.join(dst_pth, '{0}-{1}'.format(idx, list_image[step]))
                map = tensor.cpu().detach().numpy()
                cv2.imwrite(save_path, map * 10)
