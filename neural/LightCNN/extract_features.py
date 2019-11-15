from __future__ import print_function
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import cv2
from lightcnn.light_cnn import LightCNN_29Layers_v2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Feature Extracting')
parser.add_argument('--arch', '-a', metavar='ARCH', default='LightCNN')
parser.add_argument('--cuda', '-c', default=False)
parser.add_argument('--resume', default='../dat/LightCNN_29Layers_V2_checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='../output/lightcnn_feature.txt', type=str, metavar='PATH',
                    help='save root path for features of face images.')
parser.add_argument('--num_classes', default=80013, type=int, metavar='N', help='mini-batch size (default: 79077)')


def main():
    args = parser.parse_args()
    model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    model.eval()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    img_list = read_list('../dat/LightCNN_list')
    transform = transforms.Compose([transforms.ToTensor()])
    count = 0
    input = torch.zeros(1, 1, 128, 128)
    for img_name in img_list:
        count = count + 1
        img = cv2.imread(os.path.join('../dat/', img_name), cv2.IMREAD_GRAYSCALE)
        img = np.reshape(img, (128, 128, 1))
        img = transform(img)
        input[0, :, :, :] = img
        print("input shape: ", input.shape)
        start = time.time()
        if args.cuda:
            input = input.cuda()
        with torch.no_grad():
            _, features = model(input)
            end = time.time() - start
            print("{}({}/{}). Time: {}".format(os.path.join('../dat/', img_name), count, len(img_list), end))
            save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])


def read_list(list_path):
    img_list = []
    print("list_path", list_path)
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split()
            img_list.append(img_path[0])
    print('There are {} images..'.format(len(img_list)))
    return img_list


def save_feature(save_path, img_name, features):
    img_path = os.path.join(save_path, img_name)
    img_dir = os.path.dirname(img_path) + '/'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fname = os.path.splitext(img_path)[0]
    fname = fname + '.feat'
    fid = open(fname, 'wb')
    fid.write(features)
    fid.close()


if __name__ == '__main__':
    main()
