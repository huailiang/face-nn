#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-04-12

import tensorflow as tf
from parse import parser
from model import Artgan


tf.set_random_seed(228)


def main(_):
    args = parser.parse_args()
    tfconfig = tf.ConfigProto(allow_soft_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = Artgan(sess, args)

        if args.phase == 'train':
            print("Train.")
            model.train(args, ckpt_nmbr=args.ckpt_nmbr)
        if args.phase == 'inference' or args.phase == 'test':
            print("Inference.")
            model.inference(args, args.inference_images_dir,
                            resize_to_original=False,
                            to_save_dir=args.save_dir,
                            ckpt_nmbr=args.ckpt_nmbr)

        if args.phase == 'inference_on_frames' or args.phase == 'test_on_frames':
            print("Inference on frames sequence.")
            model.inference_video(args, path_to_folder=args.inference_images_dir[0],
                                  resize_to_original=False,
                                  to_save_dir=args.save_dir,
                                  ckpt_nmbr=args.ckpt_nmbr)

        if args.phase == "export_layers":
            print("export_layers.")
            model.export_layers(args.inference_images_dir,
                                to_save_dir=args.save_dir,
                                ckpt_nmbr=args.ckpt_nmbr)

        if args.phase == "export_arg":
            print("export_arg.")
            model.export_arg(args.ckpt_name)

        sess.close()


if __name__ == '__main__':
    tf.app.run()
