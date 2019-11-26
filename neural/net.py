#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

import atexit
import socket
import json
import utils
import random
import util.logit as log


class Net(object):
    """
    此模块用来和引擎通信 unity菜单栏：Tools->Connect
    使用udp在进程间通信，udp不保证时序性，也不保证引擎一定能收到
    """

    def __init__(self, port, arguments):
        """
        net initial
        :param port: udp 端口号
        :param arguments: parse options
        """
        atexit.register(self.close)
        self.port = port
        self.args = arguments
        self.buffer_size = 1024
        self.open = False
        log.info("socket start,  port:" + str(port))
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.bind = ("localhost", port)
            self.open = True
        except Exception as e:
            self.close()
            raise

    def send_params(self, param, name, step):
        """
        batch params
        :param step: train step
        :param param: torch.Tensor [batch, params_cnt]
        :param name: list of name [batch]
        """
        list_ = param.cpu().detach().numpy().tolist()
        cnt = len(list_)
        for i in range(cnt):
            self.send_param(list_[i], name[i][:-4] + "_" + str(step))  # name remove ext .jpg

    def send_param(self, param, name):
        """
        发送参数给引擎
        :param name: 图片名
        :param param: 捏脸参数
        """
        shape = utils.curr_roleshape(self.args.path_to_dataset)
        dic = {"shape": shape, "param": param, "name": name}
        self._send('p', json.dumps(dic))

    def send_message(self, message):
        self._send('m', message)

    def _send(self, cmd, message):
        """
        private method to send message
        :param message: message body
        """
        if self.open:
            try:
                message = cmd + message
                self.socket.sendto(message.encode('utf-8'), self.bind)
            except Exception as e:
                log.error(e)
                raise
        else:
            log.warn("connect closed")

    def close(self):
        """
        close connect
        """
        if self.open:
            log.warn("socket close")
            self._send('q', "-")  # quit
            self.socket.close()
            self.open = False


if __name__ == '__main__':
    from parse import parser
    import logging

    args = parser.parse_args()
    log.init("FaceNeural", logging.INFO, log_path="./output/log.txt")
    log.info(utils.curr_roleshape(args.path_to_dataset))

    net = Net(args.udp_port, args)

    while True:
        r_input = input("command: ")
        if r_input == "m":
            net.send_message("hello world")
        elif r_input == "p":
            params = utils.random_params(args.params_cnt)
            net.send_param(params, str(random.randint(1000, 9999)))
        elif r_input == "q":
            net.close()
            break
        else:
            log.error("unknown code, quit")
            net.close()
            break
