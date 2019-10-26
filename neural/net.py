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
    此模块用来和引擎通信
    使用udp在进程间通信，udp不保证时序性，也不保证引擎一定能收到
    """

    def __init__(self, port, args):
        atexit.register(self.close)
        self.port = port
        self.args = args
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

    def send_params(self, param, name):
        """
        发送参数给引擎
        :param param: 捏脸参数
        """
        shape = utils.curr_roleshape(self.args.path_to_dataset)
        dic = {"shape": shape, "param": param, "name": name}
        self._send('p', json.dumps(dic))

    def send_message(self, message):
        self._send('m', message)

    def _send(self, cmd, message):
        """
        私有方法 发送消息
        :param message: 消息体
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
        关闭连接
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
    net = Net(5011, args)
    while True:
        r_input = input("command: ")
        if r_input == "m":
            net.send_message("hello world")
        elif r_input == "p":
            params = utils.random_params(95)
            net.send_params(params, str(random.randint(1000, 9999)))
        elif r_input == "q":
            net.close()
            break
        else:
            log.error("unknown code, quit")
            net.close()
            break
