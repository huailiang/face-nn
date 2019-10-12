#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-04

import atexit
import socket
import util.logit as log


class Net(object):
    """
    此模块用来和引擎通信
    使用udp在进程间通信，udp不保证时序性，也不保证引擎一定能收到
    """

    def __init__(self, port1, port2):
        atexit.register(self.close)
        self._port1 = port1
        self._port2 = port2
        self._buffer_size = 1024
        self._open_socket = False
        self._open_send = False
        self._loaded = False
        self._bind = ("localhost", port1)
        log.error("socket start, rcv port:" + str(port1) + "  send port:" + str(port2))

        try:
            self._rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._rcv_socket.bind(self._bind)
            self._open_socket = True
            data = self._rcv_socket.recvfrom(1024)
            log.info("receive data")
            log.info(data[0].decode('utf-8'))
        except Exception as e:
            self._open_socket = False
            self.close()
            log.error(socket.error("socket error" + str(e.message)))
            raise

        try:
            self._snd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._bind2 = ("localhost", port2)
            self._open_send = True
        except Exception as e:
            self._snd_socket.close()
            self._open_send = False
            raise

    def send_recv(self, msg):
        """
        发送之后 也接收
        :param msg:
        """
        try:
            msg = "rcv" + msg
            self._snd_socket.sendto(msg.encode('utf-8'), self._bind2)
            if msg != "quit":
                self.recv()
        except Exception as e:
            log.error(e.message)
            raise

    def only_send(self, msg):
        """
        只发送 不接收
        :param msg:
        """
        try:
            self._snd_socket.sendto(msg.encode('utf-8'), self._bind2)
            print("send success")
        except Exception as e:
            log.error(e.message)
            raise

    def recv(self):
        try:
            data = self._rcv_socket.recvfrom(self._buffer_size)
            print("receive data")
            print(data[0].decode('utf-8'))
        except Exception as e:
            log.error(e.message)
            raise

    def close(self):
        """
        关闭连接
        :return:
        """
        print("socket close")
        if self._open_socket:
            self._rcv_socket.close()
        if self._open_send:
            self._snd_socket.close()
