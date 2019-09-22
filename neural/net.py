import atexit
import io
import json
import os
import socket
import traceback
import struct
import time
import logging
logger = logging.getLogger("nn-face")



class Net(object):
    """docstring for Net"""
    def __init__(self, port1, port2):
        atexit.register(self.close)
        self._port1 = port1
        self._port2 = port2
        self._buffer_size = 95 * 4
        self._open_socket = False
        self._open_send = False
        self._loaded = False
        self._bind = ("localhost", port1)
        print("socket rcv port is:"+str(port1)+"  send port:"+str(port2))

        try:
            self._rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._rcv_socket.bind(self._bind)
            self._open_socket = True
            data = self._rcv_socket.recvfrom(1024)
            print("receive data")
            print(data)
        except Exception as e:
            self._open_socket = False
            self.close()
            logger.error(socket.error("socket error"+str(e.message)))
            raise 

        try:
            self._snd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._bind2 = ("localhost", port2)
            self._open_send = True
        except Exception as e:
            self._snd_socket.close()
            self._open_send = False
            raise


    def sendRcv(self, buffer):
        try:
            self._snd_socket.sendto(buffer.encode(), self._bind2)
            if buffer != 0x8:
                self.recv()
        except Exception as e:
            logger.error(e.message)
            raise 

    def onlySend(self, buffer):
        try:
            self._snd_socket.sendto(buffer.encode(), self._bind2)
            print("send success")
        except Exception as e:
            logger.error(e.message)
            raise


    def recv(self):
        try:
            s = self._rcv_socket.recvfrom(self._buffer_size)
            message_len = struct.unpack('I', bytearray(s[:4]))[0]
        except Exception as e:
            logger.error(e.message)
            raise
    

    def close(self):
        print("socket close")
        if self._open_socket:
            self._rcv_socket.close()
        if self._open_send:
            self._snd_socket.close()