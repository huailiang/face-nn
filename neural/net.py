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
    def __init__(self, port):
        atexit.register(self.close)
        self._port = port
        self._buffer_size = 95 * 4
        self._open_socket = False
        self._loaded = False
        print("socket port is:"+str(self._port))

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(("localhost", self._port))
            self._open_socket = True
        except Exception as e:
            self._open_socket = False
            self.close()
            raise socket.error("socket error"+str(e.message))
        self._socket.settimeout(30)
        try:
            try:
                self._socket.listen(1)
                print("socker is listenning")
                self._conn, _ = self._socket.accept()
                self._conn.settimeout(30)
                s = self._conn.recv(self._buffer_size)
                print("init recieve")
                print(s)
            except socket.timeout as e:
                logger.error(e.message)
                raise 
            self._loaded = True
        except Exception as e:
            logger.error(e.message)
            raise 


    def sendRcv(self, buffer):
        try:
            self._conn.send(buffer)
            if buffer != 0x8:
                self.recv()
        except Exception as e:
            logger.error(e.message)
            raise 

    def onlySend(self, buffer):
        try:
            self._conn.send(buffer)
        except Exception as e:
            logger.error(e.message)
            raise



    def recv(self):
        try:
            s = self._conn.recv(self._buffer_size)
            message_len = struct.unpack('I', bytearray(s[:4]))[0]
        except Exception as e:
            logger.error(e.message)
            raise
    

    def close(self):
        print("socket close")
        if self._open_socket:
            self._socket.close()
        if self._loaded:
            self._conn.close()
