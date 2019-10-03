import atexit
import socket
import logging
logger = logging.getLogger("nn-face")


class Net(object):
    """
    采用udp通信
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
        print("socket start, rcv port:"+str(port1)+"  send port:"+str(port2))

        try:
            self._rcv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._rcv_socket.bind(self._bind)
            self._open_socket = True
            data = self._rcv_socket.recvfrom(1024)
            print("receive data")
            print(data[0].decode('utf-8'))
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

    def send_recv(self, msg):
        try:
            msg = "rcv"+msg
            self._snd_socket.sendto(msg.encode('utf-8'), self._bind2)
            if msg != "quit":
                self.recv()
        except Exception as e:
            logger.error(e.message)
            raise 

    def only_send(self, msg):
        """
        只发送 不接收
        :param msg:
        :return:
        """
        try:
            self._snd_socket.sendto(msg.encode('utf-8'), self._bind2)
            print("send success")
        except Exception as e:
            logger.error(e.message)
            raise

    def recv(self):
        try:
            data = self._rcv_socket.recvfrom(self._buffer_size)
            print("receive data")
            print(data[0].decode('utf-8'))
        except Exception as e:
            logger.error(e.message)
            raise

    def close(self):
        print("socket close")
        if self._open_socket:
            self._rcv_socket.close()
        if self._open_send:
            self._snd_socket.close()