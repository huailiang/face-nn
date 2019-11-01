#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-12

import os
import logging
import sys

"""

level:
    FATAL：致命错误
    CRITICAL：特别糟糕的事情，如内存耗尽、磁盘空间为空，一般很少使用
    ERROR：发生错误时，如IO操作失败或者连接问题
    WARNING：发生很重要的事件，但是并不是错误时，如用户登录密码错误
    INFO：处理请求或者状态变化等日常事务
    DEBUG：调试过程中使用DEBUG等级，如算法中每个循环的中间状态

formatter:
    %(levelno)s：打印日志级别的数值
    %(levelname)s：打印日志级别的名称
    %(pathname)s：打印当前执行程序的路径，其实就是sys.argv[0]
    %(filename)s：打印当前执行程序名
    %(funcName)s：打印日志的当前函数
    %(lineno)d：打印日志的当前行号
    %(asctime)s：打印日志的时间
    %(thread)d：打印线程ID
    %(threadName)s：打印线程名称
    %(process)d：打印进程ID
    %(message)s：打印日志信息
    
handler:
    StreamHandler：logging.StreamHandler；日志输出到流，可以是sys.stderr，sys.stdout或者文件
    FileHandler：logging.FileHandler；日志输出到文件
    BaseRotatingHandler：logging.handlers.BaseRotatingHandler；基本的日志回滚方式
    RotatingHandler：logging.handlers.RotatingHandler；日志回滚方式，支持日志文件最大数量和日志文件回滚
    TimeRotatingHandler：logging.handlers.TimeRotatingHandler；日志回滚方式，在一定时间区域内回滚日志文件
    SocketHandler：logging.handlers.SocketHandler；远程输出日志到TCP/IP sockets
    DatagramHandler：logging.handlers.DatagramHandler；远程输出日志到UDP sockets
    SMTPHandler：logging.handlers.SMTPHandler；远程输出日志到邮件地址
    SysLogHandler：logging.handlers.SysLogHandler；日志输出到syslog
    NTEventLogHandler：logging.handlers.NTEventLogHandler；远程输出日志到Windows NT/2000/XP的事件日志
    MemoryHandler：logging.handlers.MemoryHandler；日志输出到内存中的指定buffer
    HTTPHandler：logging.handlers.HTTPHandler；通过"GET"或者"POST"远程输出到HTTP服务器
"""


def init(name="LOG", level=logging.INFO, log_path="./output/log.txt"):
    clear_log(log_path)
    global _log
    _log = logging.getLogger(name)
    _log.setLevel(level)
    console = logging.StreamHandler()
    console.emit = add_console_to_emit(console.emit)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    console.setLevel(level)
    _log.addHandler(console)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_path)
    handler.emit = add_file_to_emit(handler.emit)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    _log.addHandler(handler)


def clear_log(log_path):
    try:
        if os.path.exists(log_path):
            os.remove(log_path)
    except IOError:
        logging.error("log module error")
        raise


def add_file_to_emit(fn):
    def new(*args):
        msg = args[0].msg
        if isinstance(msg, str):
            idx = msg.find('\t')
            if idx <= 0:
                idx = 5
            else:
                idx += 1
            args[0].msg = args[0].msg[idx:-4]
        return fn(*args)

    return new


def add_console_to_emit(fn):
    def new(*args):
        levelno = args[0].levelno
        if levelno >= 50:
            color = '\x1b[35m'  # pink - critical
        elif levelno >= 40:
            color = '\x1b[31m'  # red - error
        elif levelno >= 30:
            color = '\x1b[33m'  # yellow - warn
        elif levelno >= 20:
            color = '\x1b[37m'  # white - info
        elif levelno >= 10:
            color = '\x1b[32m'  # green - debug
        else:
            color = '\x1b[0m'  # normal
        msg = args[0].msg
        if isinstance(msg, str):
            args[0].msg = color + format_header() + args[0].msg + '\x1b[0m'
        else:
            args[0].msg = color + format_header() + str(args[0].msg) + '\x1b[0m'
        return fn(*args)

    def format_header():
        filename, no, function = get_stack_info()
        filename = os.path.basename(filename)
        header = filename + ":" + str(no) + "\t"
        return header

    def get_stack_info():
        currentframe = lambda: sys._getframe(3)
        f = currentframe()
        while f is not None:
            f = f.f_back
            if hasattr(f, "f_code"):
                co = f.f_code
                if co.co_filename.find("logit.py") > 0:
                    f = f.f_back
                    co = f.f_code
                    return co.co_filename, f.f_lineno, co.co_name

    return new


def is_init():
    try:
        type(eval("_log"))
    except:
        return False
    else:
        return True


def set_level(level):
    if is_init():
        _log.setLevel(level)


def debug(msg, *args, **kwargs):
    if is_init():
        _log.debug(msg, *args, **kwargs)
    else:
        print(msg, *args)


def info(msg, *args, **kwargs):
    if is_init():
        _log.info(msg, *args, **kwargs)
    else:
        print(msg, *args)


def warn(msg, *args, **kwargs):
    if is_init():
        _log.warning(msg, *args, **kwargs)
    else:
        print(msg, *args)


def error(msg, *args, **kwargs):
    if is_init():
        _log.error(msg, *args, **kwargs)
    else:
        print(msg, *args)


def fatal(msg, *args, **kwargs):
    if is_init():
        _log.fatal(msg, *args, **kwargs)
    else:
        print(msg, *args)


def critical(msg, *args, **kwargs):
    if is_init():
        _log.critical(msg, *args, **kwargs)
    else:
        print(msg, *args)


if __name__ == '__main__':
    init("test", level=logging.DEBUG, log_path="../output/test.txt")
    debug("hello world {0} is {1}".format(12, "x"))
    info("hello {0} is {1}".format((1, 2, 3), type("x")))
    warn("hello world %s is %d", "amy", 123)
    error("Houston, we have a %s", "major disaster", exc_info=1)
