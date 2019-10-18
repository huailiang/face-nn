#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-12

import os
import logging

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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_path)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(level)
    _log.addHandler(handler)
    _log.addHandler(console)


def clear_log(log_path):
    try:
        if os.path.exists(log_path):
            os.remove(log_path)
    except IOError:
        raise


def set_level(level):
    _log.setLevel(level)


def debug(msg, *args):
    _log.debug(msg, *args)


def info(msg, *args):
    _log.info(msg, *args)


def warn(msg, *args):
    _log.warning(msg, *args)


def error(msg, *args):
    _log.error(msg, *args)


def fatal(msg, *args):
    _log.fatal(msg, *args)


def critical(msg, *args):
    _log.critical(msg, *args)
