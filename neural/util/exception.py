#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-20

import logit as log
import traceback


class NnException(Exception):
    """docstring for NnException"""

    def __init__(self, message):
        log.error("neural error: " + message)
        self.message = message


class IOException(Exception):
    """docstring for TimeoutException"""

    def __init__(self, message):
        log.error("socket error: " + message)
        self.message = message
