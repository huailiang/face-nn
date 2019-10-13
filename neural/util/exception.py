#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-09-20


import traceback
import util.logit as log


class NeuralException(Exception):
    """docstring for NnException"""

    def __init__(self, message):
        log.info("neural error: " + message)
        self.message = "neural exception: " + message


class IOException(Exception):
    """docstring for io"""

    def __init__(self, message):
        log.error("io error: " + message)
        self.message = message
