#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019-10-12

import logit as log


class Test:
    def __init__(self):
        pass

    def add(self, a, b):
        log.info("%d + %d = %d", a, b, a + b)
