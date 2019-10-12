#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: penghuailiang
# @Date  : 2019/10/12

import logit as log
from test import Test

if __name__ == '__main__':
    log.init()
    log.info("test %s %d", "wwe", 123)
    log.info("test")
    log.warn("test %d %s", 123, "xxxx")
    test = Test()
    test.add(1, 3)
