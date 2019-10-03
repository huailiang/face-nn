import logging
logger = logging.getLogger("nn-face")


class NnException(object):
    """docstring for NnException"""
    def __init__(self, message):
        logger.error("neural error: "+message)


class TimeoutException(Exception):
    """docstring for TimeoutException"""
    def __init__(self, message):    
        logger.error("socket error: "+message)