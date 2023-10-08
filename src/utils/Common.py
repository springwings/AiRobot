# -*- coding: utf-8 -*-
"""
Common

Author: chengwen
date:   3/28/2019 1:55 PM
Description:Global Common functions
"""
import datetime
import logging
import re
from enum import Enum

SYS_LOGGER = logging.getLogger()
PROJECT_NAME = "AIROBOT"

DEBUG = False

class LOG_LEVEL(Enum):
    """
    defined locality log type
    """
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

def getLogger(basePath):
    """
    get singleton logger to write logs
    :return:logger
    """
    fname = PROJECT_NAME + "_" + datetime.datetime.now().strftime('%Y%m%d')
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]  %(message)s')
    logHandler = logging.FileHandler(basePath + fname + ".log")
    logHandler.setFormatter(formatter)
    SYS_LOGGER.addHandler(logHandler)
    SYS_LOGGER.setLevel(logging.ERROR)
    if DEBUG:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        SYS_LOGGER.setLevel(logging.DEBUG)
        SYS_LOGGER.addHandler(console_handler)

    return SYS_LOGGER


def formatException(e):
    return str(e)

def checkParameters(args, checkParameters):
    for param in checkParameters.split(','):
        if (param not in args):
            return False
    return True

def getParameters(request, response):
    if request.json == None:
        response.setRequest(request.values.to_dict())
        return request.values.to_dict()
    response.setRequest(request.json)
    return request.json

def getBoolParam(param):
    if param is None or str(param).lower()=='false':
        return False
    return True

def is_contain_chinese(contents):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = pattern.search(contents)
    return True if match else False