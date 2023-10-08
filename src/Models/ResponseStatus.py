# -*- coding: utf-8 -*-
"""
ResponseStatus

Author: chengwen
date:   3/28/2019 1:55 PM
Description: 状态码枚举类
usage：
    结构为：错误枚举名-错误码code-错误说明message
"""
from enum import Enum, unique

@unique
class ResponseStatus(Enum):
    OK = {"0": "success"}
    UNKNOWN_EXCEPTION = {"101": "Unknown exception!"}
    METHOD_NOT_FOUND = {"102": "method not found exception!"}
    PARAM_NOT_SET = {"201": "Parameter not set!"}
    PARAM_NOT_MATCH = {"202": "Parameter not match!"}
    PARAM_DATA_EXCEPTION = {"203": "Parameter data exception!"}
    CODE_EXCEPTON = {"400": "Code Running Exception!"}
    EXTERNAL_EXCEPTION = {"405": "External run exception!"}
    FRAME_EXCEPTION = {"500": "Frame run exception!"}
    EXTERNAL_INTERRUPTED = {"505": "Code execution was externally interrupted!"}
    def getCode(self):
        """
        :return: Enum code
        """
        return list(self.value.keys())[0]

    def getMsg(self,ext=""):
        """
        :return: Enum message
        """
        return list(self.value.values())[0]+" "+ext
