# -*- coding: utf-8 -*-
"""
Response

Author: chengwen
date:   3/28/2019 1:55 PM
Description:define rest service response data
"""

import json
import time

from ResponseStatus import ResponseStatus

class extJsonEncoder(json.JSONEncoder):
    """
    support json encode array and bytes
    """
    def default(self, obj):
        import numpy
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding=DATA_ENCODING.utf.value)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


class Response:

    __runSpaceDir = None


    def __init__(self):
        self.__data = {}
        self.setStatus(ResponseStatus.OK.getCode())
        self.setInfo()
        self.__data['request'] = {}
        self.__data['data'] = {}
        self.startTime = time.time()

    def setRunSpaceDir(self,runSpaceDir):
        self.__runSpaceDir = runSpaceDir

    def setEnv(self, env):
        self.__data['env'] = env

    def setStatus(self, status):
        self.__data['status'] = status

    def getStatus(self):
        return int(self.__data['status'])

    def setInfo(self, info="success"):
        self.__data['info'] = info

    def getInfo(self):
        return self.__data['info']

    def setDataType(self,dtype):
        self.__data['dataType'] = dtype

    def setKV(self, k, v,autoResume=False):
        self.__data['data'][k] = self.resume(v) if autoResume else v

    def setRequest(self, rq):
        self.__data['request'] = rq

    def recoverResponse(self,path):
        with open(path, 'r') as f:
            self.__data = json.loads(f.read())

    def getResponse(self):
        return self.__data

    def setData(self, data={}):
        self.__data['data'] = data

    def addData(self,data={}):
        if len(self.__data['data'])==0:
            self.setData(data)
        else:
            self.getModelData().update(data['datas'])

    def getData(self):
        return self.__data['data']

    def getModelData(self):
        return self.__data['data']['datas']

    def setFramework(self,framework):
        self.__data['framework'] = framework

    def toString(self):
        self.__data['useTime'] = str(round((time.time() - self.startTime) * 1000, 3)) + " ms"
        self.__data['createTime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.__data['__SOURCE'] = "NER"
        self.__data['__DEBUG'] = False
        self.__data['__VERSION'] = 0.1
        return json.dumps(self.__data, cls=extJsonEncoder,indent=4,ensure_ascii=False)
