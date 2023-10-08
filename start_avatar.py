#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
start

Author: chengwen
date:   3/28/2019 1:55 PM
Description: app entrance position
"""
import os
import sys
import threading
from queue import Queue

# from Service.FaceRecognition import FaceRecognition 
from src.Service.PlayVideo import PlayVideo
from src.utils.Common import formatException

os.environ['TZ'] = 'Asia/Shanghai'

current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)



class Chat():
    """

    """

    def __init__(self):
        basePath = "/work/py3/airobot/"
        self.processInterval = 3
        self.processVideoQueue = Queue(maxsize=3)
        self.playvideo = PlayVideo(basePath)
        self.storePath = basePath+"Tmp/"


    def DM(self):
        """
        Dialogue management
        """
        pass

    def processVideo(self):
        """
        process Video
        """
        # faceRecon = FaceRecognition(self.storePath)
        faceRecon = None
        self.playvideo()
        while(True):
            if self.processVideoQueue.empty()==False:
                current = self.processVideoQueue.get()
                videoPath = self.storePath+current+".avi"
                users = faceRecon(videoPath=videoPath)
                if len(users)>1:
                    print("start answer")
                    try:
                        self.tts("您好,"+" ".join(users))
                    except Exception as e:
                        self._logger.error(formatException(e))
                try:
                    os.remove(videoPath)
                except Exception as e:
                    self._logger.error(formatException(e))


    def start(self):
        threading.Thread(target=self.processVideo).start()

if __name__ == "__main__":
    chat = Chat()
    chat.start()
