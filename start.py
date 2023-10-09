#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
start

Author: chengwen
date:   3/28/2019 1:55 PM
Description: app entrance position
"""
import datetime
import logging
import os
import sys
import time
from collections import deque
from queue import Queue
import threading

from src.Service.ASR import ASR
# from src.Service.Avatar import Avatar
from src.Service.Camera import Camera
from src.Service.ChatGpt import ChatGpt
# from Service.FaceRecognition import FaceRecognition
from src.Service.Microphone import Microphone
# from src.Service.PlayVideo import PlayVideo
from src.Service.TTS import TTS
from src.utils.Common import formatException, getLogger, is_contain_chinese

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
        self.processAudioQueue = Queue(maxsize=3)
        self.processVideoQueue = Queue(maxsize=3)
        self.asr = ASR(basePath)
        self.tts = TTS(basePath)
        # self.playvideo = PlayVideo(basePath)
        # self.avatar = Avatar(basePath+"Tmp", "./avatar.jpg", basePath+"Dist/avatar")
        self.chatgpt = ChatGpt()
        self.storePath = basePath+"Tmp/"

        self._logger = getLogger(basePath+"log/")
        from paddlespeech.cli.log import logger
        logger.logger.removeHandler(logger.handler)
        logger.logger.setLevel(logging.ERROR)
        self.chatHistory = deque(maxlen=2)

    def DM(self):
        """
        Dialogue management
        """
        pass

    def fetchVideo(self):
        current = str(int(datetime.datetime.now().timestamp()))
        camera = Camera(videoPath=self.storePath+current+".avi")
        camera.cameraStart()
        while(True):
            time.sleep(self.processInterval)
            newcurrent = str(int(datetime.datetime.now().timestamp()))
            isPause = False
            while isPause==False:
                isPause = camera.cameraFlush(newVideoPath=self.storePath+newcurrent+".avi")
                time.sleep(0.2)
            self.processVideoQueue.put(current)
            current = newcurrent

    def fetchAudio(self):
        current = str(int(datetime.datetime.now().timestamp()))
        self.mic = Microphone()
        self.mic.micStart()
        time.sleep(3)
        isFirst = True
        while(True):
            time.sleep(self.processInterval)
            newcurrent = str(int(datetime.datetime.now().timestamp()))
            isPause = False
            while isPause is False:
                isPause = self.mic.micFlush(audiopath=self.storePath+current+".wav")
                time.sleep(0.2)
            if self.processAudioQueue.full():
                tmp = self.processAudioQueue.get()
                os.remove(self.storePath+tmp+".wav")

            self.processAudioQueue.put(current)
            if isFirst:
                time.sleep(25)
            else:
                time.sleep(9)
            current = newcurrent

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

    def processAudio(self):
        """
        process Audio
        """
        self.chatHistory.append({'role': 'system', 'content': '你是中文百科助手。'})
        while(True):
            if self.processAudioQueue.empty()==False:
                current = self.processAudioQueue.get()
                audioPath = self.storePath+current+".wav"
                qu = ""
                ans = ""
                try:
                    qu = self.asr(audioPath)
                except Exception as e:
                    self._logger.error(formatException(e))
                if len(qu)>1:
                    print("Q: {}".format(qu))
                    try:
                        self.chatHistory.append({"role": "user", "content": qu})
                        ans = self.chatgpt(list(self.chatHistory))
                        self.chatHistory.append({"role": "system", "content": ans})
                    except Exception as e:
                        self._logger.error(formatException(e))
                    print("A:{}".format(ans))
                    if is_contain_chinese(ans):
                        self.mic.micPause()
                        try:
                            print("start speak...")
                            tmp_audio = self.tts(ans)
                            print("--gen {}".format(tmp_audio))
                            if tmp_audio is not None:
                                self.avatar(tmp_audio,enhancer=None)
                                time.sleep(len(ans)*0.2)
                            print("finished speak...")
                        except Exception as e:
                            self._logger.error(formatException(e))
                        self.mic.micRestart()
                try:
                    os.remove(audioPath)
                except Exception as e:
                    pass

    def clearDir(self):
        for f in os.listdir(self.storePath):
            try:
                os.remove(os.path.join(self.storePath, f))
            except:
                pass

    def start(self):
        print("clean "+self.storePath)
        self.clearDir()
        threading.Thread(target=self.fetchAudio).start()
        threading.Thread(target=self.processAudio).start()
        # multiprocessing.Process(target=self.fetchVideo).start()
        #threading.Thread(target=self.processVideo).start()

if __name__ == "__main__":
    chat = Chat()
    chat.start()
