# -*- coding: utf-8 -*-
# python 3.6
"""
tts

Author: chengwen
Modifier:
date:   2023/3/13 上午11:43 
"""
import os
import random
from paddlespeech.cli.tts.infer import TTSExecutor
from playsound import playsound

class TTS():
    """

    """
    def __init__(self,basePath):
        self.tts = TTSExecutor()
        self.basePath = basePath+"Tmp/"
        self.tts("hello world!")

    def __call__(self, content,play=False):
        filePath = None
        if len(content)>1:
            fname = str(random.randint(1,99))+"_tmp.wav"
            filePath = self.basePath + fname
            self._predict(content,filePath)
            if play:
                self._play(filePath)
        return filePath

    def _predict(self,content,filePath):
        self.tts(text=content, output=filePath)

    def _play(self,filePath,remove=True):
        playsound(filePath)
        if remove:
            os.remove(filePath)


if __name__ == '__main__':
    tts = TTS("/work/py3/airobot/")
    tts("来安装环境")
