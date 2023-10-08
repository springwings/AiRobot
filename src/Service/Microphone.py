# -*- coding: utf-8 -*-
# python 3.6
"""
Microphone

Author: chengwen
Modifier:
date:   2023/3/13 下午3:15
Description:
"""
import threading
import time
import wave

import numpy as np
import pyaudio


class Microphone():

    last_file = ""

    def __init__(self):
        self.audio_thread = Microphone.AudioRecorder() 

    def micStart(self):
        self.flush_time = int(time.time())
        self.audio_thread.startRec()

    def micPause(self):
        self.audio_thread.pauseRec()

    def micRestart(self):
        self.audio_thread.restartRec()

    def micFlush(self,audiopath):
        """
        syn video and audio

        """
        if self.audio_thread.flush(audiopath,self.flush_time):
            self.flush_time = int(time.time())
            return True
        return False

    class AudioRecorder():
        "Audio class based on pyAudio and Wave"
        def __init__(self, rate=16000, fpb=1024, channels=2):
            self.pause = False
            self.rate = rate
            # 初始短时能量高门限
            self.amp1 = 940
            # 初始短时能量低门限
            self.amp2 = 120
            # 初始短时过零率高门限
            self.zcr1 = 30
            # 初始短时过零率低门限
            self.zcr2 = 2
            # 允许最大静音长度
            self.maxsilence = 45     #允许换气的最长时间
            # 语音的最短长度
            self.minlen = 60        #  过滤小音量
            self.count = 0
            self.silence = 0
            self.cur_status = 0
            self.max_en = 20000
            self.frames_per_buffer = fpb
            self.channels = channels
            self.format = pyaudio.paInt16
            self.audio = pyaudio.PyAudio()
            self.currentFrame = None
            self.stream = self.audio.open(format=self.format,
                                          channels=self.channels,
                                          rate=self.rate,
                                          input=True,
                                          frames_per_buffer = self.frames_per_buffer)
            self.audio_frames = []

        def ZCR(self,curFrame):
            # 过零率
            tmp1 = curFrame[:-1]
            tmp2 = curFrame[1:]
            sings = (tmp1 * tmp2 <= 0)
            diffs = (tmp1 - tmp2) > 0.02
            zcr = np.sum(sings * diffs)
            return zcr

        def STE(self,curFrame):
            # 短时能量
            amp = np.sum(np.abs(curFrame))
            return amp


        def speechStatus(self, amp, zcr):
            """
            实际测试
            说话时的过零率 在0-3之间    呼吸时的过零率在0-5之间
            说话时的短时能量 在940-12000之间    15000-23000之间
            """
            status = 0
            # 0= 静音， 1= 可能开始, 2=确定进入语音段   3语音结束
            if self.cur_status in [0, 1]:    #如果在静音状态或可能的语音状态，则执行下面操作
                # 确定进入语音段
                if amp > self.amp1 or zcr > self.zcr1:    #超过最大  短时能量门限了
                    status = 2
                    self.silence = 0
                    self.count += 1
                # 可能处于语音段   能量处于浊音段，过零率在清音或浊音段
                elif amp > self.amp2 or zcr > self.zcr2:
                    status = 2
                    self.count += 1
                # 静音状态
                else:
                    status = 0
                    self.count = 0
                    self.count = 0
            # 2 = 语音段
            elif self.cur_status == 2:
                # 保持在语音段    能量处于浊音段，过零率在清音或浊音段
                if amp > self.amp2 or zcr > self.zcr2:
                    self.count += 1
                    status = 2
                # 语音将结束
                else:
                    # 静音还不够长，尚未结束
                    self.silence += 1
                    if self.silence < self.maxsilence:
                        self.count += 1
                        status = 2
                    # 语音长度太短认为是噪声
                    elif self.count < self.minlen:
                        status = 0
                        self.silence = 0
                        self.count = 0
                    # 语音结束
                    else:
                        status = 3
                        self.silence = 0
                        self.count = 0
            return status


        def record(self):
            "Audio starts being recorded"
            self.stream.start_stream()
            while True:
                if self.pause is False:
                    data = self.stream.read(self.frames_per_buffer)
                    self.audio_frames.append(data)
                    self.currentFrame = data

        def pauseRec(self):
            # self.stream.stop_stream()
            self.pause = True

        def restartRec(self):
            # self.stream.start_stream()
            self.pause = False

        def flush(self,filepath,lastFlushTime):
            """
            """
            wave_data = np.frombuffer(self.currentFrame, dtype=np.int16)
            wave_data = wave_data * 1.0 / self.max_en
            data = wave_data[np.arange(0, 256)]
            zcr = self.ZCR(data)
            # 获得音频的短时能量, 平方放大
            amp = self.STE(data)**2
            # 返回当前音频数据状态
            # 0= 静音， 1= 可能开始, 2=确定进入语音段   3语音结束
            res = self.speechStatus(amp, zcr)
            self.cur_status = res
            if int(time.time())-lastFlushTime>20:
                res = 3
            if res<=2:
                return False
            elif res==3:
                waveFile = wave.open(filepath, 'wb')
                waveFile.setnchannels(self.channels)
                waveFile.setsampwidth(self.audio.get_sample_size(self.format))
                waveFile.setframerate(self.rate)
                waveFile.writeframes(b''.join(self.audio_frames))
                self.audio_frames.clear()
                waveFile.close()
                return True

        def startRec(self):
            "Launches the audio recording function using a thread"
            audio_thread = threading.Thread(target=self.record)
            audio_thread.start()



if __name__ == '__main__':
    camera = Microphone()
    camera.cameraStart()
    time.sleep(5)
    camera.micFlush("../Tmp/test.wav")
    # camera.fileManage()
