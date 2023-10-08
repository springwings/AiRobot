# -*- coding: utf-8 -*-
# python 3.6
"""
Camera

Author: chengwen
Modifier:
date:   2023/3/13 下午3:15
Description:
"""
import threading
import time

import cv2


class Camera():

    last_file = ""

    def __init__(self,videoPath=""):
        self.video_thread = Camera.VideoRecorder(videoPath)

    def cameraStart(self):
        self.video_thread.start()

    def cameraPause(self):
        pass

    def cameraRestart(self):
        pass

    def cameraFlush(self,newVideoPath):
        """
        syn video and audio

        """
        if self.video_thread.flush(newVideoPath):
            return True
        return False



    class VideoRecorder():
        "Video class based on openCV"
        def __init__(self, videoPath="temp_video.avi", fourcc="MJPG", sizex=640, sizey=480, camindex=0, fps=30):
            self.open = True
            self.device_index = camindex
            self.fps = fps                  # fps should be the minimum constant rate at which the camera can
            self.fourcc = fourcc            # capture images (with no decrease in speed over time; testing is required)
            self.frameSize = (sizex, sizey) # video formats and sizes also depend and vary according to the camera used
            self.video_filename = videoPath
            self.video_cap = cv2.VideoCapture(self.device_index)
            self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
            self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)

        def record(self):
            self.frame_counts = 1
            self.start_time = time.time()
            "Video starts being recorded"
            while True:
                ret, video_frame = self.video_cap.read()
                if ret and self.open:
                    self.video_out.write(video_frame)
                    self.frame_counts += 1
                    time.sleep(1/self.fps)


        def flush(self,new_video_filename):
            "Finishes the video recording therefore the thread too"
            self.open = False
            self.video_out.release()
            self.video_out = cv2.VideoWriter(new_video_filename, self.video_writer, self.fps, self.frameSize)
            self.frame_counts = 1
            self.open = True
            return True


        def start(self):
            "Launches the video recording function using a thread"
            video_thread = threading.Thread(target=self.record)
            video_thread.start()






if __name__ == '__main__':
    camera = Camera(videoPath="../Tmp/test.avi")
    camera.cameraStart()
    time.sleep(5)
    camera.cameraFlush()
    # camera.fileManage()