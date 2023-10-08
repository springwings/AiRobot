# -*- coding: utf-8 -*-
# python 3.6
"""
PlayVideo

Author: chengwen
Modifier:
date:   2023/10/7 上午11:48
Description:
"""
import os
import time

import vlc


class PlayVideo():

    def __init__(self,basePath):
        self.current_time = int(time.time())-3600*24*10
        self.video_folder = basePath+"Tmp/"
        self.default_video = basePath+"default_avatar.mp4"
        self.instance = vlc.Instance('--no-xlib')
        self.media_list = self.instance.media_list_new([])
        # 创建一个媒体列表播放器
        self.player = self.instance.media_list_player_new()
        self.player.set_media_list(self.media_list)

    def __call__(self):
        media = self.instance.media_new(self.default_video)
        self.media_list.add_media(media)
        self.player.play()
        is_first = True
        while True:
            new_video = self.checkNewVideo()
            if is_first:
                if new_video != "__.avi":
                    self.playVideo(new_video)
                else:
                    time.sleep(1)
                    continue
            while self.player.get_state() in (vlc.State.Playing, vlc.State.Buffering):
                pass

    def checkNewVideo(self):
        while True:
            files = os.listdir(self.video_folder)
            for file in files:
                if file.endswith(".mp4"):
                    ctime = int(os.path.getctime(self.video_folder+"/"+file))
                    if self.current_time<ctime:
                        self.current_time = ctime
                        return file
            return "__.avi"

    def playVideo(self,video_file,is_first=False):
        video_path = self.video_folder+"/"+video_file
        if not os.path.exists(video_path):
            print(f"Playing default video.")
            media = self.instance.media_new(self.default_video)
            self.media_list.add_media(media)
        else:
            media = self.instance.media_new(video_path)
            self.media_list.add_media(media)
        self.player.play()

if __name__ == "__main__":
    PL = PlayVideo("/work/py3/airobot/")
    PL()
