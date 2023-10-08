# -*- coding: utf-8 -*-
# python 3.6
"""
FaceRecognition

Author: chengwen
Modifier:
date:   2023/3/13 上午11:44
Description:
"""
import cv2
from deepface import DeepFace

class FaceRecognition():

    def __init__(self,basePath):
        self.userList = [{"name":"springwing","thumb":"/VM/test22.jpg"},
                         {"name":"springwing","thumb":"/VM/test22.jpg"}]
        self.basePath = basePath

    def __call__(self,videoPath):
        cap = cv2.VideoCapture(videoPath)
        picPath = self.basePath+'/_tmp_.jpg'
        users = []
        while(True):
            success, frame = cap.read()
            if success:
                cv2.imwrite(picPath,frame)
                user = self.processPic(picPath)
                if user is not None and user not in users:
                    users.append(user)
            else:
                break
        cap.release()
        return users

    def processPic(self,picPath):
        for person in self.userList:
            try:
                obj = DeepFace.verify(person["thumb"], picPath
                                      , model_name = 'ArcFace', detector_backend = 'retinaface')
                if obj["verified"]:
                    return person["name"]
            except:
                return
        return None