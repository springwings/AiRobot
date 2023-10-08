# -*- coding: utf-8 -*-
# python 3.6
"""
test

Author: chengwen
Modifier:
date:   2023/9/28 下午5:23
Description:
"""
from src.Service.Avatar import Avatar

if __name__ == '__main__':
    avatar = Avatar("/work/py3/airobot/Tmp", "/work/py3/faceControl/examples/source_image/art_12.png", "/work/py3/airobot/Dist/avatar")
    avatar("/VM/test.wav")