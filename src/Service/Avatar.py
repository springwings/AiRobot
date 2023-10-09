# -*- coding: utf-8 -*-
# python 3.6
"""
Avatar

Author: chengwen
Modifier:
date:   2023/9/28 上午9:51
Description:
"""
import os
import random
import shutil

import torch

from src.facerender.animate import AnimateFromCoeff
from src.utils.generate_batch import get_data
from src.utils.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract
from src.utils.test_audio2coeff import Audio2Coeff


class Avatar():

    def __init__(self, storePath, avatarPic, modelDir, size=256, preprocess="full"):
        self.avatar_path = avatarPic
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("Avatar {}".format(self.device))
        self.modelBaseDir = modelDir
        os.environ["MODEL_AVATAR_PATH"] = self.modelBaseDir
        self.save_dir = storePath + "/avatar_tmp"
        os.makedirs(self.save_dir, exist_ok=True)
        self.size = size
        self.preprocess = preprocess
        self.initModel()
        print("init Avatar success.")

    def initModel(self,ref_pose=None,ref_eyeblink=None):
        """
        @param ref_pose           path to reference video providing pose
        @param ref_eyeblink       path to reference video providing eye blinking
        @return None
        """
        sadtalker_paths = init_path(self.modelBaseDir + "/base", self.modelBaseDir + "/config", self.size, None,
                                    self.preprocess)
        self.preprocess_model = CropAndExtract(sadtalker_paths, self.device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, self.device)
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, self.device)
        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(self.save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print('3DMM Extraction for source image')
        self.first_coeff_path, self.crop_pic_path, self.crop_info = self.preprocess_model.generate(self.avatar_path, first_frame_dir,
                                                                                    self.preprocess, \
                                                                                    source_image_flag=True,
                                                                                    pic_size=self.size)


        if self.first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return

        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
            ref_eyeblink_frame_dir = os.path.join(self.save_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            self.ref_eyeblink_coeff_path, _, _ = self.preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir,
                                                                           self.preprocess,
                                                                           source_image_flag=False)
        else:
            self.ref_eyeblink_coeff_path = None

        if ref_pose is not None:
            if ref_pose == ref_eyeblink:
                self.ref_pose_coeff_path = self.ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
                ref_pose_frame_dir = os.path.join(self.save_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                self.ref_pose_coeff_path, _, _ = self.preprocess_model.generate(ref_pose, ref_pose_frame_dir,
                                                                           self.preprocess,
                                                                           source_image_flag=False)
        else:
            self.ref_pose_coeff_path = None

    def __call__(self, audio_path, pose_style=0,  batch_size=8, input_yaw=None,
                 input_yaw_list=None, input_pitch_list=None, input_roll_list=None,
                 face3dvis=None, enhancer="gfpgan",
                 background_enhancer=None, size=256, still_mode=False, expression_scale=1.,
                 face3dvisParams=None):
        """
        @param pose_style         input pose style from [0, 46)
        @apram batch_size         the batch size of facerender
        @param input_yaw the      input yaw degree of the user
        @param input_yaw_list     the input yaw degree of the user
        @param input_pitch_list   the input pitch degree of the user
        @param input_roll_list    the input roll degree of the user
        @param enhancer           Face enhancer, [gfpgan, RestoreFormer]
        @param background_enhancer background enhancer, [realesrgan]
        @param still_mode          can crop back to the original videos for the full body aniamtion
        @param expression_scale    the batch size of facerender
        """

        # audio2ceoff
        batch = get_data(self.first_coeff_path, audio_path, self.device, self.ref_eyeblink_coeff_path)
        coeff_path = self.audio_to_coeff.generate(batch, self.save_dir, pose_style, self.ref_pose_coeff_path)

        # 3dface render
        if face3dvis:
            from src.face3d.visualize import gen_composed_video
            gen_composed_video(face3dvisParams, self.device, self.first_coeff_path, coeff_path, audio_path,
                               os.path.join(self.save_dir, '3dface.mp4'))

        # coeff2video
        data = get_facerender_data(coeff_path, self.crop_pic_path, self.first_coeff_path, audio_path,
                                   batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                   expression_scale=expression_scale, still_mode=still_mode, preprocess=self.preprocess,
                                   size=size)

        result = self.animate_from_coeff.generate(data, self.save_dir, self.avatar_path, self.crop_info, \
                                                  enhancer=enhancer, background_enhancer=background_enhancer,
                                                  preprocess=self.preprocess, img_size=size)

        tmp_path = self.save_dir + str(random.randint(1,99))+ '.mp4'
        shutil.move(result, tmp_path)
        # shutil.rmtree(self.save_dir)
        return tmp_path

if __name__ == '__main__':
    avatar = Avatar("/work/py3/airobot/Tmp", "/VM/avatar.jpg", "/work/py3/airobot/Dist/avatar")
    avatar("/VM/test.wav")
