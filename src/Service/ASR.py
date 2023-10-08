# -*- coding: utf-8 -*-
# python 3.6
"""
asr

Author: chengwen
Modifier:
date:   2023/3/13 上午11:43
Description:  
"""
import paddle
from paddlespeech.cli.text import TextExecutor
from paddlespeech.cli.asr.infer import ASRExecutor
import torchaudio
import soundfile as sf
import noisereduce as nr
from scipy.signal import wiener,stft,istft
import librosa
import numpy as np
from scipy.io import wavfile
import whisper

from src.utils.Common import is_contain_chinese


class ASR():

    def __init__(self,basePath):
        self.asr = whisper.load_model("base")
        self.txt_executor = TextExecutor()
        self.basePath = basePath

    def __call__(self,filePath):
        audio, sr = librosa.load(filePath, sr=None)
        reduced_noise = nr.reduce_noise(y=audio, sr=sr)

        SNR_dB = 20
        SNR_linear = 10 ** (SNR_dB / 10)
        noise_level = np.std(audio) / SNR_linear
        noise = np.random.normal(0, noise_level, len(audio))
        enhanced_audio = reduced_noise - noise
        enhanced_audio = np.clip(enhanced_audio, -32768, 32767)
        sf.write(filePath, enhanced_audio, sr)

        content = self.asr.transcribe(filePath)["text"]
        if len(content)>0 and is_contain_chinese(content):
            return self.txt_executor(
                text = content,
                task = "punc",
                model = "ernie_linear_p7_wudao",
                lang = "zh",
                config = self.basePath+"Dist/ernie_linear_p7_wudao-punc-zh/ckpt/model_config.json",
                ckpt_path = self.basePath+"Dist/ernie_linear_p7_wudao-punc-zh/ckpt/model_state.pdparams",
                punc_vocab = self.basePath+"Dist/ernie_linear_p7_wudao-punc-zh/punc_vocab.txt",
                device=paddle.get_device()
            )
        return ""
