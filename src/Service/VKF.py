# -*- coding: utf-8 -*-
# python 3.6
"""
VKF

Author: chengwen
Modifier:
date:   2022/7/12 上午10:41
Description:
"""
import copy
import importlib
import os
import shutil

from PIL import Image
import cv2
import numpy as np
import time
from paddle.inference import Config
from paddle.inference import create_predictor
# import paddleclas
# model = paddleclas.PaddleClas(model_name="person_exists")
#
from functools import partial
import yaml


class VKF:

    def __init__(self,url,processFun,interval=2,min_area = 30*30,max_area=600*600,min_ratio = 0.2,max_ratio = 2.):
        """
        :param
            *_area,*_ratio: Candidate box control
            processFun: Processing function
            url: Video path/Camera rtsp
            interval: Sampling interval
        :return:
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.vibe = self._ViBe()
        self.url = url
        self.processFun = processFun
        self.interval = interval
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            rval, tmp = self.cap.read()
        else:
            rval = False

        tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2GRAY)

        cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
        cv2.namedWindow("source",cv2.WINDOW_NORMAL)
        self.height = int(tmp.shape[0]/2)
        self.width = int(tmp.shape[1]/2)
        tmp = cv2.resize(tmp,(self.width,self.height))
        self.vibe.ProcessFirstFrame(tmp)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        count = 0

        while rval:
            rval, curr_frame = self.cap.read()
            if curr_frame is None:
                continue
            count+=1
            if count % self.interval==0:
                resize_frame = cv2.resize(curr_frame,(self.width,self.height))
                gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)
                self.vibe.Update(gray)
                segMat=self.vibe.getFGMask()
                segMat = segMat.astype(np.uint8)
                img = cv2.morphologyEx(segMat, cv2.MORPH_OPEN, kernel)
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
                #search contours
                contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                bboxs = vkf.select_roi(contours)
                self.processFun(curr_frame,resize_frame,bboxs)
                # print("count {},boxs {}".format(count,len(bboxs)))
            if count>300000:
                count = 0
        print(count)
    def stop(self):
        if self.cap is not None:
            self.cap.release()

    class _ViBe:
        '''
        ViBe运动检测，分割背景和前景运动图像
        '''
        def __init__(self,num_sam=20,min_match=2,radiu=20,rand_sam=16):
            self.defaultNbSamples = num_sam
            self.defaultReqMatches = min_match
            self.defaultRadius = radiu
            self.defaultSubsamplingFactor = rand_sam
            self.background = 0
            self.foreground = 255

        def __buildNeighborArray(self,img):
            '''
            构建一副图像中每个像素的邻域数组
            参数：输入灰度图像
            返回值：每个像素9邻域数组，保存到self.samples中
            '''
            height,width=img.shape
            self.samples=np.zeros((self.defaultNbSamples,height,width),dtype=np.uint8)

            ramoff_xy=np.random.randint(-1,2,size=(2,self.defaultNbSamples,height,width))

            xr_=np.tile(np.arange(width),(height,1))
            yr_=np.tile(np.arange(height),(width,1)).T

            xyr_=np.zeros((2,self.defaultNbSamples,height,width))
            for i in range(self.defaultNbSamples):
                xyr_[1,i]=xr_
                xyr_[0,i]=yr_

            xyr_=xyr_+ramoff_xy

            xyr_[xyr_<0]=0
            tpr_=xyr_[1,:,:,-1]
            tpr_[tpr_>=width]=width-1
            tpb_=xyr_[0,:,-1,:]
            tpb_[tpb_>=height]=height-1
            xyr_[0,:,-1,:]=tpb_
            xyr_[1,:,:,-1]=tpr_

            xyr=xyr_.astype(int)
            self.samples=img[xyr[0,:,:,:],xyr[1,:,:,:]]


        def ProcessFirstFrame(self,img):
            '''
            处理视频的第一帧
            1、初始化每个像素的样本集矩阵
            2、初始化前景矩阵的mask
            3、初始化前景像素的检测次数矩阵
            参数：
            img: 传入的numpy图像素组，要求灰度图像
            返回值：
            每个像素的样本集numpy数组
            '''
            self.__buildNeighborArray(img)
            self.fgCount=np.zeros(img.shape)
            self.fgMask=np.zeros(img.shape)

        def Update(self,img):
            '''
            处理每帧视频，更新运动前景，并更新样本集。该函数是本类的主函数
            输入：灰度图像
            '''
            height,width=img.shape
            dist=np.abs((self.samples.astype(float)-img.astype(float)).astype(int))
            dist[dist<self.defaultRadius]=1
            dist[dist>=self.defaultRadius]=0
            matches=np.sum(dist,axis=0)
            matches=matches<self.defaultReqMatches
            self.fgMask[matches]=self.foreground
            self.fgMask[~matches]=self.background
            self.fgCount[matches]=self.fgCount[matches]+1
            self.fgCount[~matches]=0
            fakeFG=self.fgCount>50
            matches[fakeFG]=False

            upfactor=np.random.randint(self.defaultSubsamplingFactor,size=img.shape)
            upfactor[matches]=100
            upSelfSamplesInd=np.where(upfactor==0)
            upSelfSamplesPosition=np.random.randint(self.defaultNbSamples,size=upSelfSamplesInd[0].shape)  #生成随机更新自己样本集的的索引
            samInd=(upSelfSamplesPosition,upSelfSamplesInd[0],upSelfSamplesInd[1])
            self.samples[samInd]=img[upSelfSamplesInd]
            upfactor=np.random.randint(self.defaultSubsamplingFactor,size=img.shape)
            upfactor[matches]=100
            upNbSamplesInd=np.where(upfactor==0)
            nbnums=upNbSamplesInd[0].shape[0]
            ramNbOffset=np.random.randint(-1,2,size=(2,nbnums))
            nbXY=np.stack(upNbSamplesInd)
            nbXY+=ramNbOffset
            nbXY[nbXY<0]=0
            nbXY[0,nbXY[0,:]>=height]=height-1
            nbXY[1,nbXY[1,:]>=width]=width-1
            nbSPos=np.random.randint(self.defaultNbSamples,size=nbnums)
            nbSamInd=(nbSPos,nbXY[0],nbXY[1])
            self.samples[nbSamInd]=img[upNbSamplesInd]

        def getFGMask(self):
            '''
            返回前景mask
            '''
            return self.fgMask

    def compute_iou(self,rec1, rec2):
        """
        computing IoU
        :param rec1: (y0, x0, y1, x1), which reflects
                (top, left, bottom, right)
        :param rec2: (y0, x0, y1, x1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            IOU = []
            IOU.append((intersect / (sum_area - intersect))*1.0)
            IOU.append((intersect / S_rec1)*1.0)
            IOU.append((intersect / S_rec2)*1.0)
            return np.max(IOU)

    def inclusion_check(self,rec1, rec2):
        selectA = False
        if rec1[0]<=rec2[0] and rec1[1]<=rec2[1]:
            selectA = True
        if rec1[2]>=rec2[2] and rec1[3]>=rec2[3]:
            if selectA:
                return True
        else:
            if not selectA:
                return True
        return False

    def combine_roi(self,rec1, rec2):
        x1 = rec1[0] if rec1[0]<rec2[0] else rec2[0]
        y1 = rec1[1] if rec1[1]<rec2[1] else rec2[1]
        x2 = rec1[2] if rec1[2]>rec2[3] else rec2[3]
        y2 = rec1[3] if rec1[3]>rec2[3] else rec2[3]
        return [x1,y1,x2,y2]

    def select_roi(self,contours):
        """
        The main body entry function finds
        the region of interest from the candidate region
        :param contours: candidate region (x,y,w,h)
        :return: the region of interest
        """
        res = []
        bboxs = []
        final = []
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            bboxs.append([x, y,x+w, y+h])

        if len(bboxs)>0:
            remove = []
            for i,sbox in enumerate(bboxs):
                if i in remove:
                    continue
                for j,cbox in enumerate(bboxs):
                    if i!=j and j not in remove:
                        val = self.compute_iou(sbox,cbox)
                        if val>0.6:
                            remove.append(i)
                            remove.append(j)
                            res.append(self.combine_roi(sbox,cbox))
                            break
            for k,cbox in enumerate(bboxs):
                if k not in remove:
                    res.append(cbox)

            #inclusion combine
            while(True):
                remove = []
                out = []
                for i,sbox in enumerate(res):
                    if i in remove:
                        continue
                    for j,cbox in enumerate(res):
                        if i!=j and j not in remove:
                            if self.inclusion_check(sbox,cbox):
                                remove.append(i)
                                remove.append(j)
                                out.append(self.combine_roi(sbox,cbox))
                                break
                for k,cbox in enumerate(res):
                    if k not in remove:
                        out.append(cbox)
                if len(res)==len(out):
                    break
                else:
                    res = out

            for box in res:
                w = box[2]-box[0]
                h = box[3]-box[1]
                if w * h > self.min_area and w * h < self.max_area and w/(h+0.)>self.min_ratio and w/(h+0.)<self.max_ratio:
                    final.append(box)
        return final

sift = cv2.SIFT_create()
pre_keypoints = None
pre_descriptors = None

bf = cv2.BFMatcher()
def calculateMatches(des1,des2):
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])

    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    try:
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                topResults2.append([m])
    except:
        pass
    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

class Predictor(object):
    def __init__(self, args, inference_model_dir=None):
        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_tensorrt is True
        self.args = args
        if self.args.get("use_onnx", False):
            self.predictor, self.config = self.create_onnx_predictor(
                args, inference_model_dir)
        else:
            self.predictor, self.config = self.create_paddle_predictor(
                args, inference_model_dir)

    def predict(self, image):
        raise NotImplementedError

    def create_paddle_predictor(self, args, inference_model_dir=None):
        if inference_model_dir is None:
            inference_model_dir = args.inference_model_dir
        if "inference_int8.pdiparams" in os.listdir(inference_model_dir):
            params_file = os.path.join(inference_model_dir,
                                       "inference_int8.pdiparams")
            model_file = os.path.join(inference_model_dir,
                                      "inference_int8.pdmodel")
            assert args.get(
                "use_fp16", False
            ) is False, "fp16 mode is not supported for int8 model inference, please set use_fp16 as False during inference."
        else:
            params_file = os.path.join(inference_model_dir,
                                       "inference.pdiparams")
            model_file = os.path.join(inference_model_dir, "inference.pdmodel")
            assert args.get(
                "use_int8", False
            ) is False, "int8 mode is not supported for fp32 model inference, please set use_int8 as False during inference."

        config = Config(model_file, params_file)

        if args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, 0)
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # there is no set_mkldnn_cache_capatity() on macOS
                if platform.system() != "Darwin":
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(args.cpu_num_threads)

        if args.enable_profile:
            config.enable_profile()
        config.disable_glog_info()
        config.switch_ir_optim(args.ir_optim)  # default true
        if args.use_tensorrt:
            precision = Config.Precision.Float32
            if args.get("use_int8", False):
                precision = Config.Precision.Int8
            elif args.get("use_fp16", False):
                precision = Config.Precision.Half

            config.enable_tensorrt_engine(
                precision_mode=precision,
                max_batch_size=args.batch_size,
                workspace_size=1 << 30,
                min_subgraph_size=30,
                use_calib_mode=False)

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        return predictor, config

    def create_onnx_predictor(self, args, inference_model_dir=None):
        import onnxruntime as ort
        if inference_model_dir is None:
            inference_model_dir = args.inference_model_dir
        model_file = os.path.join(inference_model_dir, "inference.onnx")
        config = ort.SessionOptions()
        if args.use_gpu:
            raise ValueError(
                "onnx inference now only supports cpu! please specify use_gpu false."
            )
        else:
            config.intra_op_num_threads = args.cpu_num_threads
            if args.ir_optim:
                config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        predictor = ort.InferenceSession(model_file, sess_options=config)
        return predictor, config



class ResizeImage(object):
    """ resize image """

    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2"):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2"):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        _pil_interp_from_str = {
            'nearest': Image.Resampling.NEAREST,
            'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC,
            'box': Image.Resampling.BOX,
            'lanczos': Image.Resampling.LANCZOS,
            'hamming': Image.Resampling.HAMMING
        }

        def _pil_resize(src, size, resample):
            pil_img = Image.fromarray(src)
            pil_img = pil_img.resize(size, resample)
            return np.asarray(pil_img)

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(cv2.resize, interpolation=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(_pil_resize, resample=interpolation)
        else:

            self.resize_func = cv2.resize

    def __call__(self, src, size):
        return self.resize_func(src, size)


def create_operators(params):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    mod = importlib.import_module(__name__)
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(mod, op_name)(**param)
        ops.append(op)

    return ops


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                (img, pad_zeros), axis=2))
        return img.astype(self.output_dtype)


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))

class CropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]

        if img_h < h or img_w < w:
            raise Exception(
                f"The size({h}, {w}) of CropImage must be greater than size({img_h}, {img_w}) of image. Please check image original size and size of ResizeImage if used."
            )

        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


def build_postprocess(config):
    if config is None:
        return None

    mod = importlib.import_module(__name__)
    config = copy.deepcopy(config)

    main_indicator = config.pop(
        "main_indicator") if "main_indicator" in config else None
    main_indicator = main_indicator if main_indicator else ""

    func_list = []
    for func in config:
        func_list.append(getattr(mod, func)(**config[func]))
    return PostProcesser(func_list, main_indicator)


class SavePreLabel(object):
    def __init__(self, save_dir):
        if save_dir is None:
            raise Exception(
                "Please specify save_dir if SavePreLabel specified.")
        self.save_dir = partial(os.path.join, save_dir)

    def __call__(self, x, file_names=None):
        if file_names is None:
            return
        assert x.shape[0] == len(file_names)
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-1].astype("int32")
            self.save(index, file_names[idx])

    def save(self, id, image_file):
        output_dir = self.save_dir(str(id))
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(image_file, output_dir)


class ThreshOutput(object):
    def __init__(self, threshold, label_0="0", label_1="1"):
        self.threshold = threshold
        self.label_0 = label_0
        self.label_1 = label_1

    def __call__(self, x, file_names=None):
        y = []
        for idx, probs in enumerate(x):
            score = probs[1]
            if score < self.threshold:
                result = {
                    "class_ids": [0],
                    "scores": [1 - score],
                    "label_names": [self.label_0]
                }
            else:
                result = {
                    "class_ids": [1],
                    "scores": [score],
                    "label_names": [self.label_1]
                }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y


class PostProcesser(object):
    def __init__(self, func_list, main_indicator="Topk"):
        self.func_list = func_list
        self.main_indicator = main_indicator

    def __call__(self, x, image_file=None):
        rtn = None
        for func in self.func_list:
            tmp = func(x, image_file)
            if type(func).__name__ in self.main_indicator:
                rtn = tmp
        return rtn

class ClsPredictor(Predictor):
    def __init__(self, config):
        super().__init__(config["Global"])

        self.preprocess_ops = []
        self.postprocess = None
        if "PreProcess" in config:
            if "transform_ops" in config["PreProcess"]:
                self.preprocess_ops = create_operators(config["PreProcess"][
                                                           "transform_ops"])
        if "PostProcess" in config:
            self.postprocess = build_postprocess(config["PostProcess"])

        # for whole_chain project to test each repo of paddle
        self.benchmark = config["Global"].get("benchmark", False)
        if self.benchmark:
            import auto_log
            import os
            pid = os.getpid()
            size = config["PreProcess"]["transform_ops"][1]["CropImage"][
                "size"]
            if config["Global"].get("use_int8", False):
                precision = "int8"
            elif config["Global"].get("use_fp16", False):
                precision = "fp16"
            else:
                precision = "fp32"
            self.auto_logger = auto_log.AutoLogger(
                model_name=config["Global"].get("model_name", "cls"),
                model_precision=precision,
                batch_size=config["Global"].get("batch_size", 1),
                data_shape=[3, size, size],
                save_path=config["Global"].get("save_log_path",
                                               "./auto_log.log"),
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=None,
                time_keys=[
                    'preprocess_time', 'inference_time', 'postprocess_time'
                ],
                warmup=2)

    def predict(self, images):
        use_onnx = self.args.get("use_onnx", False)
        if not use_onnx:
            input_names = self.predictor.get_input_names()
            input_tensor = self.predictor.get_input_handle(input_names[0])

            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
        else:
            input_names = self.predictor.get_inputs()[0].name
            output_names = self.predictor.get_outputs()[0].name

        if self.benchmark:
            self.auto_logger.times.start()
        if not isinstance(images, (list,)):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)
        if self.benchmark:
            self.auto_logger.times.stamp()

        if not use_onnx:
            input_tensor.copy_from_cpu(image)
            self.predictor.run()
            batch_output = output_tensor.copy_to_cpu()
        else:
            batch_output = self.predictor.run(
                output_names=[output_names],
                input_feed={input_names: image})[0]

        if self.benchmark:
            self.auto_logger.times.stamp()
        if self.postprocess is not None:
            batch_output = self.postprocess(batch_output)
        if self.benchmark:
            self.auto_logger.times.end(stamp=True)
        return batch_output

class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return copy.deepcopy(dict(self))


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config

car_config = parse_config("./inference_car_exists.yaml")
predictor = ClsPredictor(car_config)
count = 0
predictnum = 0
def process(curr_frame,resize_frame,bboxs):
    containObj = False
    global pre_keypoints
    global pre_descriptors
    global predictor
    global count
    global predictnum
    if len(bboxs)>0:
        predictnum+=1
    for bbox in bboxs:
        res = predictor.predict(resize_frame[bbox[1]:bbox[3],bbox[0]:bbox[2]])
        if res[0]['class_ids'][0] ==1:
            containObj = True
            # cur_keypoints,cur_descriptors=sift.detectAndCompute(curr_frame[bbox[1]:bbox[3],bbox[0]:bbox[2]], None)
        cv2.rectangle(resize_frame, (bbox[0],bbox[1]), (bbox[2], bbox[3]),(0, 0, 255), 2)
    if containObj:

        if pre_keypoints is not None:
            matches = calculateMatches(pre_descriptors, cur_descriptors)
            score = calculateScore(len(matches),len(pre_keypoints),len(cur_keypoints))
            if score<10:
                count+=1
                print("count {},boxs {}".format(count,len(bboxs)))
                # pre_keypoints = cur_keypoints
                # pre_descriptors = cur_descriptors
                cv2.imshow("frame", resize_frame)
        else:
            count+=1
            # pre_keypoints = cur_keypoints
            # pre_descriptors = cur_descriptors
            cv2.imshow("frame", resize_frame)
    cv2.imshow("source", resize_frame)

    k = cv2.waitKey(10)
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()

vkf = VKF('/VM/car.mp4',process,interval=2)
start = time.time()
vkf.start()
print(time.time()-start)
print(predictnum)
