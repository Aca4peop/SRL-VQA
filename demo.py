import argparse
import json
import math
import os
import re
import subprocess

import cv2
import numpy as np
import skvideo.io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from einops import rearrange, reduce, repeat
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from timm.utils import AverageMeter
from torch.utils.data import DataLoader, Dataset

from config import get_config
from models.build import build_model


# --------------------------------------------------------
# DVQA
# Code Link: https://github.com/Tencent/DVQA
# --------------------------------------------------------'
class CropSegment(object):
    r"""
    Crop a clip along the spatial axes, i.e. h, w
    DO NOT crop along the temporal axis

    args:
        size_x: horizontal dimension of a segment
        size_y: vertical dimension of a segment
        stride_x: horizontal stride between segments
        stride_y: vertical stride between segments
    return:
        clip (tensor): dim = (N, C, D, H=size_y, W=size_x). N are segments number by applying sliding window with given window size and stride
    """

    def __init__(self, size_x, size_y, stride_x, stride_y):

        self.size_x = size_x
        self.size_y = size_y
        self.stride_x = stride_x
        self.stride_y = stride_y

    def __call__(self, clip):
        # input dimension [C, D, H, W]
        channel = clip.shape[0]
        depth = clip.shape[1]

        clip = clip.unfold(2, self.size_x, self.stride_x)
        clip = clip.unfold(3, self.size_y, self.stride_y)
        clip = clip.permute(2, 3, 0, 1, 4, 5)
        clip = clip.contiguous().view(-1, channel, depth, self.size_x, self.size_y)

        return clip


def load_yuv(file_path, frame_height, frame_width, stride_t=0, frameWant=32, start=0, transform=None):
    r"""
    Load frames on-demand from raw video, currently supports only yuv420p

    args:
        file_path (str): path to yuv file
        frame_height
        frame_width
        stride_t (int): sample the 1st frame from every stride_t frames
        start (int): index of the 1st sampled frame
    return:
        ret (tensor): contains sampled frames (Y channel). dim = (C, D, H, W)
    """
    mean = 0.458971
    std = 0.225609

    bytes_per_frame = int(frame_height * frame_width * 1.5)
    frame_count = os.path.getsize(file_path) / bytes_per_frame

    if frameWant != 0:
        stride_t = math.ceil(frame_count / frameWant) - 1
    else:
        stride_t = 1

    ret = []
    count = 0
    get = 1

    with open(file_path, 'rb') as f:
        while count < frame_count:
            if count % stride_t == 0 and (frameWant == 0 or get <= frameWant):
                get += 1
                offset = count * bytes_per_frame
                f.seek(offset, 0)
                frame = f.read(frame_height * frame_width)
                frame = np.frombuffer(frame, "uint8")
                frame = frame.astype('float32') / 255.
                # frame = rearrange(frame, 'd h w c -> d h w c')
                frame = (frame - mean) / std
                frame = frame.reshape(1, 1, frame_height, frame_width)
                ## TODO : transforms
                if transform is not None:
                    img = torch.from_numpy(frame).squeeze(0)
                    img = transform(img)
                    img = img.unsqueeze(0)
                    frame = img.numpy()
                ret.append(frame)
            count += 1

    ret = np.concatenate(ret, axis=1)
    ret = torch.from_numpy(np.asarray(ret))

    return ret


def load_mp4(file_path, frame_height, frame_width, stride_t=0, frameWant=32, start=0, transform=None):

    mean = 0.458971
    std = 0.225609

    # just load the luminance channel of the input video
    video = skvideo.io.vread(file_path,  as_grey=True)

    ret = []
    get = 1
    frameNum = video.shape[0]

    if frameWant != 0:
        stride_t = math.ceil(frameNum / frameWant) - 1
    else:
        stride_t = 1

    for i in range(frameNum):
        if i % stride_t == 0 and (frameWant == 0 or get <= frameWant):
            get += 1
            frame = video[i].astype('float32') / 255.
            frame = rearrange(frame, 'h w c -> 1 c h w')
            frame = (frame - mean) / std
            ## TODO : transforms
            if transform is not None:
                img = torch.from_numpy(frame).squeeze(0)
                img = transform(img)
                img = img.unsqueeze(0)
                frame = img.numpy()
            ret.append(frame)
    ret = np.concatenate(ret, axis=1)
    ret = torch.from_numpy(np.asarray(ret))
    return ret


def load_video(config, transform=None):

    if config.video_path.endswith(('.YUV', '.yuv')):
        video = load_yuv(config.video_path, config.frame_height, config.frame_width,
                         frameWant=config.frameWant, transform=transform)
    else:
        video = load_mp4(config.video_path, config.frame_height, config.frame_width,
                         frameWant=config.frameWant, transform=transform)

    if min(config.frame_height, config.frame_width) <= 400:
        stride_x, stride_y = 224, 224

    if config.stride_x and config.stride_y:
        spatial_crop = CropSegment(config.crop_size_x, config.crop_size_y, config.stride_x, config.stride_y)
        video = spatial_crop(video)

    return video


def load_model(config):
    print("Just load model, WithOut Pre-Trained Weight")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model)
    model.to(device)
    return model, device

def load_pre_trained(config, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model)

    # new model dict
    model_dict = model.state_dict()
    # load pre trained model
    if config.model != 'pre_train':
        pre_dict = torch.load(path, device)
        pretrained_dict = {k: v for k,
                           v in pre_dict.items() if k in model_dict}
        print(f"Model Type : {config.model}\t Length Of Pre trained model : {len(pretrained_dict)}")
        model_dict.update(pretrained_dict)
    else:
        pretrained_model = torch.load(path, device)['model']
        if 'head.weight' in pretrained_model:
            pretrained_model.pop('head.weight')
        if 'head.bias' in pretrained_model:
            pretrained_model.pop('head.bias')
        # get the same weight
        # pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if 'backbone.' + k in model_dict}
        # overwrite the new model dict
        model_dict.update(pretrained_dict)
        # update the dict to new model
        print(f"Model Type : {config.MODEL.TYPE}\t Length Of Pre trained model : {len(pretrained_dict)}")
        print(f"length of pretrained dict : {len(pretrained_dict)}")

    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    return model, device


def predict(config):

    if config.fine_tune == False:
        VQAModel, device = load_model(config)
    else:
        prefix = os.path.abspath('.')
        VQAModel, device = load_pre_trained(config, config.pre_trained_path)

    video = load_video(config)

    N, C, D, H, W = video.size()
    video = video.to(device)
    pred = VQAModel(video)
    pred = torch.mean(pred)
    print(f"Predicted Score = {pred}")


def parse_option():
    parser = argparse.ArgumentParser(
        'Demo for Self-Supervised Representation Learning for Video Quality Assessment', add_help=False)

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--video_path', type=str, default='./test_video/2999049224_original_centercrop_960x540_8s.mp4',
                        help="path for test video")
    parser.add_argument('--frame_width', type=int, default=960,
                        help="frame width for test video")
    parser.add_argument('--frame_height', type=int, default=540,
                        help="frame height for test video")
    parser.add_argument('--stride_x', type=int, default=224,
                        help='stride size_x for test video')
    parser.add_argument('--stride_y', type=int, default=224,
                        help='stride size_y for test video')
    parser.add_argument('--crop_size_x', type=int, default=224,
                        help='crop size_x for test video')
    parser.add_argument('--crop_size_y', type=int, default=224,
                        help='crop size_y for test video')
    parser.add_argument('--model', type=str,
                        default='vb_cnn_transformer', help='Model Type')
    parser.add_argument('--frameWant', type=int, default=16,
                        help='test frame for test video')

    parser.add_argument('--pre_trained_path', type=str,
                        default='./pretrained/pre_trained.pth', help='pretrained weight path')
    parser.add_argument('--fine_tune', type=bool, help='Fine Tune Or Not')

    config = parser.parse_args()

    return config



if __name__ == '__main__':

    config = parse_option()
    predict(config)

