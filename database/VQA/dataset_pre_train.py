import os
import re
import json
import numpy as np
import subprocess
import torch
import math
import cv2
from einops import reduce, repeat, rearrange
from torch.utils.data import DataLoader, Dataset
from PIL import Image


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


class VideoDataset(Dataset):
    r"""
    A Dataset for a folder of videos

    args:
        subj_score_file (str): path to the subjective score file. It contains train/test split, ref list, dis list, fps list and mos list
        directory (str): the path to the directory containing all videos
        mode (str, optional): determines whether to read train/test data
        channel (int, optional): number of channels of a sample
        size_x: horizontal dimension of a segment
        size_y: vertical dimension of a segment
        stride_x: horizontal stride between segments
        stride_y: vertical stride between segments
    """

    def __init__(self, subj_score_file, directory, mode='train', channel=1, size_x=112, size_y=112, stride_x=80, stride_y=80, frameWant=32, transform=None):

        with open(subj_score_file, "r") as f:
            data = json.load(f)
        self.video_dir = directory
        data = data[mode]
        self.ref = data['ref']
        self.dis = data['dis']
        self.label = data['mos']
        self.framerate = data['fps']
        self.frame_height = data['height']
        self.frame_width = data['width']
        self.type = data['type']
        self.video_type = data['video_type']
        self.channel = channel
        self.size_x = size_x
        self.size_y = size_y
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.frameWant = frameWant
        self.transform = transform
        self.mean = 0.458971
        self.std = 0.225609

    def __getitem__(self, index):

        ref = os.path.join(self.video_dir, self.ref[index])
        dis = os.path.join(self.video_dir, self.dis[index])
        label = float(self.label[index])
        dis_type = int(self.type[index])
        framerate = int(self.framerate[index])
        frame_height = int(self.frame_height[index])
        frame_width = int(self.frame_width[index])
        video_type = int(self.video_type[index])

        bytes_per_frame = int(frame_height * frame_width * 1.5)
        frame_count = os.path.getsize(ref) / bytes_per_frame


        if frame_count >= 280:
            stride_t = 10
        elif frame_count >= 220:
            stride_t = 6
        elif frame_count >= 180:
            stride_t = 4
        elif frame_count >= 120:
            stride_t = 2
        else:
            stride_t = 1


        if ref.endswith(('.YUV', '.yuv')):
            ref = self.load_yuv(ref, frame_height, frame_width, stride_t,
                                frameWant=self.frameWant, transform=self.transform)
        elif ref.endswith(('.mp4')):
            ref = self.load_encode(ref, frame_height, frame_width, stride_t,
                                   frameWant=self.frameWant, transform=self.transform)
        else:
            raise ValueError('Unsupported video format')

        if dis.endswith(('.YUV', '.yuv')):
            dis = self.load_yuv(dis, frame_height, frame_width, stride_t,
                                frameWant=self.frameWant, transform=self.transform)
        elif dis.endswith(('.mp4')):
            dis = self.load_encode(dis, frame_height, frame_width, stride_t,
                                   frameWant=self.frameWant, transform=self.transform)
        else:
            raise ValueError('Unsupported video format')

        if self.stride_x and self.stride_y:
            offset_v = (frame_height - self.size_y) % self.stride_y
            offset_t = int(offset_v / 4 * 2)
            offset_b = offset_v - offset_t
            offset_h = (frame_width - self.size_x) % self.stride_x
            offset_l = int(offset_h / 4 * 2)
            offset_r = offset_h - offset_l
            # print(frame_height, frame_width, offset_t, offset_b, offset_l, offset_r)
            ref = ref[:, :, offset_t:frame_height -
                      offset_b, offset_l:frame_width-offset_r]
            dis = dis[:, :, offset_t:frame_height -
                      offset_b, offset_l:frame_width-offset_r]
            spatial_crop = CropSegment(
                self.size_x, self.size_y, self.stride_x, self.stride_y)
            ref = spatial_crop(ref)
            dis = spatial_crop(dis)
            # print(dis.shape)


        label = torch.from_numpy(np.asarray(label))
        return {'ref': ref, 'dis': dis, 'label': label, 'fps': framerate, 'type': dis_type, 'video_type': video_type}


    def load_yuv(self, file_path, frame_height, frame_width, stride_t=0, frameWant=32, start=0, transform=None):
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
        # print(f"file path : {file_path}")

        bytes_per_frame = int(frame_height * frame_width * 1.5)
        frame_count = os.path.getsize(file_path) / bytes_per_frame


        ret = []
        count = 0
        get = 1

        with open(file_path, 'rb') as f:
            while count < frame_count and get <= frameWant:
                # while count < frame_count:
                if count % stride_t == 0 and (frameWant == 0 or get <= frameWant):
                    get += 1
                    offset = count * bytes_per_frame
                    f.seek(offset, 0)
                    frame = f.read(frame_height * frame_width)
                    frame = np.frombuffer(frame, "uint8")
                    frame = frame.astype('float32') / 255.
                    # frame = rearrange(frame, 'd h w c -> d h w c')
                    frame = (frame - self.mean) / self.std
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

    def load_encode(self, file_path, frame_height, frame_width, stride_t, frameWant=32, start=0, transform=None):
        r"""
        Load frames on-demand from encode bitstream

        args:
            file_path (str): path to yuv file
            frame_height
            frame_width
            stride_t (int): sample the 1st frame from every stride_t frames
            start (int): index of the 1st sampled frame
        return:
            ret (array): contains sampled frames. dim = (C, D, H, W)
        """

        enc_path = file_path
        enc_name = re.split('/', enc_path)[-1]

        yuv_name = enc_name.replace('.mp4', '.yuv')
        yuv_path = os.path.join('/dockerdata/tmp/', yuv_name)
        cmd = "ffmpeg -y -i {src} -f rawvideo -pix_fmt yuv420p -vsync 0 -an {dst}".format(
            src=enc_path, dst=yuv_path)
        subprocess.run(cmd, shell=True, stdin=subprocess.PIPE,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ret = self.load_yuv(yuv_path, frame_height,
                            frame_width, stride_t, start=0)

        return ret

    def __len__(self):
        return len(self.dis)


if __name__ == '__main__':

    subj_dataset = './database/VQA/LIVE/live_subj_score_nr_ref.json'
    video_path = '/home1/server823-2/database/2D-Video/live/videos'

    channel = 1
    size_x = 224
    size_y = 224
    stride_x = 224
    stride_y = 224

    video_dataset = VideoDataset(subj_dataset, video_path, 'train', channel,
                                 size_x, size_y, stride_x, stride_y, frameWant=200, transform=None)
    dataloaders = DataLoader(video_dataset, batch_size=2,
                             shuffle=False, num_workers=4, drop_last=False)
    print(len(video_dataset))

    for i, (video, label) in enumerate(dataloaders):
        print(video.shape)
