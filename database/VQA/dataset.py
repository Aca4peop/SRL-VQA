import os
import re
import json
import numpy as np
import subprocess
import torch
import cv2
import torch.backends.cudnn as cudnn
import torch.nn as nn
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
        self.mode = mode
        self.video_dir = directory
        data = data[mode]
        # self.ref = data['ref']
        self.dis = data['dis']
        self.label = data['mos']
        self.framerate = data['fps']
        self.frame_height = data['height']
        self.frame_width = data['width']
        self.channel = channel
        self.size_x = size_x
        self.size_y = size_y
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.frameWant = frameWant
        self.transform = transform

    def __getitem__(self, index):

        # ref = os.path.join(self.video_dir, self.ref[index])
        dis = os.path.join(self.video_dir, self.dis[index])
        label = float(self.label[index])
        framerate = int(self.framerate[index])
        frame_height = int(self.frame_height[index])
        frame_width = int(self.frame_width[index])

        if framerate <= 30:
            stride_t = 2
        elif framerate <= 60:
            stride_t = 4
        else:
            raise ValueError('Unsupported fps')

        # todo
        # stride_t = 1

        # if ref.endswith(('.YUV', '.yuv')):
        #     ref = self.load_yuv(ref, frame_height, frame_width, stride_t)
        # elif ref.endswith(('.mp4')):
        #     ref = self.load_encode(ref, frame_height, frame_width, stride_t)
        # else:
        #     raise ValueError('Unsupported video format')

        if dis.endswith(('.YUV', '.yuv')):
            dis = self.load_yuv(dis, frame_height, frame_width, stride_t, frameWant=self.frameWant, transform=self.transform)
        elif dis.endswith(('.mp4')):
            dis = self.load_encode(dis, frame_height, frame_width, stride_t)
        else:
            raise ValueError('Unsupported video format')

        offset_v = (frame_height - self.size_y) % self.stride_y
        offset_t = int(offset_v / 4 * 2)
        offset_b = offset_v - offset_t
        offset_h = (frame_width - self.size_x) % self.stride_x
        offset_l = int(offset_h / 4 * 2)
        offset_r = offset_h - offset_l

        # ref = ref[:, :, offset_t:frame_height-offset_b, offset_l:frame_width-offset_r]
        # dis = dis[:, :, offset_t:frame_height-offset_b, offset_l:frame_width-offset_r]
        #
        # spatial_crop = CropSegment(self.size_x, self.size_y, self.stride_x, self.stride_y)
        # # ref = spatial_crop(ref)
        # dis = spatial_crop(dis)

        # ref = torch.from_numpy(np.asarray(ref))
        img = torch.zeros((self.frameWant, 3, 224, 224))
        dis = np.transpose(dis, (0, 2, 3, 1))
        for i in range(dis.shape[0]):
            img[i] = self.transform(Image.fromarray(torch.mul(dis[i], 255).numpy().astype('uint8'), mode='RGB'))
        del dis
        # dis = torch.from_numpy(np.asarray(dis))
        label = torch.from_numpy(np.asarray(label))
        # return ref, dis, label
        return img, label

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

        bytes_per_frame = int(frame_height * frame_width * 1.5)
        frame_count = os.path.getsize(file_path) / bytes_per_frame

        # TODO frame Want
        stride_t = int((frame_count + frameWant - 1) / frameWant)

        ret = []
        count = 0

        with open(file_path, 'rb') as f:
            while count < frame_count:
                if count % stride_t == 0:
                    offset = count * bytes_per_frame
                    f.seek(offset, 0)
                    frame = f.read(int(frame_height * frame_width * 1.5))
                    yuv = read_YUV420(frame, frame_height, frame_width)
                    frame = merge_YUV2RGB_v1(*yuv)
                    # frame = np.frombuffer(frame, "uint8")
                    frame = frame.astype('float32') / 255.
                    frame = np.expand_dims(frame, axis=0)
                    # frame = rearrange(frame, 'd h w c -> d h w c')
                    # frame = frame.reshape(1, 3, frame_height, frame_width)
                    ret.append(frame)
                count += 1

        ret = np.concatenate(ret, axis=0)
        ret = torch.from_numpy(np.asarray(ret))

        return ret

    def load_encode(self, file_path, frame_height, frame_width, stride_t, start=0):
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
        cmd = "ffmpeg -y -i {src} -f rawvideo -pix_fmt yuv420p -vsync 0 -an {dst}".format(src=enc_path, dst=yuv_path)
        subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ret = self.load_yuv(yuv_path, frame_height, frame_width, stride_t, start=0)

        return ret

    def __len__(self):
        return len(self.dis)




def read_YUV420(image, rows, cols):
    # create Y
    gray = np.zeros((rows, cols), np.uint8)
    # print(type(gray))
    # print(gray.shape)

    # create U,V
    img_U = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    img_V = np.zeros((int(rows / 2), int(cols / 2)), np.uint8)
    # print(type(image))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            gray[i, j] = image[idx]
            idx += 1

    for i in range(int(rows / 2)):
        for j in range(int(cols / 2)):
            img_U[i, j] = image[idx]
            idx += 1

    for i in range(int(rows / 2)):
        for j in range(int(cols / 2)):
            img_V[i, j] = image[idx]
            idx += 1

    return [gray, img_U, img_V]


def merge_YUV2RGB_v1(Y, U, V):
    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])

    rgb = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return rgb


def merge_YUV2RGB_v2(Y, U, V):
    rows, cols = Y.shape[:2]

    shrink_Y = cv2.resize(Y, (cols / 2, rows / 2), interpolation=cv2.INTER_AREA)

    img_YUV = cv2.merge([shrink_Y, U, V])

    dst = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    cv2.COLOR_YUV2BGR_I420

    enlarge_dst = cv2.resize(dst, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return enlarge_dst


if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.realpath(__file__))
    subj_score_file = os.path.join(root_dir, 'csiq_subj_score.json')
    video_dir = '/dockerdata/CSIQ_YUV'
    csiq_dataset = VideoDataset(subj_score_file, video_dir)
    print(len(csiq_dataset))
