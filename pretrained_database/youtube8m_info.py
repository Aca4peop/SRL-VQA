import cv2
import os
import json
import time
import math
import logging
import numpy as np
import scipy.ndimage
import skvideo.io
import subprocess
import skvideo.measure
import skvideo.utils
import ffmpeg
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime, timedelta

rootdir = '/home2/youtube8M/Clips/'
disdir = '/home2/youtube8M/Clips/dis/'
resultdir = '/home2/youtube8M/Clips/dis/yuv/'

dis_type = ['h264', 'hevc', 'GaussBlur', 'GaussNoise', 'fps', 'MotionBlur', 'Contrast']
caters = ['Nature', 'Food', 'Guitar', 'Apartment', 'Animal', 'Sport', 'Dance']

DAY = "{0:%Y-%m-%d}".format(datetime.now())


def get_video_info(source_video_path):
    probe = ffmpeg.probe(source_video_path)
    # print('source_video_path: {}'.format(source_video_path))
    format = probe['format']
    bit_rate = int(format['bit_rate'])/1000
    duration = format['duration']
    size = int(format['size'])/1024/1024
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found!')
        return
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])
    fps = int(video_stream['r_frame_rate'].split('/')[0])/int(video_stream['r_frame_rate'].split('/')[1])
    duration = float(video_stream['duration'])
    
    ret = OrderedDict()
    ret['bit_rate'] = bit_rate
    ret['fps'] = fps
    ret['duration'] = duration 
    ret['num_frames'] = num_frames
    ret['width'] = width
    ret['height'] = height
    # print(ret['fps'])
    return ret


def getlogger(path=None, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    rq = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    log_path = os.path.join(resultdir, path, DAY)
    if not os.path.exists(log_path):
        print("log_path don't existed...")
        os.makedirs(log_path)
    log_name = os.path.join(log_path, rq + '.log')
    logfile = log_name

    print("log file: {}".format(logfile))

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def write_info(info, ref_path, dis_path, dis, cater, logger):
    info['video_type'].append(cater)

    ret_dis = get_video_info(dis_path)
    ret_ref = get_video_info(ref_path)

    try:
        ret_dis = get_video_info(dis_path)
        ret_ref = get_video_info(ref_path)

        if math.ceil(float(ret_ref['fps']) >= 23.0):
            info['ref'].append(ref_path)
            info['dis'].append(dis_path)
            info['width'].append(ret_ref['width'])
            info['height'].append(ret_ref['height'])            
            info['dis_type'].append(dis)
            for k, v in ret_dis.items():
                if k != 'width' and k != 'height':
                    info['dis_' + k].append(v) 
            for k, v in ret_ref.items():
                if k != 'width' and k != 'height':
                    info['ref_' + k].append(v)
        else:
            logger.error(f"ref_path : {ref_path}\t ref_fps : {ret_ref['fps']}")

    except Exception as e:
        logger.error(f"ref_path : {ref_path}\t dis_path : {dis_path}\t error : {str(e)}")


def to_yuv(src, dst):
    dst = dst.replace('.mp4', '.yuv')
    cmd = f"ffmpeg -y -i {src} -f rawvideo -pix_fmt yuv420p -vsync 0 -an {dst}"
    subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def dump_json():

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
        print(f"make dir : {resultdir} success! ")

    logger = getlogger(path='error_info')

    info = OrderedDict()
    info['ref'] = []
    info['dis'] = []
    info['width'] = []
    info['height'] = []
    info['video_type'] = []
    
    info['dis_type'] = []
    info['ref_duration'] = []
    info['ref_bit_rate'] = []
    info['ref_num_frames'] = []
    info['ref_fps'] = []
    info['dis_duration'] = []
    info['dis_bit_rate'] = []
    info['dis_num_frames'] = []
    info['dis_fps'] = []
    
    # t_files = []
    # for file in files:
    #     if file.endswith('.mp4'):
    #         t_files.append(file)
    # files = t_files
    # length = len(files)
    # randomIdx = np.random.permutation(np.arange(length))
    # split = int(np.floor(0.5 * length))
    # alls = randomIdx[:split]

    for cater in caters:
        cater_path = os.path.join(rootdir, cater)
        files = os.listdir(cater_path)

        for idx, file in enumerate(tqdm(files)):
            if file.endswith('.mp4'):
                suffix = file.split('/')[-1]
                ref_path = os.path.join(rootdir, cater, suffix)
                # yuv_path = os.path.join(resultdir, suffix)
                # to_yuv(ref_path, yuv_path)

                for dis in dis_type:
                    if dis != 'fps':
                        for d in range(1, 4):
                            dis_path = os.path.join(disdir, dis, str(d) + '_' + suffix)
                            write_info(info, ref_path, dis_path, dis, cater, logger)
                    else:
                        for d in [15, 24]:
                            dis_path = os.path.join(disdir, dis, str(d) + '_' + suffix)
                            write_info(info, ref_path, dis_path, dis, cater, logger)

        logger.info(f"index {idx} success ~")

    with open(os.path.join(resultdir, 'youtube8M_info_bits_type(contrast_motionblur)_fps_yuv_type.json'), 'w') as f:
        json.dump(info, f, indent=4, sort_keys=True)

    # print(f"length of files : {len(files)}")
    print(f"length of keys : {len(info['ref'])}")
    print(f"length of keys : {len(info['dis'])}")
    print(f"length of keys : {len(info['dis_fps'])}")
    print(f"length of keys : {len(info['ref_fps'])}")
    print(f"length of keys : {len(info['video_type'])}")


if __name__ == "__main__":
    dump_json()
    # test()
