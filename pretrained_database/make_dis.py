import json
import logging
import os
import random
import subprocess
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np
import skvideo.io
from imgaug import augmenters as iaa
from tqdm import tqdm

DAY = "{0:%Y-%m-%d}".format(datetime.now())

# rootdir: 310 videos path
rootdir = '/home2/youtube8M/Clips/'

# result dir: mp4
resultdir = '/home2/youtube8M/Clips/dis/'

# result dir: yuv
yuvdir = '/home2/youtube8M/Clips/dis/yuv/'

caters = ['Nature', 'Food', 'Guitar', 'Apartment', 'Animal', 'Sport', 'Dance']
compress_type = ['hevc', 'h264']


def getlogger(path='logs', level=logging.INFO):
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


def to_yuv(src, dst):
    dst = dst.replace('.mp4', '.yuv')
    cmd = f"/usr/bin/ffmpeg -y -i {src} -f rawvideo -pix_fmt yuv420p -vsync 0 -an {dst}"
    subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def getJsonString(strFileName):
    strCmd = 'ffprobe -v quiet -print_format json -show_format -show_streams -i "' + \
        strFileName + '"'
    mystring = os.popen(strCmd).read()
    return mystring


def get_btrts(filename):
    filecontent = getJsonString(filename)
    try:
        js = json.loads(filecontent)
        btrt = int(js['streams'][0]['bit_rate'])//1000
        btrts = [2 * btrt//3, btrt // 2, btrt // 4]
        return btrts

    except Exception as e:
        print(e)
        return []


def compress(ref_path, btrts):
    filename = ref_path.split('/')[-1]

    if btrts:
        for code in compress_type:
            for path in [resultdir, yuvdir]:
                checkDir(os.path.join(path, code))

            level = 1
            for btrt in btrts:
                dis_path = os.path.join(resultdir, code, str(level) + '_' + filename)
                yuv_path = os.path.join(yuvdir, code, str(level) + '_' + filename)

                strCmd = '/usr/bin/ffmpeg -i '+ ref_path +'  -vcodec '+ code +' -b:v ' + \
                    str(btrt)+'k ' + dis_path + ' -loglevel -8'

                print(strCmd)
                os.system(strCmd)
                to_yuv(dis_path, yuv_path)
                level += 1


def GaussNoise(ref_video, ref_file):
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'GaussNoise', str(i) + '_' + ref_file)
        yuv_path = os.path.join(yuvdir, 'GaussNoise', str(i) + '_' + ref_file)

        checkDir(os.path.join(resultdir, 'GaussNoise'))
        checkDir(os.path.join(yuvdir, 'GaussNoise'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.02*255*i)
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(aug.augment_image(frame))
        writer.close()
        to_yuv(dis_path, yuv_path)


def GaussBlur(ref_video, ref_file):
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'GaussBlur', str(i) + '_' + ref_file)
        yuv_path = os.path.join(yuvdir, 'GaussBlur', str(i) + '_' + ref_file)

        checkDir(os.path.join(resultdir, 'GaussBlur'))
        checkDir(os.path.join(yuvdir, 'GaussBlur'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        # Gauss Blur
        ks = 2 * i + 1
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(cv2.GaussianBlur(frame, (ks, ks), i))
        writer.close()
        to_yuv(dis_path, yuv_path)


def MontionBlur(ref_video, ref_file):
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'MontionBlur', str(i) + '_' + ref_file)
        yuv_path = os.path.join(yuvdir, 'MontionBlur', str(i) + '_' + ref_file)

        checkDir(os.path.join(resultdir, 'MontionBlur'))
        checkDir(os.path.join(yuvdir, 'MontionBlur'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        aug = iaa.MotionBlur(k=8 * i)
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(aug.augment_image(frame))
        writer.close()
        to_yuv(dis_path, yuv_path)


def Contrast(ref_video, ref_file):
    mp = {1 : 0.8, 2 : 1.2, 3 : 1.6}
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'Contrast', str(i) + '_' + ref_file)
        yuv_path = os.path.join(yuvdir, 'Contrast', str(i) + '_' + ref_file)

        checkDir(os.path.join(resultdir, 'Contrast'))
        checkDir(os.path.join(yuvdir, 'Contrast'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        aug = iaa.GammaContrast((mp[i], mp[i]))
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(aug.augment_image(frame))
        writer.close()
        to_yuv(dis_path, yuv_path)


def fps(ref_video, file, ref_path):
    for fps in ['15', '24']:
        dis_path = os.path.join(resultdir, 'fps', fps +  '_' + file)
        yuv_path = os.path.join(yuvdir, 'fps', fps +  '_' + file)

        checkDir(os.path.join(resultdir, 'fps'))
        checkDir(os.path.join(yuvdir, 'fps'))
            
        strCmd='/usr/bin/ffmpeg -i '+ ref_path + ' -r ' + fps + ' ' + dis_path
        # print(strCmd)
        os.system(strCmd)
        to_yuv(dis_path, yuv_path)
    


def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"make dir : {dir} success! ")

if __name__ == "__main__":

    for dir in [resultdir, yuvdir]:
        checkDir(dir)

    logger = getlogger()
    
    for cater in caters:
        cater_path = os.path.join(rootdir, cater)
        files = os.listdir(cater_path)

        for file in tqdm(files):
            if file.endswith('.mp4'):
                ref_path = os.path.join(cater_path, file)
                yuv_path = os.path.join(yuvdir, file)

                # Org. Video YUV
                to_yuv(ref_path, yuv_path)

                # Compress
                btrts = get_btrts(ref_path)
                compress(ref_path, btrts)

                video = skvideo.io.vread(ref_path)
                # GuassNoise
                GaussNoise(video, file)
                # GaussBlur
                GaussBlur(video, file)
                # Contrast
                Contrast(video, file)
                # MontionBlur 
                MontionBlur(video, file)
                # fps
                fps(video, file, ref_path)
