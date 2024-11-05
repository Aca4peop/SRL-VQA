import os
import skvideo.io
from collections import OrderedDict
from scipy import io as sio
import numpy as np
import json


if __name__ == "__main__":
    matPath = './data.mat'
    trainRatio = 0
    filePrefix = '/home1/server823-2/database/2D-Video/LIVE_Video_Quality_Challenge(VQC)_Database/video/'

    
    data = sio.loadmat(matPath)
    name, mos = data['video_list'], data['mos']
    length = len(name)


    randomIdx = np.random.permutation(np.arange(length))
    split = int(np.floor(trainRatio * length))
    trainIdx, testIdx = randomIdx[:split], randomIdx[split:]

    ret = OrderedDict()
    ret['train'] = OrderedDict()
    ret['test'] = OrderedDict()

    trn_dis = []
    trn_mos = []
    trn_height = []
    trn_width = []
    trn_fps = []


    tst_dis = []
    tst_mos = []
    tst_height = []
    tst_width = []
    tst_fps = []

    for idx in trainIdx:
        suffix = name[idx][0][0]
        videoName = os.path.join(filePrefix, suffix)
        videoMos = mos[idx][0]
        metadata = skvideo.io.ffprobe(videoName)['video']
        width = metadata['@width']
        height = metadata['@height']
        fps = int(metadata['@avg_frame_rate'].split('/')[0])

        trn_dis.append(suffix)
        trn_mos.append(float(videoMos))
        trn_height.append(height)
        trn_width.append(width)
        trn_fps.append(fps)

    for idx in testIdx:
        suffix = name[idx][0][0]
        videoName = os.path.join(filePrefix, suffix)
        videoMos = mos[idx][0]
        metadata = skvideo.io.ffprobe(videoName)['video']
        width = metadata['@width']
        height = metadata['@height']
        fps = int(metadata['@avg_frame_rate'].split('/')[0])

        tst_dis.append(suffix)
        tst_mos.append(float(videoMos))
        tst_height.append(height)
        tst_width.append(width)
        tst_fps.append(fps)

    print("train num : ", len(trn_dis))
    print("test num : ", len(tst_dis))
    ret['train']['dis'] = trn_dis
    # ret['train']['ref'] = trn_ref
    ret['train']['mos'] = trn_mos
    ret['train']['height'] = trn_height
    ret['train']['width'] = trn_width
    ret['train']['fps'] = trn_fps

    ret['test']['dis'] = tst_dis
    # ret['test']['ref'] = tst_ref
    ret['test']['mos'] = tst_mos
    ret['test']['height'] = tst_height
    ret['test']['width'] = tst_width
    ret['test']['fps'] = tst_fps

    with open('./VQC_subj_score_TEST.json', 'w') as f:
        json.dump(ret, f, indent=4)