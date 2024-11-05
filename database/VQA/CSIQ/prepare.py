import os
import skvideo.io
from collections import OrderedDict
from scipy import io as sio
import numpy as np
import json


if __name__ == "__main__":
    matPath = './CSIQData.mat'
    filePrefix = '/home1/server823-2/database/2D-Video/CSIQVideo/'

    
    data = sio.loadmat(matPath)
    name, mos, disType = data['file_name'], data['dmos_all'], data['dis_type']
    refIdx, refName = data['ref_index'], data['ref_name']
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
    trn_type = []


    tst_dis = []
    tst_mos = []
    tst_height = []
    tst_width = []
    tst_fps = []
    tst_type = []

    for idx in trainIdx:
        suffix = name[idx][0][0]
        videoName = os.path.join(filePrefix, suffix)
        videoMos = mos[idx][0]
        videoType = disType[idx][0]

        trn_dis.append(suffix)
        trn_mos.append(float(videoMos))
        trn_type.append(int(videoType))
        trn_height.append(480)
        trn_width.append(832)
        trn_fps.append(0)

    for idx in testIdx:
        if disType[idx] == 0:
            continue
        suffix = name[idx][0][0]
        videoName = os.path.join(filePrefix, suffix)
        videoMos = mos[idx][0]
        videoType = disType[idx][0]

        tst_dis.append(suffix)
        tst_mos.append(float(videoMos))
        tst_type.append(int(videoType))
        tst_height.append(480)
        tst_width.append(832)
        tst_fps.append(0)

    print("train num : ", len(trn_dis))
    print("test num : ", len(tst_dis))
    ret['train']['dis'] = trn_dis
    # ret['train']['ref'] = trn_ref
    ret['train']['mos'] = trn_mos
    ret['train']['type'] = trn_type
    ret['train']['height'] = trn_height
    ret['train']['width'] = trn_width
    ret['train']['fps'] = trn_fps

    ret['test']['dis'] = tst_dis
    # ret['test']['ref'] = tst_ref
    ret['test']['mos'] = tst_mos
    ret['test']['type'] = tst_type
    ret['test']['height'] = tst_height
    ret['test']['width'] = tst_width
    ret['test']['fps'] = tst_fps

    with open('./CSIQ_subj_score_TEST_NoRef.json', 'w') as f:
        json.dump(ret, f, indent=4)