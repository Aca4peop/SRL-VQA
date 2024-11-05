import os
import skvideo.io
from collections import OrderedDict
from scipy import io as sio
import numpy as np
import json


if __name__ == "__main__":
    repeat = 5
    need = 2

    all_scenes = ['Keiba', 'Timelapse', 'BQTerrace', 'Carving', 'Chipmunks',
                  'Flowervase', 'ParkScene', 'PartyScene', 'BQMall', 'Cactus',
                  'Kimono','BasketballDrive']

    matPath = './CSIQData.mat'
    filePrefix = '/home1/server823-2/database/2D-Video/CSIQVideo/'

    
    data = sio.loadmat(matPath)
    name, mos, disType = data['file_name'], data['dmos_all'], data['dis_type']

    for i in range(repeat):
        length = len(all_scenes)
        randomIdx = np.random.permutation(np.arange(length))
        test_scenes = set()
        for idx in randomIdx[:need]:
            test_scenes.add(all_scenes[idx])

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

        for idx in range(len(name)):
            curScen = name[idx][0][0].split('_')[0]
            # fps = name[idx][0][0].split('_')[1][:2]
            fps = 0
            # Name DMOS DisType
            suffix = name[idx][0][0]
            videoMos = mos[idx][0]
            videoType = disType[idx][0]

            if curScen in test_scenes:
                tst_dis.append(suffix)
                tst_mos.append(float(videoMos))
                tst_type.append(int(videoType))
                tst_height.append(480)
                tst_width.append(832)
                tst_fps.append(float(fps))
            else:
                trn_dis.append(suffix)
                trn_mos.append(float(videoMos))
                trn_type.append(int(videoType))
                trn_height.append(480)
                trn_width.append(832)
                trn_fps.append(float(fps))

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

        path = './CSIQ_LEAVE2_' + str(i) + '.json'
        with open(path, 'w') as f:
            json.dump(ret, f, indent=4)