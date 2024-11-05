import os
import re
from collections import OrderedDict
from sklearn.model_selection import KFold
import numpy as np
import json
import random
import csv

def make_score_file():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    info_file_name = './kon_kfold.csv'

    repeat = 5
    width = 960
    height = 540
    fps = 0

    length = 1200
    index = list(range(length))
    fold_indexs = []

    for i in range(repeat):
        fold_indexs.append(index[i::5])


    for i in range(repeat):
        testIdx = []
        trainIdx = []

        for idx in range(repeat):
            # test index
            if idx == i:
                testIdx.extend(fold_indexs[idx])
            else:
                trainIdx.extend(fold_indexs[idx])

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

        csvFile = open(os.path.join(dir_path, info_file_name), "r")
        seqs = csv.reader(csvFile)

        for index, item in enumerate(seqs):
            dst = item[2]
            mos = item[3]
            dst = dst.replace('.mp4', '.yuv')

            if index not in trainIdx:
                tst_dis.append(dst)
                # tst_ref.append(ref)
                tst_mos.append(float(mos))
                tst_height.append(height)
                tst_width.append(width)
                tst_fps.append(fps)
            else:
                trn_dis.append(dst)
                # trn_ref.append(ref)
                trn_mos.append(float(mos))
                trn_height.append(height)
                trn_width.append(width)
                trn_fps.append(fps)
                    
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

        print(f"train num : {len(trainIdx)}")
        print(f"test num : {len(testIdx)}")
        
        path = f"./KON_hzy_{i}.json"
        with open(path, 'w') as f:
            json.dump(ret, f, indent=4, sort_keys=True)
            
        csvFile.close()


if __name__ == "__main__":

    make_score_file()

    print('Done')