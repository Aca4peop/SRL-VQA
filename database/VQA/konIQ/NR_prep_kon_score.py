import os
import re
from collections import OrderedDict
import numpy as np
import json
import random
import csv

def make_score_file():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    info_file_name = 'KoNViD_1k_attributes.csv'

    train_ratio = 0.8

    width = 960
    height = 540
    fps = 0

    csvFile = open(os.path.join(dir_path, info_file_name), "r")
    seqs = csv.reader(csvFile)

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

    random_list = list(range(1,1201))
    random.shuffle(random_list)
    train_num = int(train_ratio * 1200)
    train_index_list = random_list[:train_num]
    # print(train_index_list)

    for index, item in enumerate(seqs):
        if(index == 0):
            continue
        dst = item[2]
        mos = item[3]
        dst = dst.replace('.mp4', '.yuv')

        if index not in train_index_list:
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
            
    csvFile.close()

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

    with open('./kon_subj_score_4.json', 'w') as f:
        json.dump(ret, f, indent=4, sort_keys=True)


if __name__ == "__main__":

    make_score_file()

    print('Done')
