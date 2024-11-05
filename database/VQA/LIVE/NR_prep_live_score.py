import os
import re
from collections import OrderedDict
import numpy as np
import json
import random

def make_score_file():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    seq_file_name = 'live_video_quality_seqs.txt'
    score_file_name = 'live_video_quality_data.txt'

    train_ratio = 0.8
    
    all_scenes = ['bs', "st", 'sh', "mc", "pa", 'sf', 'rb', 'tr', 'pr', 'rh']
    test_scene = ['bs', 'st']
    
    framerate = {
        'pa1_25fps.yuv': 25,
        'rb1_25fps.yuv': 25,
        'rh1_25fps.yuv': 25,
        'tr1_25fps.yuv': 25,
        'st1_25fps.yuv': 25,
        'sf1_25fps.yuv': 25,
        'bs1_25fps.yuv': 25,
        'sh1_50fps.yuv': 50,
        'mc1_50fps.yuv': 50,
        'pr1_50fps.yuv': 50}

    width = 768
    height = 432

    seqs = np.genfromtxt(os.path.join(dir_path, seq_file_name), dtype='str')
    score = np.genfromtxt(os.path.join(dir_path, score_file_name), dtype='float')[...,0]

    # print(score)
    # print(np.mean(score))
    # print(np.std(score))

    ret = OrderedDict()
    ret['train'] = OrderedDict()
    ret['test'] = OrderedDict()

    trn_dis = []
    trn_ref = []
    trn_mos = []
    trn_height = []
    trn_width = []
    trn_fps = []

    tst_dis = []
    tst_ref = []
    tst_mos = []
    tst_height = []
    tst_width = []
    tst_fps = []

    # max_label = np.max(score)
    # min_label = np.min(score)
    # score = 100*(score-min_label)/(max_label-min_label)

    # mean_label = np.mean(score)
    # std_label = np.std(score,ddof=1)
    # score = (score-mean_label)/std_label

    total_num = len(seqs)
    print("total num :", total_num)

    random_list = list(range(total_num))

    random.shuffle(random_list)

    train_num = int(train_ratio * total_num)

    test_index_list = random_list[0 : total_num - train_num] 
    # test_index_list = [115,139,80,88,110,121,35,119,122,18,135,54,73,9,84,4,15,13,78,56,99,60,129,3,58,97,27,16,1,112,152,159]
    # test_index_list = [115,139,80,88,110,121,35,119,122,18,135,54,73,9,84,4,15,13,78,56,99,60,129,3,58,97,27,16,1,112]
    fps = 0
    

    for i, idx in enumerate(random_list):
        dst = seqs[idx]
        mos = score[idx]

        print(f"dst : {dst}")
        print(f"mos : {mos}")
        print(f"idx : {idx}")

        if i < train_num:
            trn_dis.append(dst)
            # trn_ref.append(ref)
            trn_mos.append(float(mos))
            trn_height.append(height)
            trn_width.append(width)
            trn_fps.append(fps)
        else:
            tst_dis.append(dst)
            # tst_ref.append(ref)
            tst_mos.append(float(mos))
            tst_height.append(height)
            tst_width.append(width)
            tst_fps.append(fps)

    # for index, item in enumerate(seqs):
    #     dst = item
    #     mos = score[index]
        
        # print(index)
        # print(item)
        # print(mos)
        # if(item[4] == '1' or item[4] == '2'):
        #     width = 640
        #     height = 480
        # else:
        #     width = 1280
        #     height = 720
        # if(index>=150):
        #     continue

        # if index in test_index_list:
        #     tst_dis.append(dst)
        #     # tst_ref.append(ref)
        #     tst_mos.append(float(mos))
        #     tst_height.append(height)
        #     tst_width.append(width)
        #     tst_fps.append(fps)
        # else:
        #     trn_dis.append(dst)
        #     # trn_ref.append(ref)
        #     trn_mos.append(float(mos))
        #     trn_height.append(height)
        #     trn_width.append(width)
        #     trn_fps.append(fps)

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

    with open('./live_subj_score_nr_ref_4.json', 'w') as f:
        json.dump(ret, f, indent=4)
    # with open('./live_subj_score_{}_nr_ref.json'.format('_'.join(test_scene)), 'w') as f:
    #     json.dump(ret, f, indent=4, sort_keys=True)


if __name__ == "__main__":

    make_score_file()

    print('Done')
