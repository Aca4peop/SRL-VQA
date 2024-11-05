import os
import json
import numpy as np
from collections import OrderedDict, defaultdict

dis_type_map = {'h264': 0, 'hevc': 1, 'GaussBlur': 2,
                'GaussNoise': 3, 'fps': 4, 'MotionBlur': 5, 'Contrast': 6}
# FPS:  0 : 30, 1 : 24, 2 : 15
video_type_map = {'Sport': 0, 'Nature': 1, 'Guitar': 2,
                  'Food': 3, 'Dance': 4, 'Apartment': 5, 'Animal': 6}

yuv_dir = '/home2/youtube8M/Clips/dis/yuv'


def path_change(file, type):
    suffix = file.split('/')[-1]
    suffix = suffix.replace('.mp4', '.yuv')
    if type == 'ref':
        return suffix

    cater = file.split('/')[-2]
    return os.path.join(cater, suffix)


def json_dump(subj_score_file, ww=0.8):

    ALL = []
    ref = os.listdir(yuv_dir)
    for file in ref:
        if file.endswith('yuv') == False:
            continue
        file = file.replace('yuv', 'mp4')
        ALL.append(file)

    randomIdx = np.random.permutation(np.arange(len(ALL)))
    split = int(np.floor(trainRatio * len(ALL)))
    train, test = set(), set()

    for idx in randomIdx[:split]:
        train.add(ALL[idx])
    for idx in randomIdx[split:]:
        test.add(ALL[idx])

    with open(subj_score_file, "r") as f:
        info = json.load(f)

    Mx = max(max(info['ref_bit_rate']), max(info['dis_bit_rate']))
    Mn = min(min(info['ref_bit_rate']), min(info['dis_bit_rate']))

    data = OrderedDict()
    data['test'] = OrderedDict()
    data['train'] = OrderedDict()

    test_dis = []
    test_ref = []
    test_type = []
    test_fps = []
    test_bits = []
    test_height = []
    test_width = []
    test_video_type = []

    train_dis = []
    train_ref = []
    train_type = []
    train_fps = []
    train_bits = []
    train_height = []
    train_width = []
    train_video_type = []

    # length = len(info['dis'])
    # print(length)
    # randomIdx = np.random.permutation(np.arange(length))
    # split = int(np.floor(0.8 * length))
    # train, test = randomIdx[:split], randomIdx[split:]
    for idx in range(len(info['ref'])):
        suffix = info['ref'][idx].split('/')[-1]
        if suffix in train:
            train_dis.append(info['dis'][idx])
            train_ref.append(info['ref'][idx])
            train_type.append(dis_type_map[info['dis_type'][idx]])
            fps, t = 0, float(info['dis_fps'][idx])

            # if t <= 25.0 and t >= 23.9:
            if t >= 25.1:
                fps = 0
            elif t >= 23.9 and t <= 25.0:
                fps = 1
            elif t <= 15.1:
                fps = 2

            train_fps.append(fps)
            train_bits.append(
                abs(float(info['ref_bit_rate'][idx]) - float(info['dis_bit_rate'][idx])))
            train_width.append(info['width'][idx])
            train_height.append(info['height'][idx])
            train_video_type.append(video_type_map[info['video_type'][idx]])
        else:
            test_dis.append(info['dis'][idx])
            test_ref.append(info['ref'][idx])
            test_type.append(dis_type_map[info['dis_type'][idx]])
            fps, t = 0, float(info['dis_fps'][idx])
            # if t <= 25.0 and t >= 23.9:
            #     fps = 1
            # elif t <= 15.1:
            #     fps = 2
            # if t <= 25.0 and t >= 23.9:
            if t >= 25.1:
                fps = 0
            elif t >= 23.9 and t <= 25.0:
                fps = 1
            elif t <= 15.1:
                fps = 2
            test_fps.append(fps)
            test_bits.append(
                abs(float(info['ref_bit_rate'][idx]) - float(info['dis_bit_rate'][idx])))
            test_width.append(info['width'][idx])
            test_height.append(info['height'][idx])
            test_video_type.append(video_type_map[info['video_type'][idx]])

    # for idx, bits in enumerate(train_bits):
    #     train_bits[idx] = 100.0 * (bits - Mn) / (Mx - Mn)

    # for idx, bits in enumerate(test_bits):
    #     test_bits[idx] = 100 * (bits - Mn) / (Mx - Mn)

    for idx, file in enumerate(train_dis):
        train_dis[idx] = path_change(file, 'dis')

    for idx, file in enumerate(train_ref):
        train_ref[idx] = path_change(file, 'ref')

    for idx, file in enumerate(test_dis):
        test_dis[idx] = path_change(file, 'dis')

    for idx, file in enumerate(test_ref):
        test_ref[idx] = path_change(file, 'ref')

    data['train']['dis'] = train_dis
    data['train']['ref'] = train_ref
    data['train']['mos'] = train_bits
    data['train']['type'] = train_type
    data['train']['fps'] = train_fps
    data['train']['width'] = train_width
    data['train']['height'] = train_height
    data['train']['video_type'] = train_video_type

    data['test']['dis'] = test_dis
    data['test']['ref'] = test_ref
    data['test']['mos'] = test_bits
    data['test']['type'] = test_type
    data['test']['fps'] = test_fps
    data['test']['width'] = test_width
    data['test']['height'] = test_height
    data['test']['video_type'] = test_video_type

    print(f"length of train : {len(train_dis)}")
    print(f"length of test : {len(test_dis)}")

    with open('./youtube8M_0.8_splitByContent.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    json_dump('./youtube8M_info_bits_type(contrast_motionblur)_fps_yuv_type.json')