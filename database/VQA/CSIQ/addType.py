import os
import re
import skvideo.io
from collections import OrderedDict
from scipy import io as sio
import numpy as np
import json


if __name__ == "__main__":

    with open('./csiq_subj_score_nr.json') as f:
        data = json.load(f)

        for mode in ['train', 'test']:
            typeInfo = []
            for name in data[mode]['dis']:
                cur = re.split(r'[_.]', name)[-2]
                if cur == 'ref':
                    typeInfo.append(0)
                else:
                    cur = int(cur)
                    ans = cur // 3
                    if cur % 3:
                        ans += 1
                    typeInfo.append(ans)
            data[mode]['type'] = typeInfo
        
        ret = data

        with open('./CSIQ_subj_score_5.json', 'w') as f:
            json.dump(ret, f, indent=4)