__author__ = "Rohit Sharma"

import os
import numpy as np
from utils import load_json, save_json


def create_smth_json(feat_file, data_file, smth_feat_cache_path):
    if os.path.exists(smth_feat_cache_path):
        print('Found smth cache, loading ...')
        return load_json(smth_feat_cache_path)
    
    show_names = load_json(data_file)
    clip_features = np.load(feat_file)
    smth_features = {}

    assert len(show_names) == len(clip_features), 'Show names len: {0}, Clip features len: {1}'.format(len(show_names), len(clip_features))

    for idx, sn in enumerate(show_names):
        sn = sn['id'].split('/')[-2]
        
        smth_features[sn] = clip_features[idx].tolist()[0]

        if idx < 5:
            print(sn, clip_features[idx].tolist()[0])
    save_json(smth_features, smth_feat_cache_path)
    return smth_features


def main():
    smth_feat_file = '/home/rsharma/dev/smth-smth-v2-baseline-with-models/bbt_20bn_features.npy'
    data_json_file = '/home/akshatabhat/tvqa_data.json'
    smth_feat_cache_path = 'data/tvqa_smth.json'

    create_smth_json(smth_feat_file, data_json_file, smth_feat_cache_path)


if __name__ == "__main__":
    main()
