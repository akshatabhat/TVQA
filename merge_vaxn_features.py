__author__ = "Rohit Sharma"

import os
import numpy as np
from utils import load_json, save_json


def merge_vaxn_features(vaxn_feat_root_path, vaxn_feat_cache_path):
    """Generates a JSON file with all the Video action based features
    
    Arguments:
        vaxn_feat_root_path {str} -- path of the root directory containing the clip level features
        vaxn_feat_cache_path {str} -- destination path of the json
    
    Returns:
        dict -- features as a dictionary of clip names as keys
    """
    if os.path.exists(vaxn_feat_cache_path):
        print("Found vaxn cache, loading ...")
        return load_json(vaxn_feat_cache_path)

    show_names = ["bbt"]
    vaxn_features = {}
    for sn in show_names:
        cur_base_path = os.path.join(vaxn_feat_root_path, "{}_frames".format(sn))
        clip_dirs = [subdir for subdir in os.listdir(cur_base_path) if os.path.isdir(os.path.join(cur_base_path, subdir))]
        for clip in clip_dirs:
            if not os.path.isfile(os.path.join(cur_base_path, clip, 'feature.npy')):
                print("**********Error no feat file {}**********".format(clip))
                vaxn_features[clip] = np.zeros(512).tolist()
            else:
                clip_feat = np.load(os.path.join(cur_base_path, clip, 'feature.npy'), allow_pickle=False)
                if clip_feat.shape[0] != 0:
                    #clip_feat = np.amax(clip_feat, axis=0)
                    vaxn_features[clip] = clip_feat.tolist()
                else:
                    print("**********Error in feature dim {}**********".format(clip))
                    vaxn_features[clip] = np.zeros(512).tolist()
    save_json(vaxn_features, vaxn_feat_cache_path)
    return vaxn_features


def main():
    vaxn_root_path = "/home/tvqa_data/frames/uncompressed/frames_hq"
    vaxn_cache_path = "data/tvqa_vaxn.json"

    merge_vaxn_features(vaxn_root_path, vaxn_cache_path)


if __name__ == "__main__":
    main()
