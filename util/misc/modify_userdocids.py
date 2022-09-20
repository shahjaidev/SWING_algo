"""
This file maps the user ids to  [0 - num_users) and maps the item ids to [0 - num_items).
This file is adapted from prism-reco-cf-training/model/als/modify_docid.py
"""
import os
from collections import defaultdict

import pandas as pd


def modify_userdocids(data_dir, name):
    in_filename = os.path.join(data_dir, name)
    out_filename = os.path.join(data_dir, "modify_userdocids_" + name)

    df = pd.read_csv(
        in_filename, delimiter=",", header=None, usecols=range(4), names=["uid", "tid", "UserId", "TargetId"]
    )

    item_map = defaultdict(int)
    user_map = defaultdict(int)
    item_counter = 0
    user_counter = 0
    for index, row in df.iterrows():
        tid = df.loc[index, "tid"]
        if tid in item_map:
            df.loc[index, "tid"] = item_map[tid]
        else:
            item_map[tid] = item_counter
            df.loc[index, "tid"] = item_counter
            item_counter += 1
        uid = df.loc[index, "uid"]
        if uid in user_map:
            df.loc[index, "uid"] = user_map[uid]
        else:
            user_map[uid] = user_counter
            df.loc[index, "uid"] = user_counter
            user_counter += 1

    df.to_csv(out_filename, index=False, header=None)


if __name__ == "__main__":
    data_dir = "/vc_data_blob/datasets/prism/image_reco/"
    name = "testset_v2_sample100.csv"
    modify_userdocids(data_dir, name)
