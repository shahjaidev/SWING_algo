from collections import defaultdict

import pandas as pd


in_rating = r"/vc_data_blob/datasets/prism/image_reco/feed1B_1month/cf_trainset.csv"
out_target_items = r"/vc_data_blob/datasets/prism/image_reco/feed1B_1month/cf_target_items.tsv"
df = pd.read_csv(in_rating, delimiter=",", header=None, usecols=range(4), names=["uid", "tid", "UserId", "TargetId"])

new_rating = list()
target_items = list()
seen_items = set()
item_map = defaultdict(int)
counter = 0
for index, row in df.iterrows():
    tid = df.loc[index, "tid"]
    if int(tid) not in seen_items and int(tid) < 1e4:
        seen_items.add(int(tid))
        target_items.append([tid, df.loc[index, "TargetId"]])
pd.DataFrame(target_items).to_csv(out_target_items, index=False, header=None, sep="\t")
