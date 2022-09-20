from collections import defaultdict

import pandas as pd


in_rating = r"/vc_data_blob/datasets/prism/image_reco/feed1B_1month/cf_trainset.csv"
out_target_items = r"/vc_data_blob/datasets/prism/image_reco/feed1B_1month/cf_target_items.tsv"
new_rating = list()
target_items = list()
seen_items = set()
item_map = defaultdict(int)
counter = 0
with open(in_rating, "r") as fr:
    for line in fr:
        counter += 1
        uid, tid, userId, targetId = line.strip().replace("\n", "").split(",")

        if counter % 10000 == 1:
            print(counter)
        if int(tid) not in seen_items and int(tid) < 1e4:
            seen_items.add(int(tid))
            target_items.append([targetId, tid])
pd.DataFrame(target_items).to_csv(out_target_items, index=False, header=None, sep="\t")
