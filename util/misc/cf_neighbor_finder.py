# This code returns the neighbors along with their click hisotry for a given user id.
# it is mainly used for debugging the output of CF models


import pandas as pd


USER_HISTORY = "/home/azureuser/cloudfiles/data/datastore/mahajiag/mahajiag/29094_short_term2/UserIdHistory.tsv"


def findNeighbors(filename, uids=["5E36C65757DC799CAF226921FFFFFFFF"]):
    df = pd.read_csv(
        filename,
        sep=",",
        header=1,
        usecols=range(5),
        names=["uid", "request_time", "murl", "prism_flag", "did"],
        engine="python",
    )

    neigbhors = set()
    for uid in uids:
        dids = df[df["uid"] == uid]["did"]
        for did in dids:
            res = df[df["did"] == did]["uid"]
            for uid in res:
                neigbhors.add(uid)
    return df, neigbhors


df, neighbors = findNeighbors(filename=USER_HISTORY, uids=["5E36C65757DC799CAF226921FFFFFFFF"])
results = df[df["uid"].isin(neighbors)]
results.to_csv(r"neighbors.tsv", sep="\t", index=False)
