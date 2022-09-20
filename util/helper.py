#!/usr/bin/python3
import ast
import functools
import logging
import math
import os
import pickle
import shutil
import time
from collections import defaultdict, deque
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from util.record_io import RecordFormatter, RecordIO


logger = logging.getLogger(__name__)


def add_and_fit(lst, size, ele):
    """
    Context: This function is to operate on a priority queue storing (ckpt_path, metrics).
    lst: the queue
    size: the length limit of queue, -1 means save all checkpoints
    ele: formatted like (ckpt_path, metrics)
    """
    lst.append(ele)
    if size >= 0 and len(lst) > size:
        m_min = min(lst, key=lambda x: x[1])
        logger.info(f"ckpt limit {size} reached, removed ckpt at {m_min[0]} based on {m_min[1]}")
        shutil.rmtree(m_min[0])
        lst.remove(m_min)


def load_npy_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, "user.npy"), "rb") as f:
        user_emb = np.load(f, allow_pickle=True)
    with open(os.path.join(ckpt_dir, "item.npy"), "rb") as f:
        item_emb = np.load(f, allow_pickle=True)
    return user_emb, item_emb


def log_memory_usage(marker):
    gb_used = psutil.virtual_memory().used / 1e9
    logger.info("%.1f GB memory used, after stage: %s", gb_used, marker)


def prepare_test_complementary(train_user_items, test_user_items, item_labels=None):
    # with the Complementary Recall Calculation logic provided by Rana
    logger.info("Prepare for test begin:")
    users = np.unique(coo_matrix(test_user_items).row)
    m = train_user_items[users].tocsr()
    click_in_train = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    m = test_user_items[users].tocsr()
    ground_truths = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    train_target_test_overlaps, complementary_ground_truths = [], []
    # should remove item from ground_truths if it also appears in training set
    for i in range(len(users)):
        # keep only items in target index
        train_target_overlap = click_in_train[i]
        if item_labels is not None:
            train_target_overlap = np.intersect1d(click_in_train[i], item_labels)
        # overlap of items in both train set and item set for user[i]
        train_target_test_overlap = np.intersect1d(train_target_overlap, ground_truths[i])
        train_target_test_overlaps.append(train_target_test_overlap)
        # remove the overlap items for complementary recall calculation
        if len(train_target_test_overlap) > 0:
            # index of overlap items
            intersect_ind = np.in1d(ground_truths[i], train_target_test_overlap).nonzero()[0]
            # remove them from test set
            complementary_ground_truths.append(np.delete(ground_truths[i], intersect_ind))
        else:
            complementary_ground_truths.append(ground_truths[i])
    logger.info(
        "Prepare for test done",
    )
    return click_in_train, train_target_test_overlaps, ground_truths, complementary_ground_truths


def prepare_test_ndcg(test_tripples):
    logger.info("Prepare for test ndcg begin:")
    user_item_ctr_map = {}
    item_ids = []
    for tripple in test_tripples:
        user, item, label = tripple[0], tripple[1], tripple[2]
        if user not in user_item_ctr_map:
            user_item_ctr_map[user] = {"label": [], "item": []}

        user_item_ctr_map[user]["label"].append(label)
        user_item_ctr_map[user]["item"].append(item)
        item_ids.append(item)
    item_ids = np.unique(np.array(item_ids, dtype=np.int32))
    return user_item_ctr_map, item_ids


def prepare_test(train_user_items, test_user_items, item_labels=None):
    logger.info("Prepare for test begin:")
    users = np.unique(coo_matrix(test_user_items).row)
    m = train_user_items[users].tocsr()
    users_liked = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    m = test_user_items[users].tocsr()
    ground_truths = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    # if user has item in both dev set and train set, should let the items to be recalled
    for i in range(len(users)):
        # keep only items in target index
        items_in_train = users_liked[i]
        if item_labels is not None:
            items_in_train = np.intersect1d(users_liked[i], item_labels)
        intersection = np.intersect1d(items_in_train, ground_truths[i])
        if len(intersection) > 0:
            # print("overlap pairs for user: " + str(i) + " =" + str(len(intersection)))
            # overlap_cnt[i] = len(intersection)
            intersect_ind = np.in1d(items_in_train, intersection).nonzero()[0]
            items_in_train = np.delete(items_in_train, intersect_ind)
        users_liked[i] = items_in_train
    logger.info("Prepare for test ended: users_liked and ground_truths generated.")
    return users_liked, ground_truths


# Append extra dim to let Euclidean distance (L2) simulate inner product.
def append_extra_dim_for_l2(embeddings, append_zero=True):
    if append_zero:
        embeddings = np.array([np.append(e, [0.0]) for e in embeddings])
    else:
        max_item_len = max([np.linalg.norm(e) for e in embeddings])
        embeddings = np.array(
            [np.append(e, [math.sqrt(pow(max_item_len, 2) - pow(np.linalg.norm(e), 2))]) for e in embeddings]
        )
    return embeddings


# used for public datast movielenz
# def get_dataframe(filename, delimiter):
#     df = pd.read_csv(filename,
#                      sep=",",
#                      header=0,
#                      usecols=range(4),
#                      names=["uid", "tid", "Rating", "Time"],
#                      engine= "python")
#     df["Rating"] = (df["Rating"] > 3).astype(int)
#     df = df[df["Rating"]!=0]
#     df["UserId"] = df["uid"].apply(str)
#     df["TargetId"] = df["tid"].apply(str)
#     df = df.drop_duplicates(keep="first", inplace=False)
#     df = df.reset_index(drop=True)
#     return df


def get_rating_matrix(tids, uids, data=None):
    if data is None:
        data = np.ones(len(uids)).astype(np.float32)
    rating = coo_matrix((data, (tids, uids))).tocsr()
    return rating


def get_dataframe_and_rating(args, filename, delimiter):
    df = get_dataframe(filename, delimiter, read_frequency=args.read_frequency)
    if "Frequency" in df:
        frequency = df["Frequency"].astype(np.int32)
    else:
        frequency = None
    rating = get_rating_matrix(df["tid"].astype(np.int32), df["uid"].astype(np.int32), frequency)
    return rating, df


def bm25_weight(X, K1, B):
    X = coo_matrix(X)
    N = float(X.shape[0])
    idf = np.log(N) - np.log1p(np.bincount(X.col))
    # calculate length_norm per document (artist)
    item_sums = np.bincount(X.row)
    average_length = item_sums.mean()
    length_norm = (1.0 - B) + B * item_sums / average_length
    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X


def binary_weight(X):
    X = coo_matrix(X)
    X.data = np.ones(len(X.data))
    return X


def frequency_weight(X):
    X = coo_matrix(X)
    return X


def log_frequency_weight(X):
    X = coo_matrix(X)
    X.data = np.log2(X.data + 1)
    return X


def ips_weight(X, ips_power):
    # Based on equation 11 of this paper [Saito, Yuta. "Unbiased Pairwise Learning from Implicit Feedback." NeurIPS 2019 Workshop on Causal Machine Learning. 2019.]
    # Use popularity score as propensity score, and set click_weights as inverse of propensity score
    X = coo_matrix(X)
    item_sums = np.bincount(X.row)
    normalized_item_sums = (item_sums / np.max(item_sums)) ** ips_power
    X.data = 1 / normalized_item_sums[X.row]
    return X


def convert_time_to_timestamp(df):
    """Convert the requesttime to POSIX timestamp"""
    format = "%m/%d/%Y %I:%M:%S %p"
    df["time"] = df["time"].map(lambda time_string: datetime.strptime(time_string, format).timestamp())
    return df


def get_ratings_transformed_by_time(args, filename, delimiter):
    if args.read_frequency is True:
        ratings_df = pd.read_csv(
            filename,
            delimiter=delimiter,
            header=None,
            usecols=[0, 1, 2, 3, 4, 5],
            names=["uid", "tid", "UserId", "TargetId", "Frequency", "time"],
        )
        frequency = ratings_df["Frequency"].astype(np.int32)
    else:
        ratings_df = pd.read_csv(
            filename,
            delimiter=delimiter,
            header=None,
            usecols=[0, 1, 2, 3, 5],
            names=["uid", "tid", "UserId", "TargetId", "time"],
        )
        ratings_df = ratings_df.sort_values(by="time", ascending=False)
        ratings_df = ratings_df.drop_duplicates(
            subset=["uid", "tid", "UserId", "TargetId"], keep="first", inplace=False
        )
        frequency = None

    ratings_df = convert_time_to_timestamp(ratings_df)
    ratings_df = ratings_df.reset_index(drop=True)
    raw_ratings = get_rating_matrix(ratings_df["tid"].astype(np.int32), ratings_df["uid"].astype(np.int32), frequency)
    raw_ratings.eliminate_zeros()
    ratings_df = polynomial_transform(ratings_df, args.range_max, args.range_min, args.power)
    data = ratings_df["weight"].values.astype(np.float32)
    weight_matrix = coo_matrix((data, (ratings_df["tid"].astype(np.int32), ratings_df["uid"].astype(np.int32))))
    ratings, N, idf, item_sums, average_length = reweight_data(args, raw_ratings)
    ratings.data *= args.rating_scale
    ratings.data += args.rating_offset
    ratings += weight_matrix
    ratings = ratings.tocsr()
    return ratings, ratings_df, N, idf, item_sums, average_length


def reweight_data(args, raw_ratings):
    X = coo_matrix(raw_ratings)
    N = float(X.shape[0])
    idf = np.log(N) - np.log1p(np.bincount(X.col))
    # calculate length_norm per document (artist)
    item_sums = np.bincount(X.row)
    average_length = item_sums.mean()

    if args.weighting_method == "binary_weight":
        ratings = binary_weight(raw_ratings)
    elif args.weighting_method == "frequency_weight":
        ratings = frequency_weight(raw_ratings)
    elif args.weighting_method == "log_frequency_weight":
        ratings = log_frequency_weight(raw_ratings)
    elif args.weighting_method == "ips_weight":
        ratings = ips_weight(raw_ratings, args.ips_power)
    elif args.weighting_method == "bm25_weight":
        ratings = bm25_weight(raw_ratings, args.K1, args.B)

    return ratings, N, idf, item_sums, average_length


def polynomial_transform(df, range_max, range_min, power):
    k = (range_max - range_min) / (df["time"].max() - df["time"].min())
    b = range_min - k * df["time"].min()
    df["weight"] = k * df["time"] + b
    df["weight"] = df["weight"] ** power
    return df


def load_train_data_weighted_by_time(args, filename, delimiter=",", logger_in=None):
    """Load training data"""
    if args.logger_in:
        global logger
        logger = logger_in
    logger.info("loading data from %s", filename)
    train_item_users, train_df, N, idfs, item_sums, average_length = get_ratings_transformed_by_time(
        args, filename, delimiter
    )
    train_user_items = train_item_users.T.tocsr()

    num_ratings = len(train_df)
    num_users = train_user_items.shape[0]
    num_items = train_user_items.shape[1]
    density = num_ratings / (num_users * num_items) * 100
    logger.info("train ratings: %s", num_ratings)
    logger.info("train users: %s", num_users)
    logger.info("train items: %s", num_items)
    logger.info("train matrix density: %.6f %%", density)

    return train_user_items, train_df, N, idfs, item_sums, average_length


def load_train_data(args, filename, delimiter=",", logger_in=None):
    """Load training data"""
    if logger_in:
        global logger
        logger = logger_in
    logger.info("loading data from %s", filename)
    train_item_users, train_df, N, idfs, item_sums, average_length = get_ratings(args, filename, delimiter)
    train_user_items = train_item_users.T.tocsr()

    num_ratings = len(train_df)
    num_users = train_user_items.shape[0]
    num_items = train_user_items.shape[1]
    density = num_ratings / (num_users * num_items) * 100
    logger.info("train ratings: %s", num_ratings)
    logger.info("train users: %s", num_users)
    logger.info("train items: %s", num_items)
    logger.info("train matrix density: %.6f %%", density)

    return train_user_items, train_df, N, idfs, item_sums, average_length


def load_target_items(filename, train_df):
    if filename is not None:
        logger.info("loading target items from %s", filename)
        target_items = pd.read_csv(filename, delimiter="\t", header=None, names=["TargetId", "id"])
        items = train_df[["TargetId", "tid"]]
        items = items.drop_duplicates(keep="first", inplace=False)
        target_df = items[items["TargetId"].isin(target_items["TargetId"])]
        item_labels = target_df["tid"].tolist()
    else:
        item_labels = None

    return item_labels


def print_input_args(args):
    logger.info("input args:")
    for arg in vars(args):
        logger.info("\t%s: %s", arg, getattr(args, arg))
    logger.info("")


def log_input_args(args):
    logger.info("logging input args to mlflow")
    for arg in vars(args):
        mlflow.log_param(arg, getattr(args, arg))


def save_compressed(filename: str, **kwargs):
    """Save keyword arguments as compressed data with numpy.

    This is a convenience wrapper around numpy.savez_compressed()

    Args:
        filename (str): where to store the file, conventionally .npz
    """
    # Ensure the parent directory exists
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    with open(filename, "wb") as f:
        np.savez_compressed(f, **kwargs)


def save_ckpt(trainer, ckpt_path):
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    with open(os.path.join(ckpt_path, "user.npy"), "wb") as f:
        np.save(f, trainer.model.get_user_embeddings())
    with open(os.path.join(ckpt_path, "item.npy"), "wb") as f:
        np.save(f, trainer.model.get_item_embeddings())

    state_dict = trainer.state_dict()
    state_dict["model"] = None
    torch.save(state_dict, os.path.join(ckpt_path, "engine-states.pt"))


def save_model(output_dir, model):
    """Save model for DLIS serving"""
    model_path = os.path.join(output_dir, "model.pkl")
    logger.info("saving serving model to %s", model_path)
    pickle.dump(model, open(model_path, "wb"), protocol=4)


def write_user_doc_preds(user_doc_preds, train_df, test_df, output_dir, filename):
    """
    For each UserId, we save the recommended list of TargetIds to later visualize results.
    user_doc_preds provides uid and their recommended list of tids. We obtain corresponding UserId and TargetId from uid and tid in train set.
    If uid or tid does not exist in train set, we obtain UserId or TargetId from test set.
    """
    if not os.path.exists(output_dir):
        logger.info("creating output directory %s", output_dir)
        os.makedirs(output_dir)
    else:
        logger.info("%s exists", output_dir)
    file = os.path.join(output_dir, filename)

    logger.info("saving " + file)

    user_id_map_from_train = {}
    doc_id_map_from_train = {}
    user_id_map_from_test = {}
    doc_id_map_from_test = {}

    # counting the number of users that clicked the document in trainset
    docClick = train_df[["TargetId", "UserId"]].groupby(["TargetId"]).count().reset_index()
    docClick_dict = docClick.set_index("TargetId").to_dict()["UserId"]
    docClick_dict = defaultdict(int, docClick_dict)

    # counting the number of items the user has clicked in trainset
    userClick = train_df[["TargetId", "UserId"]].groupby(["UserId"]).count().reset_index()
    userClick_dict = userClick.set_index("UserId").to_dict()["TargetId"]
    userClick_dict = defaultdict(int, userClick_dict)

    for user, uid, doc, did in zip(train_df["UserId"], train_df["uid"], train_df["TargetId"], train_df["tid"]):
        user_id_map_from_train[uid] = user
        doc_id_map_from_train[did] = doc
    for user, uid, doc, did in zip(test_df["UserId"], test_df["uid"], test_df["TargetId"], test_df["tid"]):
        user_id_map_from_test[uid] = user
        doc_id_map_from_test[did] = doc
    file_distances = file[:-4] + "_distances.tsv"
    file_connections = file[:-4] + "_connections.tsv"
    with open(file, "w") as fw, open(file_distances, "w") as fw_distances, open(
        file_connections, "w"
    ) as fw_connections:
        for uid, docIds_distances in user_doc_preds.items():
            docIds = docIds_distances[0]
            distances = docIds_distances[1]
            recommendation_diversity = docIds_distances[2]
            recommendation_relevancy = docIds_distances[3]
            recommendation_soft_recall = docIds_distances[4]

            if uid in user_id_map_from_train:
                user = user_id_map_from_train[uid]
            elif uid in user_id_map_from_test:
                user = user_id_map_from_test[uid]
            else:
                print("Not in input-path or test-path,  uid:", uid)
            doc_list = []
            connection_list = []
            for did in docIds:
                if did in doc_id_map_from_train:
                    doc = doc_id_map_from_train[did]
                    doc_list.append(doc)
                    connection_list.append(str(docClick_dict[doc]))
                elif did in doc_id_map_from_test:
                    doc = doc_id_map_from_test[did]
                    doc_list.append(doc)
                    connection_list.append(str(docClick_dict[doc]))
                else:
                    doc_list.append("Not in input-path or test-path, did:" + str(did))
                    connection_list.append("Not in input-path or test-path, did:" + str(did))
            line = (
                user
                + "\t"
                + "|".join(doc_list)
                + "\t"
                + recommendation_diversity
                + "\t"
                + recommendation_relevancy
                + "\t"
                + recommendation_soft_recall
                + "\n"
            )
            fw.write(line)
            distances_list = []
            for distance in distances:
                distances_list.append(str(round(distance, 2)))
            line_distances = user + "\t" + "|".join(distances_list) + "\n"
            fw_distances.write(line_distances)
            line_connections = user + "\t" + "|".join(connection_list) + "\n"
            fw_connections.write(line_connections)


def load_model(base_model_dir, factors, load_embeddings):
    user_ids, user_embeddings, item_ids, item_embeddings = load_results(base_model_dir, factors, load_embeddings)
    logger.info("Initialize embeddings...")
    uids = list(map(int, user_ids))
    iids = list(map(int, item_ids))
    udim = max(uids) + 1  # add 1 for 0, as input data is 1-based
    idim = max(iids) + 1
    user_factors = np.zeros((udim, factors))
    item_factors = np.zeros((idim, factors))
    for i in range(len(uids)):
        user_factors[
            uids[i],
        ] = user_embeddings[i]
    utup = user_factors.shape
    logger.info("User embeddings initialized. Shape:(" + str(utup[0]) + "," + str(utup[1]) + ").")
    for i in range(len(iids)):
        item_factors[
            iids[i],
        ] = item_embeddings[i]
    itup = item_factors.shape
    logger.info("Item embeddings initialized. Shape:(" + str(itup[0]) + "," + str(itup[1]) + ").")
    return user_factors, item_factors


def load_results(model_dir, factors, load_embeddings, load_weights=False):
    user_fields = [
        ["char", 32],
        ["float", factors],
        ["double", 1],
    ]
    item_fields = [
        ["char", 32],
        ["float", factors],
        ["double", 1],
    ]
    if load_weights:
        item_fields.append(["float", 1])
    user_rec_io = RecordIO(RecordFormatter(user_fields))
    item_rec_io = RecordIO(RecordFormatter(item_fields))
    logger.info("Loading embeddings...")
    if load_embeddings == "npy":
        users = []
        items = []
        npy_dir = os.path.join(model_dir, "npy")
        with open(os.path.join(npy_dir, "userids.npy"), "rb") as f:
            users.append(list(np.load(f, allow_pickle=True)))
        with open(os.path.join(npy_dir, "uservector.npy"), "rb") as f:
            users.append(list(np.load(f, allow_pickle=True)))
        with open(os.path.join(npy_dir, "useridfs.npy"), "rb") as f:
            users.append(list(np.load(f, allow_pickle=True)))

        with open(os.path.join(npy_dir, "itemids.npy"), "rb") as f:
            items.append(list(np.load(f, allow_pickle=True)))
        with open(os.path.join(npy_dir, "itemvector.npy"), "rb") as f:
            items.append(list(np.load(f, allow_pickle=True)))
        with open(os.path.join(npy_dir, "itemidfs.npy"), "rb") as f:
            items.append(list(np.load(f, allow_pickle=True)))

        if load_weights:
            with open(os.path.join(npy_dir, "itemweights.npy"), "rb") as f:
                items.append(list(np.load(f, allow_pickle=True)))

    elif load_embeddings == "binary":
        users = user_rec_io.read_binary(os.path.join(model_dir, "uservector.bin"))
        items = item_rec_io.read_binary(os.path.join(model_dir, "itemvector.bin"))
    elif load_embeddings == "text":
        users = user_rec_io.read_text(os.path.join(model_dir, "uservector.tsv"))
        items = item_rec_io.read_text(os.path.join(model_dir, "itemvector.tsv"))

    user_ids = users[0]
    user_embeddings = users[1]
    item_ids = items[0]
    item_embeddings = items[1]
    return user_ids, np.array(user_embeddings), item_ids, np.array(item_embeddings)


def get_users_items_to_save(train_df, max_item_index_to_save):

    logger.info("prepare users to save")
    users = train_df[["UserId", "uid"]]
    users = users.drop_duplicates(keep="first", inplace=False)
    users.set_index("uid", inplace=True)
    users.sort_index(inplace=True)

    logger.info("prepare items to save")
    items = train_df[["TargetId", "tid"]]
    items = items.drop_duplicates(keep="first", inplace=False)
    items.set_index("tid", inplace=True)
    items.sort_index(inplace=True)

    items = items[: max_item_index_to_save + 1]

    return users, items


# this method may cause trouble, like if your data is 1-based, or other case.
def save_results(
    output_dir,
    save_npy,
    save_text,
    save_binary,
    save_sequential,
    save_io_workers,
    users,
    items,
    user_embeddings,
    item_embeddings,
    idfs,
    item_sums,
    item_labels=None,
    item_weights=None,
):
    logger.info("In save_results")
    logger.info(
        "if program crashes in save_results, set save_sequential to True or decrease save_io_workers. Currently set as save_sequential: "
        + str(save_sequential)
        + "  and save_io_workers: "
        + str(save_io_workers)
    )

    if not os.path.exists(output_dir):
        logger.info("creating output directory %s", output_dir)
        os.makedirs(output_dir)
    else:
        logger.info("%s exists", output_dir)

    # save user vector (and idf)
    logger.info("saving user vectors")

    user_labels = users.index
    if len(user_labels) != user_embeddings.shape[0]:
        logger.warning(
            "Warning: len(user_labels) != user_embeddings.shape[0]. The train users might not consider all the users and saved embeddings may be wrong! "
            + str(len(user_labels))
            + "!="
            + str(len(user_embeddings))
        )

    user_formatter = RecordFormatter(
        [
            ["char", 32],
            ["float", user_embeddings.shape[1]],
            ["double", 1],
        ]
    )
    user_rec_io = RecordIO(user_formatter)

    if save_npy is True:
        npy_dir = os.path.join(output_dir, "npy")
        if not os.path.exists(os.path.join(output_dir, npy_dir)):
            os.makedirs(npy_dir)
        with open(os.path.join(npy_dir, "userids.npy"), "wb") as f:
            np.save(f, users["UserId"])
        with open(os.path.join(npy_dir, "uservector.npy"), "wb") as f:
            np.save(f, user_embeddings)
        with open(os.path.join(npy_dir, "useridfs.npy"), "wb") as f:
            np.save(f, idfs)
        logger.info("Saved user embeddings in npy format")

    if save_text is True:
        if save_sequential is False:
            user_text_args = [
                os.path.join(output_dir, "uservector.tsv"),
                save_io_workers,
                users["UserId"],
                user_embeddings,
                idfs,
            ]
        else:
            user_text_args = [os.path.join(output_dir, "uservector.tsv"), users["UserId"], user_embeddings, idfs]

        text_writer = user_rec_io.write_text_in_parallel if save_sequential is False else user_rec_io.write_text
        text_writer(*user_text_args)
        logger.info("Saved user embeddings in text format. save_sequential is " + str(save_sequential))

    if save_binary is True:
        if save_sequential is False:
            user_binary_args = [
                os.path.join(output_dir, "uservector.bin"),
                save_io_workers,
                users["UserId"],
                user_embeddings,
                idfs,
            ]
        else:
            user_binary_args = [os.path.join(output_dir, "uservector.bin"), users["UserId"], user_embeddings, idfs]

        binary_writer = user_rec_io.write_binary_in_parallel if save_sequential is False else user_rec_io.write_binary
        binary_writer(*user_binary_args)
        logger.info("Saved user embeddings in binary format. save_sequential is " + str(save_sequential))

    # save item vector (and item_sum)
    logger.info("saving item vectors")

    if item_labels is None:
        item_labels = items.index

    item_formatter = RecordFormatter(
        [
            ["char", 32],
            ["float", item_embeddings.shape[1]],
            ["double", 1],
        ]
    )
    if item_weights is not None:
        item_formatter.append(["float", 1])
    item_rec_io = RecordIO(item_formatter)

    if save_npy is True:
        npy_dir = os.path.join(output_dir, "npy")
        if not os.path.exists(os.path.join(output_dir, npy_dir)):
            os.makedirs(npy_dir)
        with open(os.path.join(npy_dir, "itemids.npy"), "wb") as f:
            np.save(f, items["TargetId"][item_labels])
        with open(os.path.join(npy_dir, "itemvector.npy"), "wb") as f:
            np.save(f, item_embeddings[item_labels])
        with open(os.path.join(npy_dir, "itemidfs.npy"), "wb") as f:
            np.save(f, item_sums[item_labels])
        if item_weights is not None:
            with open(os.path.join(npy_dir, "itemweights.npy"), "wb") as f:
                np.save(f, item_weights[item_labels])

        logger.info("Saved item embeddings in npy format")

    if save_text is True:
        if save_sequential is False:
            item_text_args = [
                os.path.join(output_dir, "itemvector.tsv"),
                save_io_workers,
                items["TargetId"][item_labels],
                item_embeddings[item_labels],
                item_sums[item_labels],
            ]
        else:
            item_text_args = [
                os.path.join(output_dir, "itemvector.tsv"),
                items["TargetId"][item_labels],
                item_embeddings[item_labels],
                item_sums[item_labels],
            ]
        if item_weights is not None:
            item_text_args.append(item_weights)

        text_writer = item_rec_io.write_text_in_parallel if save_sequential is False else item_rec_io.write_text
        text_writer(*item_text_args)
        logger.info("Saved item embeddings in text format. save_sequential is " + str(save_sequential))

    if save_binary is True:
        if save_sequential is False:
            item_binary_args = [
                os.path.join(output_dir, "itemvector.bin"),
                save_io_workers,
                items["TargetId"][item_labels],
                item_embeddings[item_labels],
                item_sums[item_labels],
            ]
        else:
            item_binary_args = [
                os.path.join(output_dir, "itemvector.bin"),
                items["TargetId"][item_labels],
                item_embeddings[item_labels],
                item_sums[item_labels],
            ]
        if item_weights is not None:
            item_binary_args.append(item_weights)

        binary_writer = item_rec_io.write_binary_in_parallel if save_sequential is False else item_rec_io.write_binary
        binary_writer(*item_binary_args)
        logger.info("Saved item embeddings in binary format. save_sequential is " + str(save_sequential))


def train_wrap(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        params = args[0]

        # print all input arguments
        print_input_args(params)

        # # start mlflow recording
        # if params.mlflow is True:
        #     try:
        #         mlflow.set_tracking_uri(params.mlflow_uri)
        #         mlflow.set_experiment(params.mlflow_experiment)
        #         mlflow.start_run(run_name=params.mlflow_run_name)

        #         log_input_args(params)
        #     except Exception:
        #         logger.exception("failed to start mlflow experiment tracking")

        result = fn(*args, **kwargs)

        # # end mlflow recording
        # if params.mlflow:
        #     try:
        #         mlflow.end_run()
        #     except Exception:
        #         logger.exception("failed to end mlflow experiment tracking")

        return result

    return wrapped


def get_dataframe(filename, delimiter, read_frequency):
    if read_frequency is True:
        df = pd.read_csv(
            filename,
            delimiter=delimiter,
            header=None,
            usecols=range(5),
            names=["uid", "tid", "UserId", "TargetId", "Frequency"],
        )
    else:
        df = pd.read_csv(
            filename, delimiter=delimiter, header=None, usecols=range(4), names=["uid", "tid", "UserId", "TargetId"]
        )
    # df = df.drop_duplicates(keep="first", inplace=False)
    df = df.reset_index(drop=True)
    return df


def get_ratings(args, filename, delimiter):
    raw_ratings, ratings_df = get_dataframe_and_rating(args, filename, delimiter)
    raw_ratings.eliminate_zeros()
    ratings, N, idf, item_sums, average_length = reweight_data(args, raw_ratings)
    ratings.data = ratings.data * args.rating_scale + args.rating_offset
    ratings = ratings.tocsr()
    return ratings, ratings_df, N, idf, item_sums, average_length


def load_test_data(filename, delimiter=",", user_ids=None, item_ids=None, logger_in=None):
    """Load test data"""
    if logger_in:
        global logger
        logger = logger_in
    if filename is not None:
        logger.info("loading test data from %s", filename)
        test_df = get_dataframe(filename, delimiter, read_frequency=False)
        if user_ids is not None and item_ids is not None:
            item_dict = dict(zip(item_ids, range(len(item_ids))))
            user_dict = dict(zip(user_ids, range(len(user_ids))))
            tids = deque()
            uids = deque()
            for i in range(len(test_df["TargetId"])):
                t = test_df["TargetId"][i]
                u = test_df["UserId"][i]
                if t in item_dict and u in user_dict:
                    tids.append(item_dict[t])
                    uids.append(user_dict[u])
        else:
            tids = test_df["tid"].astype(np.int32)
            uids = test_df["uid"].astype(np.int32)
        test_item_users = get_rating_matrix(tids, uids)
        test_user_items = test_item_users.T.tocsr()

        test_users = test_df[["UserId"]]
        test_users = test_users.drop_duplicates(keep="first", inplace=False)

        test_items = test_df[["TargetId"]]
        test_items = test_items.drop_duplicates(keep="first", inplace=False)

        logger.info("test ratings: %s", len(test_df))
        logger.info("test users: %s", len(test_users))
        logger.info("test items: %s", len(test_items))
    else:
        test_user_items = None
        test_df = None

    return test_user_items, test_df


def get_logger_dir(baseDir):
    timestamp = str(int(time.time()))
    outDir = os.path.abspath(os.path.join(baseDir, timestamp))
    loggerDir = os.path.abspath(os.path.join(outDir, "logs"))
    if not os.path.exists(loggerDir):
        os.makedirs(loggerDir)
    return loggerDir


def set_seed(seed, n_gpu):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class Ticker:
    def __init__(self):
        self.start = time.time()
        self.tic = self.start

    def tick(self):
        self.toc = time.time()
        delay = self.toc - self.tic
        self.tic = self.toc
        return "processing one epoch took: %fs " % (delay)

    def stop(self):
        self.end = time.time()
        return "processing the whole input file took: %fs  " % (self.end - self.start)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        print("\n", flush=True)
        self.end = time.time()
        print("processing the whole input file took: %fs  " % (self.end - self.start))


def get_max_item_index_to_save(args, logger, filename):
    stat_dir = os.path.dirname(args.input_path)
    stat_path = os.path.join(stat_dir, filename)
    if os.path.exists(stat_path):
        stat = pd.read_csv(stat_path)
        max_item_index_to_save = stat["DocSize"][0]
    else:
        logger.info(stat_path + " does not exist! Will read max_item_index_to_save from " + args.input_path)
        (
            _,
            train_df,
            _,
            _,
            _,
            _,
        ) = load_train_data(args, args.input_path, args.input_delimiter, logger)
        max_item_index_to_save = train_df["tid"].max()

    logger.info("max_item_index_to_save: %f", max_item_index_to_save)

    return max_item_index_to_save


class DocEmbeddings:
    def __init__(
        self,
        filename,
        normalized_factor,
        delimiter="\t",
        embedding_delimiter=",",
        max_item_index_to_load=None,
        item_labels=None,
        logger_in=None,
    ):
        """
        :param filename: path to visual or text embedding, containing columns. The path should have two columns named tid, Embedding.
        :param delimiter: delimiter used to separate columns.
        :param embedding_delimiter: delimiter used in embedding.
        :param a: scalar used as denominator for normalizing embeddings. a together with b is used as 'embedding / a - b'
        :param b: scalar used as offset for normalizing embedding.
        :param max_item_index_to_load: used in load_doc_embeddings function.  return visual/text embeddings for tid in [0 : max_item_index_to_load + 1). if None, return all embeddings
        :param logger_in: logger.
        """
        self.filename = filename
        self.normalized_factor = normalized_factor
        self.delimiter = delimiter  # seperator for different columns in embedding file
        self.embedding_delimiter = embedding_delimiter  # seperator for embedding elements
        self.max_item_index_to_load = max_item_index_to_load
        self.item_labels = item_labels
        self.logger_in = logger_in

        self.doc_embeddings = self.load_doc_embeddings()

    def load_doc_embeddings(self):
        """
        :return: a numpy array of unevaluated doc embeddings, that is string, with shape of (max_item_index_to_load + 1, )
        """
        if self.logger_in:
            global logger
            logger = self.logger_in

        if not os.path.exists(self.filename):
            logger.error(self.filename + "does not exist")
            return None

        logger.info("loading doc_embeddings from %s", self.filename)
        df = pd.read_csv(
            self.filename, delimiter=self.delimiter, header=0, usecols=[1], names=["Embedding"]
        )  # should already be sorted based on tid and no duplicates (based on tid) should exist
        if self.max_item_index_to_load is not None:
            df = df.head(
                self.max_item_index_to_load + 1
            )  # Only getting embeddings from prism used for recommendation. Also need this line to remove nan values before calling np.array(doc_embeddings)
        doc_embeddings = np.array(df["Embedding"])
        # If you want to evaluate doc embeddings here, comment above line and uncomment the next 3 lines
        # df["Embedding"] = df["Embedding"].apply(lambda s: [float(x) for x in s.split(self.embedding_delimiter)])
        # doc_embeddings = np.array(list(df["Embedding"]))
        # doc_embeddings = doc_embeddings / self.a - self.b

        if self.item_labels is not None:
            doc_embeddings = doc_embeddings[self.item_labels]

        return doc_embeddings

    def get_eval_embedding(self, indices):
        """
        :doc_embeddings: a 1-dimensional numpy array of doc_embeddings
        :indices: either a list of indices or (e.g., 1-dimensional corresponding to only 1 user) a list of list of indices (e.g., 2-dimensional corresponding to more than 1 user)
        :return: a list in which doc_embedding have been evaluated and normalized and correspond to indices
        """

        def replace(s):
            return s.replace(" ", ",")

        if self.doc_embeddings is None:
            return None
        same_size = len(set(map(len, indices))) == 1
        indices = np.array(indices)
        if len(indices.shape) == 2 and indices.shape[1] != 0:
            vec = self.doc_embeddings[indices]  # vec.shape: indices.shape[0] * indices.shape[1]
            vec = vec.flatten()  # vec.shape: indices.shape[0] * indices.shape[1]
        elif len(indices.shape) == 2 and indices.shape[1] == 0:
            vec = [np.array([]) for _ in range(indices.shape[0])]
            return vec
        elif len(indices.shape) == 1 and same_size:
            vec = self.doc_embeddings[indices]
        elif len(indices.shape) == 1 and not same_size:
            vec = []
            sizes = []
            for i in range(indices.shape[0]):
                vec += list(self.doc_embeddings[indices[i]])
                sizes.append(indices[i].shape[0])

        vec = list(vec)
        if self.embedding_delimiter == " ":
            vec = list(map(replace, vec))
        vec = list(map(ast.literal_eval, vec))  # vec.shape: indices.shape[0] * indices.shape[1] *  embedding_size
        vec = np.array(vec)
        vec = vec * self.normalized_factor["factor"] - self.normalized_factor["scale"]
        vec = torch.tensor(vec)
        vec = F.normalize(vec, p=2, dim=-1)
        vec = np.array(vec)
        if len(indices.shape) == 2 and indices.shape[1] != 0:
            vec = vec.reshape(
                indices.shape[0], indices.shape[1], -1
            )  # vec.shape: indices.shape[0], indices.shape[1], embedding_size
            vec = list(vec)
        elif len(indices.shape) == 1 and same_size:
            vec = list(vec)
        elif len(indices.shape) == 1 and not same_size:
            vec = np.array(vec)
            start = 0
            new_vec = []
            for i in range(len(sizes)):
                new_vec.append(vec[start : start + sizes[i]])
                start += sizes[i]
            vec = new_vec
        return vec


def get_user_results_from_i2imatrix(
    i2imatrix_csr,
    train_user_item_csr,
    users,
    max_item_index_to_save=None,
    clicks_limit=None,
    recom_per_click_limit=50,
    k=100,
):
    predictions = []
    distances = []
    num_users, num_items = train_user_item_csr.shape

    items = defaultdict(list)
    train_user_item_coo = train_user_item_csr.tocoo()
    num_data = len(train_user_item_coo.data)
    for n in range(num_data):
        items[train_user_item_coo.row[n]].append(train_user_item_coo.col[n])

    # sort items[u] based on train_user_item[i,j] from largest to smallest
    if clicks_limit is not None:
        for u in range(num_users):
            score_items = []
            for i in items[u]:
                score_items.append(train_user_item_csr[u, i])
            items[u] = [x for _, x in sorted(zip(score_items, items[u]), reverse=True)]

    sim_items = defaultdict(list)
    i2imatrix_coo = i2imatrix_csr.tocoo()
    num_data = len(i2imatrix_coo.data)
    for n in range(num_data):
        sim_items[i2imatrix_coo.row[n]].append(i2imatrix_coo.col[n])

    # filtering sim_items to prism index
    # sort sim_items[i] based on i2imatrix_csr[i,j] from largest to smallest
    for i in range(num_items):
        sim_items[i] = np.array(sim_items[i])
        sim_items[i] = sim_items[i][sim_items[i] < (max_item_index_to_save + 1)]
        score_items = []
        for j in sim_items[i]:
            score_items.append(i2imatrix_csr[i, j])
        sim_items[i] = [x for _, x in sorted(zip(score_items, sim_items[i]), reverse=True)]

    for u in users:
        scores = defaultdict(int)
        for p_i, i in enumerate(items[u]):
            if clicks_limit is not None and p_i >= clicks_limit:
                break
            for p_j, j in enumerate(sim_items[i]):
                if recom_per_click_limit is not None and p_j >= recom_per_click_limit:
                    break
                scores[j] -= i2imatrix_csr[
                    i, j
                ]  # The best match should have smallest distance, therefore consider negative sim
        predictions.append(sorted(scores, key=lambda item: scores[item])[:k])
        distances.append(sorted(scores.values())[:k])
    results = [predictions, distances]
    return results
