#!/usr/bin/python3
import functools
import logging
import math
import os
import pickle
import shutil
import sys
import time
from distutils.dist import strtobool

import numpy as np
import pandas as pd
import psutil
from scipy.sparse import coo_matrix

from util.record_io import RecordFormatter, RecordIO


logger = logging.getLogger(__name__)


def str2bool(s):
    return bool(strtobool(s))


def timer(func):
    """A decorator to time a function call."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {elapsed:.4f} seconds")
        return result

    return wrapper





def log_config_file(config_path: str):
    """Prints a configuration file to appear in a log.

    Note: prints to STDOUT without the normal logging interface. This allows the
    config to be copied verbatim and used without having to strip log formatting.

    Args:
        config_path (str): Path to the config file to print.
    """

    # Flushes to help avoid config interleaving with other log outputs
    sys.stdout.flush()
    sys.stderr.flush()
    logger.info(f"Config {config_path}")
    with open(config_path, "r") as fin:
        for line in fin:
            # not using logging so config can be copied and used
            print(line, end="")

    sys.stdout.flush()
    sys.stderr.flush()


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


def append_extra_dim_for_l2(embeddings, append_zero=True):
    """Append extra dim to let Euclidean distance (L2) simulate inner product."""
    if append_zero:
        embeddings = np.array([np.append(e, [0.0]) for e in embeddings])
    else:
        max_item_len = max([np.linalg.norm(e) for e in embeddings])
        embeddings = np.array(
            [np.append(e, [math.sqrt(pow(max_item_len, 2) - pow(np.linalg.norm(e), 2))]) for e in embeddings]
        )
    return embeddings


def get_dataframe(filename, delimiter):
    # read only five rows to get the number of columns
    df_small = pd.read_csv(filename, delimiter=delimiter, header=None, nrows=5)
    num_cols = df_small.shape[1]

    assert num_cols in (2, 3, 4)  # support two-column, three-column, and four-column versions of input data

    if num_cols == 2:  # only two columns in input data: (userId, itemId)
        df = pd.read_csv(
            filename,
            delimiter=delimiter,
            header=None,
            usecols=range(2),
            names=["uid", "tid"],
            dtype={"uid": int, "tid": int},
        )
    elif num_cols == 3:
        df = pd.read_csv(
            filename,
            delimiter=delimiter,
            header=None,
            usecols=range(3),
            names=["uid", "tid", "ctr"],
            dtype={"uid": int, "tid": int, "ctr": float},
        )
    else:
        df = pd.read_csv(
            filename,
            delimiter=delimiter,
            header=None,
            usecols=range(4),
            names=["uid", "tid", "userId", "itemId"],
            dtype={"uid": int, "tid": int, "userId": str, "itemId": str},
        )

    logger.info("dataframe loaded from %s", filename)
    # input should be distinct, when data is big, this drop_duplicates will stuck
    # df = df.drop_duplicates(keep="first", inplace=False)
    logger.info("dataframe drop duplicates done")
    df = df.reset_index(drop=True)
    logger.info("dataframe reset index done")
    return df


def get_rating_matrix(tids, uids):
    """get the coo format item-user concurrence matrix"""
    data = np.ones(len(uids)).astype(np.float32)
    rating = coo_matrix((data, (tids, uids)))
    logger.info("get ratings csr matrix done")
    return rating


def get_dataframe_and_rating(filename, delimiter):
    df = get_dataframe(filename, delimiter)
    rating = get_rating_matrix(df["tid"].astype(np.int32), df["uid"].astype(np.int32))
    return rating, df


def bm25_weight_local(X, K1=100, B=0.8):
    N = float(X.shape[0])
    idf = np.log(N) - np.log1p(np.bincount(X.col))
    # calculate length_norm per document (artist)
    item_sums = np.bincount(X.row)
    average_length = item_sums.mean()
    length_norm = (1.0 - B) + B * item_sums / average_length
    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X, N, idf, item_sums, average_length


def binary_weight(X):
    N = float(X.shape[0])
    idf = np.log(N) - np.log1p(np.bincount(X.col))
    # calculate length_norm per document (artist)
    item_sums = np.ravel(X.sum(axis=1))
    average_length = item_sums.mean()

    return X, N, idf, item_sums, average_length


def get_ratings(filename, delimiter, rating_scale, rating_offset, weight_choice=None):
    raw_ratings, ratings_df = get_dataframe_and_rating(filename, delimiter)
    raw_ratings.eliminate_zeros()
    logger.info("raw ratings eliminate zeros done")
    raw_ratings.data = np.ones(len(raw_ratings.data))

    if weight_choice == "bm25":
        ratings, N, idf, item_sums, average_length = bm25_weight_local(raw_ratings, B=0.9)
    elif weight_choice == "ctr":
        raw_ratings.data = np.array(ratings_df["ctr"].tolist(), dtype=np.float32)
        ratings, N, idf, item_sums, average_length = binary_weight(raw_ratings)
    else:
        ratings, N, idf, item_sums, average_length = binary_weight(raw_ratings)
        logger.info("binary weight done")

    ratings.data = ratings.data * rating_scale + rating_offset
    logger.info("ratings to csr done")

    return ratings, ratings_df, N, idf, item_sums, average_length


def load_train_data(filename, delimiter=",", rating_scale=5, rating_offset=0, weight_choice=None):
    """Load training data and return csr matrix."""
    logger.info("loading data from %s", filename)
    train_item_user_coo, train_df, N, idfs, item_sums, average_length = get_ratings(
        filename, delimiter, rating_scale, rating_offset, weight_choice
    )
    train_user_item_coo = train_item_user_coo.T
    train_user_item_csr = train_user_item_coo.tocsr()
    logger.info("train_user_items csr get")

    num_ratings = len(train_df)
    num_users, num_items = train_user_item_csr.shape
    density = num_ratings / (num_users * num_items) * 100
    logger.info("train ratings: %s", num_ratings)
    logger.info("train users: %s", num_users)
    logger.info("train items: %s", num_items)
    logger.info("train matrix density: %.6f %%", density)

    return train_user_item_csr, train_df, N, idfs, item_sums, average_length


def load_train_data_with_coo(filename, delimiter=",", rating_scale=5, rating_offset=0, weight_choice=None):
    """Load training data and return COO and CSR matrices."""
    logger.info("loading data from %s", filename)
    train_item_user_coo, train_df, N, idfs, item_sums, average_length = get_ratings(
        filename, delimiter, rating_scale, rating_offset, weight_choice
    )
    train_user_item_coo = train_item_user_coo.T
    train_user_item_csr = train_user_item_coo.tocsr()
    logger.info("train_user_items csr get")

    num_ratings = len(train_df)
    num_users, num_items = train_user_item_csr.shape
    density = num_ratings / (num_users * num_items) * 100
    logger.info("train ratings: %s", num_ratings)
    logger.info("train users: %s", num_users)
    logger.info("train items: %s", num_items)
    logger.info("train matrix density: %.6f %%", density)

    return train_user_item_coo, train_user_item_csr, train_df, N, idfs, item_sums, average_length


def load_test_data(filename, delimiter=",", user_ids=None, item_ids=None):
    """Load test data and return the user_item csr matrix."""
    if filename is not None:
        logger.info("loading test data from %s", filename)
        test_df = get_dataframe(filename, delimiter)

        tids = test_df["tid"].astype(np.int32)
        uids = test_df["uid"].astype(np.int32)
        test_item_user_coo = get_rating_matrix(tids, uids)
        test_user_item_csr = test_item_user_coo.T.tocsr()

        test_users = test_df[["uid"]]
        test_users = test_users.drop_duplicates(keep="first", inplace=False)

        test_items = test_df[["tid"]]
        test_items = test_items.drop_duplicates(keep="first", inplace=False)

        logger.info("test ratings: %s", len(test_df))
        logger.info("test users: %s", len(test_users))
        logger.info("test items: %s", len(test_items))
    else:
        test_user_item_csr = None
        test_df = None

    return test_user_item_csr, test_df


def load_target_items(filename, train_df):
    if filename:
        logger.info("loading target items from %s", filename)
        target_items = pd.read_csv(
            filename,
            delimiter=",",
            header=None,
            # names=["TargetId"],dtype={'TargetId':str})
            names=["tid"],
            dtype={"tid": int},
        )
        # items = train_df[["TargetId", "tid"]]
        # items = items.drop_duplicates(keep="first", inplace=False)
        # target_df = items[items["TargetId"].isin(target_items["TargetId"])]
        # item_labels = target_df["tid"].tolist()
        target_items = target_items.drop_duplicates(keep="first", inplace=False)
        item_labels = target_items["tid"].tolist()
    else:
        item_labels = None

    return item_labels



