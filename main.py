#!/usr/bin/python3
import logging
import math
import os
import sys
from collections import defaultdict

import numpy as np
import splatt
import torch
from scipy.sparse import coo_matrix, csr_matrix


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.args import init_arguments  # noqa: E402
from util.evaluation import evaluate  # noqa: E402
from util.helper import (  # noqa: E402
    get_logger_dir,
    get_max_item_index_to_save,
    load_train_data,
    load_train_data_weighted_by_time,
    train_wrap,
)
from util.logger import setup_logger  # noqa: E402
from util.metrics import matrix_query  # noqa: E402


logger = logging.getLogger(__name__)


def save_matrix_predictions(args, i2imatrix_csr, input_path, filename, max_item_index_to_save=None, logger_in=None):

    if logger_in:
        global logger
        logger = logger_in

    logger.info("Started saving matrix predictions for all users")

    # load training dataset
    if args.load_training_method == "load_train_data":
        train_user_items, train_df, _, _, _, _ = load_train_data(
            args, input_path, args.input_delimiter, logger_in=logger
        )
    elif args.load_training_method == "load_train_data_weighted_by_time":
        train_user_items, train_df, _, _, _, _ = load_train_data_weighted_by_time(
            args, input_path, args.input_delimiter, logger_in=logger
        )

    users = np.unique(coo_matrix(train_user_items).row)

    m = train_user_items[users].tocsr()
    users_liked = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    preds, _ = matrix_query(
        users,
        users_liked,
        i2imatrix_csr,
        train_user_items,
        max_item_index_to_save=max_item_index_to_save,
        k=args.user_top_k,
        reco_cutoff_score=args.reco_cutoff_score,
    )

    user_id_map_from_train = {}
    doc_id_map_from_train = {}
    file = os.path.join(args.output_dir, filename)

    for user, uid, doc, did in zip(train_df["UserId"], train_df["uid"], train_df["TargetId"], train_df["tid"]):
        user_id_map_from_train[uid] = user
        doc_id_map_from_train[did] = doc

    with open(file, "w") as fw:
        for uid in range(len(preds)):
            user = user_id_map_from_train[uid]
            docIds = preds[uid]
            if len(docIds) == 0:
                continue
            doc_list = list(map(doc_id_map_from_train.get, docIds))
            line = user + "\t" + ";".join(doc_list) + "\n"
            fw.write(line)

    logger.info("Finished saving matrix predictions for all users")


def get_swing_i2imatrix_csr(train_user_item_csr, alpha=0.001):

    num_users, num_items = train_user_item_csr.shape
    weight = np.zeros(num_users)
    items = defaultdict(set)
    users = defaultdict(list)

    train_user_item_coo = train_user_item_csr.tocoo()
    num_data = len(train_user_item_coo.data)
    for n in range(num_data):
        items[train_user_item_coo.row[n]].add(train_user_item_coo.col[n])
        users[train_user_item_coo.col[n]].append(train_user_item_coo.row[n])

    for u in range(num_users):
        weight[u] = 1 / math.sqrt(len(items[u]))

    swing_dict = defaultdict(int)
    for i in range(num_items):
        for p, u in enumerate(users[i]):
            for v in users[i][p + 1 :]:
                common_items_uv = list(items[u] & items[v])
                k = len(common_items_uv)
                for j in common_items_uv:
                    swing = weight[u] * weight[v] * 1 / (alpha + k)
                    swing_dict[i, j] += swing
                    swing_dict[j, i] += swing
    keys = list(swing_dict.keys())
    rows, cols = list(zip(*keys))
    data = list(swing_dict.values())
    swing_i2imatrix_csr = csr_matrix((data, (rows, cols)), shape=(num_items, num_items))

    return swing_i2imatrix_csr


@train_wrap
def train(args, logger):
    logger.info(str(args))
    logger.info("Create i2i matrix .....")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load training dataset
    if args.expand_user_coverage:
        input_path = args.input_path_expandusercoverage
    else:
        input_path = args.input_path
    if args.load_training_method == "load_train_data":
        train_user_item_csr, _, _, _, _, _ = load_train_data(args, input_path, args.input_delimiter, logger_in=logger)
    elif args.load_training_method == "load_train_data_weighted_by_time":
        train_user_item_csr, _, _, _, _, _ = load_train_data_weighted_by_time(
            args, input_path, args.input_delimiter, logger_in=logger
        )

    # Can initialize i2i matrix with different methods
    i2imatrix_csr = get_swing_i2imatrix_csr(train_user_item_csr)

    return i2imatrix_csr


def main():
    args = init_arguments()

    if args.save_log_to_file is True:
        log_dir = get_logger_dir(args.output_dir)
    else:
        log_dir = None
        # Removing log_dir to disable logging for AML machines since streaming for azureblob storage is taking longer than itp nfs storage
    logger = setup_logger("bprtrain", log_dir, 0)

    max_item_index_to_save = get_max_item_index_to_save(args, logger, "cf_stat.csv")

    if not args.evaluation_only:
        i2imatrix_csr = train(args, logger)

    if args.expand_user_coverage is True:
        input_path = args.input_path_expandusercoverage
    else:
        input_path = args.input_path

    if args.do_evaluation:

        index = evaluate(
            args,
            test_path=args.test_path,
            input_path=input_path,
            user_type="prism",
            filename="user_doc_preds.tsv",
            index=None,
            max_item_index_to_save=max_item_index_to_save,
            logger_in=logger,
            i2imatrix_csr=i2imatrix_csr,
        )

        if args.expand_user_coverage is True:
            if os.path.exists(args.test_path_uniqueextra) and os.stat(args.test_path_uniqueextra).st_size != 0:
                # create user_doc_preds_for unique extra (non-prism) users
                evaluate(
                    args,
                    test_path=args.test_path_uniqueextra,
                    input_path=input_path,
                    user_type="extra",
                    filename="user_doc_preds_for_extrausers.tsv",
                    index=index,
                    max_item_index_to_save=max_item_index_to_save,
                    logger_in=logger,
                    i2imatrix_csr=i2imatrix_csr,
                )

        if os.path.exists(args.test_path_uniquespecific) and os.stat(args.test_path_uniquespecific).st_size != 0:
            # create user_doc_preds_for specific users
            evaluate(
                args,
                test_path=args.test_path_uniquespecific,
                input_path=input_path,
                user_type="specific",
                filename="user_doc_preds_for_specificusers.tsv",
                index=index,
                max_item_index_to_save=max_item_index_to_save,
                logger_in=logger,
                i2imatrix_csr=i2imatrix_csr,
            )

    save_matrix_predictions(
        args=args,
        i2imatrix_csr=i2imatrix_csr,
        input_path=input_path,
        filename="matrix_predictions_for_allusers.tsv",
        max_item_index_to_save=max_item_index_to_save,
        logger_in=logger,
    )


if __name__ == "__main__":
    print("Have to import splatt to avoid bpr env errors. splatt version: ", splatt.__version__)
    main()
