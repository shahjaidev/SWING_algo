"""
This script produces a file for visualizing in https://odinvis.azurewebsites.net/cosmos
Special thanks to Dongfei Yu for providing us with the base visualization script in which this script is adapted from
"""

import argparse
import codecs
import copy
import pickle
import random
import sys
import timeit
from collections import defaultdict
from datetime import datetime
from os import path
from pathlib import Path

import pandas as pd


def load_docid_to_murl(docid_to_murl_filename, delimiter="\t"):
    # docid_to_murl_filename should contain mapping from DocId to MUrl
    df = pd.read_csv(docid_to_murl_filename, delimiter=delimiter, header=0, usecols=range(2), names=["DocId", "MUrl"])
    start_time = timeit.default_timer()
    df = df.drop_duplicates(subset="DocId", keep="first", inplace=False)
    print("time to drop duplicates from doc_to_murl", timeit.default_timer() - start_time)
    start_time = timeit.default_timer()
    doc2murl = dict(zip(df.DocId, df.MUrl))
    print("time to make doc2murl dictionary", timeit.default_timer() - start_time)

    return doc2murl


def load_user_history_and_gt_label(data_file, pickle_name_userhistory=None, pickle_name_gtlabel=None):
    """
    Output Schema: {UserId:[(query, murl, timestamp, flag)]}
    """

    if (
        pickle_name_userhistory is not None
        and pickle_name_gtlabel is not None
        and path.exists(pickle_name_userhistory)
        and path.exists(pickle_name_gtlabel)
    ):
        print("loading userhistory and gtlabel pickles")
        start_time = timeit.default_timer()
        with open(pickle_name_userhistory, "rb") as handle:
            user_history_dict = pickle.load(handle)

        with open(pickle_name_gtlabel, "rb") as handle:
            user_gtlabel_dict = pickle.load(handle)

        print("load successful", timeit.default_timer() - start_time)
    else:
        print("creating userhistory and gtlabel pickles")
        user_history_dict = {"prism": defaultdict(list), "extra": defaultdict(list), "specific": defaultdict(list)}
        user_gtlabel_dict = {"prism": defaultdict(list), "extra": defaultdict(list), "specific": defaultdict(list)}
        start_time = timeit.default_timer()
        with codecs.open(data_file, encoding="utf-8") as train_fid:
            line = train_fid.readline()
            for line in train_fid:
                terms = line[:-2].split("\t")
                if len(terms) < 5:
                    continue
                user_id = terms[0]
                murl = terms[1]
                request_time = datetime.strptime(terms[3], "%m/%d/%Y %I:%M:%S %p")
                query = ""
                flag = terms[4]
                time_stamp = request_time.timestamp()
                user_history_dict["prism"][user_id].append((query, murl, time_stamp, flag))

        user_history_dict["extra"] = copy.deepcopy(user_history_dict["prism"])
        user_history_dict["specific"] = copy.deepcopy(user_history_dict["prism"])

        for user_type in ["prism", "extra"]:

            for user_id, history in user_history_dict[user_type].items():
                history = sorted(history, key=lambda x: x[2])
                user_history_dict[user_type][user_id] = history
                pos = None
                for i in range(len(history)):
                    if user_type == "prism" and history[i][3] == "1":
                        pos = i  # find last position where flag == 1
                    elif user_type == "extra":
                        pos = i  # find last position where flag == 0

                if pos is not None:
                    user_gtlabel_dict[user_type][user_id] = history[pos]
                    del history[pos]

        for user_id, history in user_history_dict["specific"].items():
            history = sorted(history, key=lambda x: x[2])
            user_history_dict["specific"][user_id] = history
            user_gtlabel_dict["specific"][user_id] = ("", "", "", "")

        if pickle_name_userhistory is not None and pickle_name_gtlabel is not None:
            with open(pickle_name_gtlabel, "wb") as handle:
                pickle.dump(user_gtlabel_dict, handle)
            with open(pickle_name_userhistory, "wb") as handle:
                pickle.dump(user_history_dict, handle)
            print("dump succesful", timeit.default_timer() - start_time)

    return user_history_dict, user_gtlabel_dict


def load_pred_label(pred_file, n_candidate_limit=15):
    """
    Output Schema: "UserId\tCandidatesImageUrl1|CandidatesImageUrl2\n"
    """
    user_predlabel = dict()
    recommendation_diversity_dict = dict()
    recommendation_relevancy_dict = dict()
    recommendation_soft_recall_dict = dict()
    with codecs.open(pred_file, encoding="utf-8") as pred_fid:
        line_id = 0
        for line in pred_fid:
            terms = line.strip().split("\t")
            if len(terms) < 2:
                print("error format: %d %s" % (line_id + 1, line))
                line_id += 1
                continue
            user_id = terms[0]
            candidates = terms[1].split("|")

            if len(terms) >= 3:
                recommendation_diversity_dict[user_id] = terms[2]
            if len(terms) >= 4:
                recommendation_relevancy_dict[user_id] = terms[3]
            if len(terms) >= 5:
                recommendation_soft_recall_dict[user_id] = terms[4]

            if n_candidate_limit is None:
                user_predlabel[user_id] = candidates
            else:
                user_predlabel[user_id] = candidates[:n_candidate_limit]

            line_id += 1
    return (
        user_predlabel,
        recommendation_diversity_dict,
        recommendation_relevancy_dict,
        recommendation_soft_recall_dict,
    )


def random_sample_users(user_list, n_user=100, pickle_name=None):
    random.seed(1234)
    if pickle_name is not None and path.exists(pickle_name):
        with open(pickle_name, "rb") as handle:
            user_list_sample = pickle.load(handle)
        print("load user successful")
    else:
        user_list_sample = random.sample(user_list, min(n_user, len(user_list)))
        if pickle_name is not None:
            with open(pickle_name, "wb") as handle:
                pickle.dump(user_list_sample, handle)
            print("dump succesful")
    return user_list_sample


def generate_vis_data(
    user_list,
    user_history,
    user_gtlabel,
    user_predlabel,
    doc2murl,
    recommendation_diversity_dict,
    recommendation_relevancy_dict,
    recommendation_soft_recall_dict,
    user_preddistance=None,
    user_predconnection=None,
    has_gt=True,
    keyword="",
    n_history_limit=None,
    n_candidate_limit=None,
):
    """
    Output Schema: "UserId\tQuery\tClickImageUrl\tRequestTime\tGTQuery\tGTImageUrl\tGTRequestTime\tCandidateImageUrl\n"
    """
    vis_data = []
    not_count = 0
    is_count = 0
    print(
        "len(user_list), len(user_history) , len(user_predlabel), len(user_gtlabel):  ",
        len(user_list),
        len(user_history),
        len(user_predlabel),
        len(user_gtlabel),
    )
    for user_id in user_list:
        if (
            keyword is not None and keyword != "" and keyword not in user_id
        ):  # add the filter to only uses the results with specific keywords (used only for location CF for now)
            continue
        if user_id not in user_history or user_id not in user_predlabel or user_id not in user_gtlabel:
            if user_id not in user_history:
                print("not in user_histroy", user_id)
            if user_id not in user_predlabel:
                print("not in user_predlabel", user_id)
            if user_id not in user_gtlabel:
                print("not in user_gtlabel", user_id)
            not_count += 1
            continue
        else:
            is_count += 1
        history = user_history[user_id]
        gt_label = user_gtlabel[user_id]
        pred_docid = user_predlabel[user_id]
        if user_preddistance is not None:
            pred_distance = user_preddistance[user_id]
        if user_predconnection is not None:
            pred_connection = user_predconnection[user_id]
        n_history = (
            len(history) if n_history_limit is None else min(n_history_limit, len(history))
        )  # set an upper limit for # of items shown in user history
        n_candidates = (
            len(pred_docid) if n_candidate_limit is None else min(n_candidate_limit, len(pred_docid))
        )  # set an upper limit for # of items shown in prediction
        max_rows = max(n_history, n_candidates)
        for i in range(max_rows):
            if i == 0 and has_gt is True:
                gt_query, gt_murl, gt_timestamp, gt_flag = gt_label
                gt_request_time = datetime.fromtimestamp(gt_timestamp).strftime("%m/%d/%Y %I:%M:%S %p")
            else:
                gt_query, gt_murl, gt_request_time, gt_flag = "", "", "", ""
            if i == 0 and len(recommendation_diversity_dict) > 0:
                recommendation_diversity = recommendation_diversity_dict[user_id]
            else:
                recommendation_diversity = ""
            if i == 0 and len(recommendation_relevancy_dict) > 0:
                recommendation_relevancy = recommendation_relevancy_dict[user_id]
            else:
                recommendation_relevancy = ""
            if i == 0 and len(recommendation_soft_recall_dict) > 0:
                recommendation_soft_recall = recommendation_soft_recall_dict[user_id]
            else:
                recommendation_soft_recall = ""
            if i < n_history:
                query, click_murl, timestamp, flag = history[i]
                request_time = datetime.fromtimestamp(timestamp).strftime("%m/%d/%Y %I:%M:%S %p")
            else:
                query, click_murl, request_time, flag = "", "", "", ""
            if i < n_candidates:
                candidate_docid = pred_docid[i]
                candidate_murl = doc2murl.get(candidate_docid, "")
                if user_preddistance is not None:
                    candidate_distance = pred_distance[i]
                if user_predconnection is not None:
                    candidate_connection = pred_connection[i]
            else:
                candidate_murl = ""
                candidate_docid = ""
                candidate_distance = ""
                candidate_connection = ""
            if user_preddistance is not None:
                vis_data.append(
                    (
                        user_id,
                        query,
                        click_murl,
                        request_time,
                        flag,
                        gt_query,
                        gt_murl,
                        gt_request_time,
                        gt_flag,
                        recommendation_soft_recall,
                        recommendation_diversity,
                        recommendation_relevancy,
                        candidate_murl,
                        candidate_docid,
                        candidate_distance,
                        candidate_connection,
                    )
                )
            else:
                vis_data.append(
                    (
                        user_id,
                        query,
                        click_murl,
                        request_time,
                        flag,
                        gt_query,
                        gt_murl,
                        gt_request_time,
                        gt_flag,
                        recommendation_soft_recall,
                        recommendation_diversity,
                        recommendation_relevancy,
                        candidate_murl,
                        candidate_docid,
                        candidate_connection,
                    )
                )
    print("is_count:", is_count, "not_count:", not_count)
    return vis_data


def generate_vis_file(vis_data, vis_file, header):
    with codecs.open(vis_file, "w", encoding="utf-8") as vis_fid:
        vis_fid.write(header)
        for i in range(len(vis_data)):
            vis_fid.write("\t".join(vis_data[i]) + "\n")


if __name__ == "__main__":
    # set file path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize-prism", action="store_true", help="make visualization for prism users; these users have GT"
    )
    parser.add_argument(
        "--visualize-extra",
        action="store_true",
        help="make visualization for extra (non-prism) users. these users do not have GT",
    )
    parser.add_argument(
        "--visualize-specific",
        action="store_true",
        help="make visualization for sepcific users. assume these users do not have GT",
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument(
        "--n-candidate-limit",
        type=int,
        default=50,
        required=False,
        help="the parameter to control how many results visualized per user, current default is 50",
    )
    parser.add_argument(
        "--n-history-limit",
        type=int,
        default=None,
        required=False,
        help="the paramter to control how many results visualized per user, currently no limit",
    )
    parser.add_argument(
        "--visualize-all-users",
        action="store_true",
        default=False,
        help="if false, make visualization for a sample of users. If true, visualize all users, usually happens in ",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        required=False,
        help=" keyword to filter whether there are certain locations to be visualize, e.g. 'Seattle'",
    )
    args, _ = parser.parse_known_args()
    # input data file
    data_file = "ActiveUserImageLogForOfflineVisualization.tsv"
    docid_to_murl_filename = "DocIdtoMurl.tsv"

    user_type_list = []
    pred_file_dict = dict()
    pred_distance_file_dict = dict()
    pred_connections_file_dict = dict()
    vis_file_dict = dict()
    n_candidate_limit_dict = dict()
    n_history_limit_dict = dict()
    if args.visualize_prism:
        user_type_list.append("prism")
        # input prediction file
        pred_file_dict["prism"] = "user_doc_preds.tsv"
        # input prediction distance file
        pred_distance_file_dict["prism"] = "user_doc_preds_distances.tsv"
        # number of users clicked for each document
        pred_connections_file_dict["prism"] = "user_doc_preds_connections.tsv"
        # output visualization file
        vis_file_dict["prism"] = "user_doc_preds_visualization.tsv"
        n_candidate_limit_dict["prism"] = args.n_candidate_limit
        n_history_limit_dict["prism"] = args.n_history_limit
    else:
        print("--visualize-prism is not set! Will not generate visualization for prism users!")

    if args.visualize_extra:
        user_type_list.append("extra")
        # input prediction file
        pred_file_dict["extra"] = "user_doc_preds_for_extrausers.tsv"
        # input prediction distance file
        pred_distance_file_dict["extra"] = "user_doc_preds_for_extrausers_distances.tsv"
        # number of users clicked for each document
        pred_connections_file_dict["extra"] = "user_doc_preds_for_extrausers_connections.tsv"
        # output visualization file
        vis_file_dict["extra"] = "user_doc_preds_for_extrausers_visualization.tsv"
        n_candidate_limit_dict["extra"] = args.n_candidate_limit
        n_history_limit_dict["extra"] = args.n_history_limit
    else:
        print("--visualize-extra is not set! Will not generate visualization for extra users!")

    if args.visualize_specific:
        user_type_list.append("specific")
        # input prediction file
        pred_file_dict["specific"] = "user_doc_preds_for_specificusers.tsv"
        # input prediction distance file
        pred_distance_file_dict["specific"] = "user_doc_preds_for_specificusers_distances.tsv"
        # number of users clicked for each document
        pred_connections_file_dict["specific"] = "user_doc_preds_for_specificusers_connections.tsv"
        # output visualization file
        vis_file_dict["specific"] = "user_doc_preds_for_specificusers_visualization.tsv"
        n_candidate_limit_dict["specific"] = None  # No limit
        n_history_limit_dict["specific"] = None  # No limit
    else:
        print("--visualize-specific is not set! Will not generate visualization for specific users!")

    remove_list = []
    for user_type in user_type_list:
        print("user_type ", user_type)
        print("n_history_limit ", n_history_limit_dict[user_type])
        print("n_candidate_limit ", n_candidate_limit_dict[user_type])
        print("data_dir ", args.data_dir)
        print("result_dir ", args.result_dir)
        print("data_file is ", data_file)
        print("pred_file is ", pred_file_dict[user_type])
        print("pred_distance_file is ", pred_distance_file_dict[user_type])
        print("vis_file is ", vis_file_dict[user_type])
        print("filtering keyword is ", args.keyword)
        # check file exists before running code
        if not Path(path.join(args.data_dir, data_file)).is_file():
            raise ValueError(path.join(args.data_dir, data_file), " does not exist")
        if not Path(path.join(args.result_dir, pred_file_dict[user_type])).is_file():
            print(
                path.join(args.result_dir, pred_file_dict[user_type]),
                "does not exist, will not generate visualization for",
                user_type,
                "users",
            )
            remove_list.append(user_type)
        if user_type not in remove_list:
            if not Path(path.join(args.result_dir, pred_distance_file_dict[user_type])).is_file():
                raise ValueError(path.join(args.result_dir, pred_distance_file_dict[user_type]), " does not exist")
        if user_type not in remove_list:
            if not Path(path.join(args.result_dir, pred_connections_file_dict[user_type])).is_file():
                raise ValueError(path.join(args.result_dir, pred_connections_file_dict[user_type]), " does not exist")
        print("\n")
    for user_type in remove_list:
        user_type_list.remove(user_type)

    if len(user_type_list) == 0:
        print("Could not find any valid user_doc_preds, exiting visualization")
        sys.exit(0)

    print("Will generate visualization for ", user_type_list)

    doc2murl = load_docid_to_murl(path.join(args.data_dir, docid_to_murl_filename), delimiter="\t")
    user_history_dict, user_gtlabel_dict = load_user_history_and_gt_label(path.join(args.data_dir, data_file))

    for user_type in user_type_list:
        if user_type == "prism" or user_type == "extra":
            has_gt = True
        elif user_type == "specific":
            has_gt = False
        else:
            raise ValueError("has_gt not set")
        pred_file = pred_file_dict[user_type]
        pred_distance_file = pred_distance_file_dict[user_type]
        pred_connections_file = pred_connections_file_dict[user_type]
        vis_file = vis_file_dict[user_type]
        user_history = user_history_dict[user_type]
        user_gtlabel = user_gtlabel_dict[user_type]

        # load prediction file
        (
            user_predlabel,
            recommendation_diversity_dict,
            recommendation_relevancy_dict,
            recommendation_soft_recall_dict,
        ) = load_pred_label(path.join(args.result_dir, pred_file), n_candidate_limit=n_candidate_limit_dict[user_type])

        # load prediction distance file
        user_preddistance, _, _, _ = load_pred_label(
            path.join(args.result_dir, pred_distance_file), n_candidate_limit=n_candidate_limit_dict[user_type]
        )

        # load connections file
        user_predconnection, _, _, _ = load_pred_label(
            path.join(args.result_dir, pred_connections_file), n_candidate_limit=n_candidate_limit_dict[user_type]
        )

        # random select user list
        user_list_all = user_predlabel.keys()
        if args.visualize_all_users:  # by dafault we will set this as false and visualize a sample of users
            user_list_sample = user_list_all
        else:
            user_list_sample = random_sample_users(user_list_all)

        # generate visualization file
        vis_data = generate_vis_data(
            user_list_sample,
            user_history,
            user_gtlabel,
            user_predlabel,
            doc2murl,
            recommendation_diversity_dict,
            recommendation_relevancy_dict,
            recommendation_soft_recall_dict,
            user_preddistance,
            user_predconnection,
            has_gt=has_gt,
            keyword=args.keyword,
            n_history_limit=n_history_limit_dict[user_type],
            n_candidate_limit=n_candidate_limit_dict[user_type],
        )
        header = "UserId\tQuery\tClickedImage\tRequestTime\tFlag\tGTQuery\tGTImage\tGTRequestTime\tGTFlag\tSoftRecall\tRecommendationDiversity\tRecommendationRelevancy\tCFImageRecommendation\tCFImageRecommendation-DocId\tCFImageRecommendationDistance\tnumConnections\n"

        generate_vis_file(vis_data, path.join(args.result_dir, vis_file), header)
