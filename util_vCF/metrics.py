import gc
import itertools
import logging
import multiprocessing
import time
from datetime import datetime

import faiss
import hnswlib
import numpy as np
import torch
import torch.nn as nn

from util.helper import append_extra_dim_for_l2, log_memory_usage


logger = logging.getLogger(__name__)


def correct_at_k(predictions, ground_truths, k):
    correct = []
    for i, prediction in enumerate(predictions):
        intersection = set(ground_truths[i]).intersection(set(prediction[:k]))
        correct.append(len(intersection))
    return correct


# def recall_at_k(predictions, ground_truths, k):
#     correct = correct_at_k(predictions, ground_truths, k)
#     return correct / sum([len(gt) for gt in ground_truths])


def recall_at_k_new(predictions, ground_truths, k, weight):
    correct = correct_at_k(predictions, ground_truths, k)
    ideal = [len(gt) if len(gt) < k else k for gt in ground_truths]

    correct, ideal, weight = np.asarray(correct), np.asarray(ideal), np.asarray(weight)
    correct = correct[ideal != 0]
    weight = weight[ideal != 0]
    ideal = ideal[ideal != 0]

    recall, avg_recall, weighted_recall = 0.0, 0.0, 0.0

    try:
        recall = np.sum(correct) / np.sum(ideal)
    except Exception as e:
        logger.error(f"recall computation error {repr(e)}")
        recall = 0.0

    try:
        recall_per_user = correct / ideal
        avg_recall = np.sum(recall_per_user) / len(recall_per_user)
    except Exception as e:
        logger.error(f"average recall computation error {repr(e)}")
        avg_recall = 0.0

    weight = np.asarray(weight)
    try:
        weighted_recall = np.sum(correct * weight / ideal) / np.sum(weight)
    except Exception as e:
        logger.error(f"weighted recall computation error {repr(e)}")
        weighted_recall = 0.0

    return recall, avg_recall, weighted_recall


def precision_at_k(predictions, ground_truths, k):
    correct = correct_at_k(predictions, ground_truths, k)
    return correct / sum([len(prediction[:k]) for prediction in predictions])


def ndcg_at_k(pred_labels, ground_truths, k):
    pred = pred_labels[:, :k]
    gt = np.zeros((len(pred), k))
    for i, items in enumerate(ground_truths):
        length = k if k <= len(items) else len(items)
        gt[i, :length] = 1
    idcg = np.sum(gt * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(pred * 1.0 / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0.0] = -1.0
    ndcg = dcg / idcg
    ndcg[ndcg < 0.0] = 0.0
    return np.mean(ndcg)


def mrr_at_k(pred_labels, k):
    reciprocal = 1.0 / (np.arange(k) + 1)
    pred = pred_labels[:, :k]
    rr = np.sum(pred * reciprocal, axis=1)
    pred_user = np.sum(pred, axis=1)
    pred_user[pred_user == 0] = -1
    rr = rr / pred_user
    rr[rr < 0] = 0
    return np.mean(rr)


def create_ann_index(item_emb, item_labels=None, distance_func="cosine"):
    # Create ANN index
    if distance_func == "l2":
        item_emb = append_extra_dim_for_l2(item_emb, append_zero=False)
    index = hnswlib.Index(space=distance_func, dim=len(item_emb[0]))
    index.init_index(max_elements=int(len(item_emb) * 1.5), ef_construction=400, M=32, random_seed=1234)

    index.set_ef(150)
    index.set_num_threads(multiprocessing.cpu_count())
    if item_labels is None:
        item_labels = range(len(item_emb))
    index.add_items(item_emb, item_labels)

    return index


def faiss_knn_query_complementary(
    item_emb, item_labels, user_emb, users, users_liked, k=50, index=None, distance_func="cosine"
):
    """
    1. create ann index
    2. query results
    3. query complementary results
    """

    starttime = time.time()

    d = item_emb.shape[1]

    query_user_emb = user_emb[users].astype("float32")
    item_emb = item_emb.astype("float32")
    if item_labels is None:
        item_labels = range(len(item_emb))
    item_labels = np.array(item_labels, dtype=int)

    if distance_func == "cosine":
        metric = faiss.METRIC_INNER_PRODUCT
        faiss.normalize_L2(item_emb)
        faiss.normalize_L2(query_user_emb)
        logger.info("faiss metric: cosine")
    elif distance_func == "ip":
        metric = faiss.METRIC_INNER_PRODUCT
        logger.info("faiss metric: inner product")
    elif distance_func == "l2":
        metric = faiss.METRIC_L2
        logger.info("faiss metric: l2 distance")
    else:
        raise ValueError("faiss metric needs to be provided. You can choose from [cosine, ip, l2]")

    index_type = "IDMap,Flat"
    index = faiss.index_factory(d, index_type, metric)

    index.add_with_ids(item_emb, item_labels)

    endtime = time.time()
    logger.info(f"faiss build {index.ntotal} indices using {endtime - starttime:.2f} seconds")
    starttime = endtime

    # start search

    adjusted_retrieve_cnt = k + max([len(_) for _ in users_liked])
    predictions_complementary_topk = []
    _, inds = index.search(query_user_emb, k=adjusted_retrieve_cnt)
    del index
    gc.collect()
    predictions_topk = inds[:, :k]

    endtime = time.time()
    logger.info(f"faiss KNN search finished in {endtime - starttime:.2f} seconds")
    starttime = endtime

    for i, ind in enumerate(inds):
        filtered_ind = [_ for _ in ind if _ not in users_liked[i]][:k]
        predictions_complementary_topk.append(filtered_ind)

    endtime = time.time()
    logger.info(f"Complementary faiss KNN search finished in {endtime - starttime:.2f} seconds")

    return predictions_topk, predictions_complementary_topk


def ann_query_complementary(
    item_emb, item_labels, user_emb, users, users_liked, k=50, index=None, distance_func="cosine"
):
    start_time = datetime.now()

    if index is None:
        index = create_ann_index(item_emb, item_labels, distance_func=distance_func)
    seconds_used = (datetime.now() - start_time).total_seconds()
    mid_time = datetime.now()
    logger.info("ANN Index build took %.2f seconds", seconds_used)

    query_user_emb = user_emb[users]

    if distance_func == "l2":
        query_user_emb = append_extra_dim_for_l2(query_user_emb)
    # topK recall
    results = index.knn_query(query_user_emb, k=k, num_threads=multiprocessing.cpu_count())

    predictions_topk = []
    # distances_topk = []
    for i in range(len(results[0])):
        predictions_topk.append(results[0][i])
        # distances_topk.append(results[1][i])

    seconds_used = (datetime.now() - mid_time).total_seconds()
    mid_time = datetime.now()
    logger.info("TopK ANN search finished in %.2f seconds", seconds_used)

    # Query ANN results, excluding liked items
    predictions_complementary_topk = []
    for i in range(len(users)):
        count = len(users_liked[i]) + k
        single_result = index.knn_query(query_user_emb[i], k=count, num_threads=multiprocessing.cpu_count())
        # distance is in ASC order, take top from the beginning.
        ids = list(
            itertools.islice(
                (j for j in range(len(single_result[0][0])) if single_result[0][0][j] not in users_liked[i]), k
            )
        )

        predictions_complementary_topk.append(single_result[0][0][ids])

    seconds_used = (datetime.now() - mid_time).total_seconds()
    mid_time = datetime.now()
    logger.info("Complementary TopK ANN search finished in %.2f seconds", seconds_used)

    return predictions_topk, predictions_complementary_topk


def ann_query(item_emb, item_labels, user_emb, users, users_liked, k=50, index=None, distance_func="cosine"):
    start_time = datetime.now()

    log_memory_usage("ann_query->in")
    if index is None:
        index = create_ann_index(item_emb, item_labels, distance_func=distance_func)
    mid_time = datetime.now()
    seconds_used = (mid_time - start_time).total_seconds()
    logger.info("ANN Index build took %.2f seconds", seconds_used)

    log_memory_usage("ann_query->create_ann_index")
    # Query ANN results, excluding liked items
    max_likes = max([len(user_liked) for user_liked in users_liked])
    count = k + max_likes
    if count > len(item_emb):
        count = len(item_emb)

    query_user_emb = user_emb[users]
    log_memory_usage("ann_query->query_user_emb")
    if distance_func == "l2":
        query_user_emb = append_extra_dim_for_l2(query_user_emb)
    results = index.knn_query(query_user_emb, k=count, num_threads=multiprocessing.cpu_count())

    log_memory_usage("index.knn_query")

    predictions = []
    distances = []
    for i in range(len(results[0])):
        ids = list(
            itertools.islice((j for j in range(len(results[0][i])) if results[0][i][j] not in users_liked[i]), k)
        )
        predictions.append(results[0][i][ids])
        distances.append(results[1][i][ids])
    end_time = datetime.now()
    seconds_used = (end_time - start_time).total_seconds()
    logger.info("ANN search finished in %.2f seconds", seconds_used)

    return predictions, distances


def knn_query(item_emb, item_labels, user_emb, users, k=50, distance_func="cosine"):
    start_time = datetime.now()

    factors = item_emb.shape[1]
    target_items = np.array(range(item_emb.shape[0]))
    if item_labels is not None:
        target_items = np.array(item_labels)
    user_vectors = nn.Embedding(num_embeddings=len(users), embedding_dim=factors)
    user_vectors.weight.data.copy_(torch.from_numpy(user_emb[users]))
    item_vectors = nn.Embedding(num_embeddings=item_emb.shape[0], embedding_dim=factors)
    item_vectors.weight.data.copy_(torch.from_numpy(item_emb))
    item_tensor = item_vectors(torch.LongTensor(range(item_emb.shape[0])))
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    predictions = []
    for i in range(len(users)):
        user_tensor = user_vectors(torch.LongTensor([i]))
        # only cosine or InnerProduct distance function are allowed.
        if distance_func == "cosine":
            sim_scores = cos(user_tensor, item_tensor).detach().numpy()
        else:
            sim_scores = torch.inner(user_tensor, item_tensor).detach().numpy()
            # torch.inner result is bi-dimentional
            sim_scores = sim_scores[0]
        ind = np.argpartition(sim_scores, -k)[-k:]
        predictions.append(target_items[ind])
    end_time = datetime.now()
    seconds_used = (end_time - start_time).total_seconds()
    logger.info("KNN search finished in %.2f seconds", seconds_used)
    return predictions


def knn_query_complementary(item_emb, item_labels, user_emb, users, users_liked, k=50, distance_func="cosine"):
    start_time = datetime.now()

    factors = item_emb.shape[1]
    target_items = np.array(range(item_emb.shape[0]))
    if item_labels is not None:
        target_items = np.array(item_labels)
    user_vectors = nn.Embedding(num_embeddings=len(users), embedding_dim=factors)
    user_vectors.weight.data.copy_(torch.from_numpy(user_emb[users]))
    item_vectors = nn.Embedding(num_embeddings=item_emb.shape[0], embedding_dim=factors)
    item_vectors.weight.data.copy_(torch.from_numpy(item_emb))
    item_tensor = item_vectors(torch.LongTensor(range(item_emb.shape[0])))
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    predictions_origin = []
    predictions_complementary = []
    for i in range(len(users)):
        user_tensor = user_vectors(torch.LongTensor([i]))
        # only cosine or InnerProduct distance function are allowed.
        if distance_func == "cosine":
            sim_scores = cos(user_tensor, item_tensor).detach().numpy()
        else:
            sim_scores = torch.inner(user_tensor, item_tensor).detach().numpy()
            # torch.inner result is bi-dimentional
            sim_scores = sim_scores[0]
        ind = np.argpartition(sim_scores, -k)[-k:]
        predictions_origin.append(target_items[ind])
        # complementary recall
        count = len(users_liked[i]) + k
        ind_complementary = np.argpartition(sim_scores, -count)[-count:]
        recalled_items = target_items[ind_complementary]
        # exclude items in trainset and select top k.
        ids = list(
            itertools.islice((j for j in range(len(recalled_items)) if recalled_items[j] not in users_liked[i]), count)
        )
        # the sim_scores (Cosine\InnerProduct) are in ASC order, take neareast from rear
        predictions_complementary.append(recalled_items[ids[-k:]])

    end_time = datetime.now()
    seconds_used = (end_time - start_time).total_seconds()
    logger.info("KNN search finished in %.2f seconds", seconds_used)
    return predictions_origin, predictions_complementary


def count_zero_embeddings(embeddings):
    """Zero embeddings are discarded at ANN server side. ANN server incorrectly
    classifies vectors with all dimensional value < 1E-6 as zero vector, which
    we follow here.
    """
    factors = embeddings.shape[1]
    target = np.ones((factors,)) * 1e-6
    x = np.less(embeddings, target)
    y = np.all(x, axis=1)
    cnt = np.sum(y)
    pct = 1.0 * cnt / len(embeddings)
    return cnt, pct


def evaluation_at_k(
    item_emb,
    item_labels,
    user_emb,
    users,
    click_in_train,
    users_liked,
    ground_truths,
    complementary_ground_truths,
    k=50,
    knn=False,
    index=None,
    distance_func="cosine",
):
    # Get ANN query results

    faiss_preds_topk, faiss_preds_complementary_topk = faiss_knn_query_complementary(
        item_emb, item_labels, user_emb, users, users_liked, k=k, index=index, distance_func=distance_func
    )

    preds = faiss_preds_complementary_topk
    pred_labels = np.zeros((len(preds), k), dtype=np.float32)
    found = {}
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            # pred_labels[i, j] = 1. if preds[i][j] in ground_truths[i] else 0.
            if preds[i][j] in ground_truths[i]:
                pred_labels[i, j] = 1.0
                tid = preds[i][j]
                uid = users[i]
                key = str(uid) + "," + str(tid)
                found[key] = 1
            else:
                pred_labels[i, j] = 0.0

    # Compute metrics, using complementary recall
    weight = [len(_) for _ in click_in_train]  # use # of clicked items as recall weight

    ann_recall_origin, avg_ann_recall_origin, weighted_ann_recall_origin = recall_at_k_new(
        faiss_preds_topk, ground_truths, k, weight
    )
    ann_recall_complementary, avg_ann_recall_complementary, weighted_ann_recall_complementary = recall_at_k_new(
        faiss_preds_complementary_topk, complementary_ground_truths, k, weight
    )
    # ann_recall_complementary = None

    mrr = mrr_at_k(pred_labels, k)
    ndcg = ndcg_at_k(pred_labels, ground_truths, k)

    # Count zero embeddings
    zero_cnt, zero_pct = count_zero_embeddings(item_emb)

    result = {
        f"knn_overall_recall_{k}": ann_recall_origin,
        f"knn_overall_recall_complementary{k}": ann_recall_complementary,
        f"knn_click_weighted_recall_{k}": weighted_ann_recall_origin,
        f"knn_click_weighted_recall_complementary{k}": weighted_ann_recall_complementary,
        f"knn_avg_user_recall_{k}": avg_ann_recall_origin,
        f"knn_avg_user_recall_complementary{k}": avg_ann_recall_complementary,
        f"mrr_{k}": mrr,
        f"ndcg_{k}": ndcg,
        "item_zero_cnt": int(zero_cnt),
        "item_zero_pct": zero_pct,
    }
    return result, found
