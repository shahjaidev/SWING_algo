import itertools
import logging
import multiprocessing
import timeit
from collections import Counter

import hnswlib
import numpy as np
import torch
from util.helper import append_extra_dim_for_l2, get_user_results_from_i2imatrix


logger = logging.getLogger(__name__)


def item_popularities_in_trainset(train_user_items, do_smooth):
    _, tids = train_user_items.nonzero()
    # count
    tid_popularity_map = dict(Counter(tids))
    # normalize
    num_users, _ = train_user_items.shape
    if do_smooth:
        # avoid 0 probability
        tid_popularity_map.update((tid, (count + 1) / (num_users + 1)) for tid, count in tid_popularity_map.items())
    else:
        tid_popularity_map.update((tid, count / num_users) for tid, count in tid_popularity_map.items())
    return tid_popularity_map


def mean_map_metric(predictions, tid_metric_map, k):
    metric = 0.0
    for prediction in predictions:
        if len(prediction) != 0:
            metric += np.sum(np.vectorize(tid_metric_map.get)(prediction[:k]))

    num_predictions = len(list(itertools.chain(*predictions)))
    if num_predictions != 0:
        metric /= num_predictions
    return metric


def novelty_at_k(predictions, k, train_user_items):
    tid_popularity_map = item_popularities_in_trainset(train_user_items, do_smooth=True)
    tid_popularity_map.update((tid, -np.log2(popularity)) for tid, popularity in tid_popularity_map.items())
    novelty = mean_map_metric(predictions, tid_popularity_map, k)
    return novelty


def popularity_at_k(predictions, k, train_user_items):
    tid_popularity_map = item_popularities_in_trainset(train_user_items, do_smooth=False)
    popularity = mean_map_metric(predictions, tid_popularity_map, k)
    return popularity


def user_coverage_at_k(predictions):
    return sum(prediction.shape[0] != 0 for prediction in predictions) / len(predictions)


def soft_recall_at_k(
    predictions_vec, doc_embeddings, ground_truths, report_soft_recall, user_type="prism", threshold=None
):
    if predictions_vec is None or report_soft_recall is False or user_type == "specific":
        return None, None

    user_soft_recalls = []
    soft_corrects = 0
    n_gt = sum([len(gt) for gt in ground_truths])
    for i in range(len(ground_truths)):
        click_vec = doc_embeddings.get_eval_embedding(
            indices=[ground_truths[i]]
        )  # click_vec.shape: n_click_ground_truth, embedding_size
        prediction_vec = predictions_vec[i]  # prediction_vec.shape: k, embedding_size
        click_vec = torch.tensor(np.array(click_vec)).squeeze(0)
        prediction_vec = torch.tensor(np.array(prediction_vec))
        if prediction_vec.shape[0] != 0:
            cosine_similarity = torch.matmul(
                click_vec, prediction_vec.transpose(0, 1)
            )  # cosine_similarity.shape: n_click_ground_truth * k
            if threshold is not None:
                soft_correct = sum(torch.max(cosine_similarity, dim=1).values > threshold).item()
            else:
                soft_correct = sum(torch.max(cosine_similarity, dim=1).values).item()

            user_soft_recalls.append(soft_correct / ground_truths[i].shape[0])
            soft_corrects += soft_correct
        else:
            user_soft_recalls.append(np.nan)

    user_soft_recalls = np.array(user_soft_recalls)
    soft_recall = soft_corrects / n_gt

    return round(soft_recall, 4), user_soft_recalls


def relevancy_at_k(predictions_vec, doc_embeddings, metric, train_user_items, users, report_relevancy):
    """
    :param: predictions_vec: a list with shape (u, k, embedding_length) where u is the number of users, k is the number of recommendation per user
    :param doc_embeddings: a numpy array with shape (max_tid_used_in_recommendation, embedding_length)
    :param metric: Either "history_coverage" or "recommendation_relevance".
    :return: two objects are returned:
            a scalar value equal the mean of the requested metric for all the users in the input predictions list.
            a numpy array with shape (n,) which contains the requested metric for each user, where n is the length of the input predictions list.
    """
    assert metric in ["history_coverage", "recommendation_relevance"]

    if predictions_vec is None or report_relevancy is False:
        return None, None

    cosine_similarities = []
    for i, user in enumerate(users):
        click_vec = doc_embeddings.get_eval_embedding(
            indices=[train_user_items.getrow(user).nonzero()[1]]
        )  # click_vec.shape: n_click, embedding_size
        prediction_vec = predictions_vec[i]  # prediction_vec.shape: k, embedding_size
        click_vec = torch.tensor(np.array(click_vec)).squeeze(0)
        prediction_vec = torch.tensor(np.array(prediction_vec))
        if prediction_vec.shape[0] != 0:
            cosine_similarity = torch.matmul(
                click_vec, prediction_vec.transpose(0, 1)
            )  # cosine_similarity.shape: n_click * k
            if metric == "history_coverage":
                cosine_similarities.append(torch.max(cosine_similarity, dim=1).values.mean().item())
            elif metric == "recommendation_relevance":
                cosine_similarities.append(torch.max(cosine_similarity, dim=0).values.mean().item())
        else:
            cosine_similarities.append(np.nan)

    cosine_similarities = np.array(cosine_similarities)

    return round(np.mean(cosine_similarities[~np.isnan(cosine_similarities)]), 4), cosine_similarities


def diversity_at_k(predictions_vec, stat, report_diversity):
    """
    :param: predictions_vec: a list with shape (u, k, embedding_length) where u is the number of users, k is the number of recommendation per user
    :param stat: Either "average" or "median".
    :return: two objects are returned:
            a scalar value equal to the average/median diversity of recommended list for all the users from the input predictions list.
            a numpy array with shape (n,) which contains the average/median diversity of recommended list for each user, where n is the length of the input predictions list.
    """
    assert stat in ["average", "median"]

    if predictions_vec is None or report_diversity is False:
        return None, None

    if len(set(map(len, predictions_vec))) == 1 and predictions_vec[0].shape[0] != 0:
        predictions_vec = torch.tensor(np.array(predictions_vec))
        pred_sim_mat = 1 - torch.matmul(predictions_vec, predictions_vec.transpose(1, 2))
        if stat == "average":
            k = predictions_vec.shape[1]
            user_diversities = torch.sum(torch.triu(pred_sim_mat, diagonal=1), dim=(1, 2)) / (k * (k - 1) / 2)
            diversity_score = np.mean(np.array(user_diversities))
        elif stat == "median":
            user_diversities = torch.median(pred_sim_mat.flatten(1), 1).values
            diversity_score = np.median(np.array(user_diversities))
        return round(diversity_score, 4), user_diversities.numpy()
    else:
        user_diversities = []
        for i in range(len(predictions_vec)):
            prediction_vec = torch.tensor(np.array(predictions_vec[i]))
            prediction_vec = torch.unsqueeze(prediction_vec, 0)
            if prediction_vec.shape[1] == 0 or prediction_vec.shape[1] == 1:
                user_diversities.append(np.nan)
                continue
            pred_sim_mat = 1 - torch.matmul(prediction_vec, prediction_vec.transpose(1, 2))
            if stat == "average":
                k = prediction_vec.shape[1]
                user_diversities.append(
                    np.array(torch.sum(torch.triu(pred_sim_mat, diagonal=1), dim=(1, 2)) / (k * (k - 1) / 2))[0]
                )
            elif stat == "median":
                user_diversities.append(torch.median(pred_sim_mat.flatten(1), 1).values.item())
        user_diversities = np.array(user_diversities)
        if stat == "average":
            diversity_score = np.mean(user_diversities[~np.isnan(user_diversities)])
        else:
            diversity_score = np.median(user_diversities[~np.isnan(user_diversities)])
        return round(diversity_score, 4), user_diversities


def correct_at_k(predictions, ground_truths, k):
    correct = 0
    for i, prediction in enumerate(predictions):
        intersection = set(ground_truths[i]).intersection(set(prediction[:k]))
        correct += len(intersection)
    return correct


def recall_at_k(predictions, ground_truths, k):
    correct = correct_at_k(predictions, ground_truths, k)
    return correct / sum([len(gt) for gt in ground_truths])


def precision_at_k(predictions, ground_truths, k):
    correct = correct_at_k(predictions, ground_truths, k)
    num_predictions = len(list(itertools.chain(*predictions)))
    return correct / num_predictions if num_predictions != 0 else 0


def fmeasure_at_k(recall, precision):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0


def avg_num_predictions_at_k(predictions):
    num_predictions = len(list(itertools.chain(*predictions)))
    avg_num_predictions = num_predictions / len(predictions)
    return avg_num_predictions


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
    index.init_index(max_elements=int(len(item_emb) * 1.5), ef_construction=400, M=16, random_seed=1234)

    index.set_ef(200)  # ef should be greater than k
    index.set_num_threads(multiprocessing.cpu_count())
    start_time = timeit.default_timer()
    if item_labels is None:
        item_labels = range(len(item_emb))
    try:
        index.add_items(item_emb, item_labels)
    except Exception:
        raise KeyError("check your item files. make sure its targetid,id format")
    print("Time for function index.add_items: ", timeit.default_timer() - start_time)

    return index


def ann_query(
    item_emb,
    item_labels,
    user_emb,
    users,
    users_liked,
    k=50,
    index=None,
    distance_func="cosine",
    allow_click=False,
    reco_cutoff_score=None,
):

    # Query ANN results, excluding liked items
    max_likes = max([len(user_liked) for user_liked in users_liked])
    if allow_click is True:
        count = k
    else:
        count = k + max_likes
    if count > len(item_emb):
        count = len(item_emb)
    if index is None:
        index = create_ann_index(item_emb, item_labels, distance_func=distance_func)

    query_user_emb = user_emb[users]
    if distance_func == "l2":
        query_user_emb = append_extra_dim_for_l2(query_user_emb)
    start_time = timeit.default_timer()
    results = index.knn_query(query_user_emb, k=count, num_threads=multiprocessing.cpu_count())
    print("Time for function index.knn_query: ", timeit.default_timer() - start_time)

    predictions = []
    distances = []
    start_time = timeit.default_timer()
    for i in range(len(results[0])):
        if allow_click is True:
            users_liked[i] = []
        ids = list(
            itertools.islice((j for j in range(len(results[0][i])) if results[0][i][j] not in users_liked[i]), k)
        )
        if distance_func == "ip":
            # hnswlib return 1-ip but MS ANN team return -ip
            distance = results[1][i][ids] - 1
        else:
            distance = results[1][i][ids]
        if reco_cutoff_score is not None:
            ids = np.array(ids)[distance < reco_cutoff_score]
            ids = list(ids)
        predictions.append(results[0][i][ids])
        distances.append(distance)
    print("Time for for-loop in ann_query: ", timeit.default_timer() - start_time)

    return predictions, distances, index


def matrix_query(
    users,
    users_liked,
    i2imatrix_csr,
    train_user_items,
    max_item_index_to_save=None,
    k=50,
    allow_click=False,
    reco_cutoff_score=None,
):

    # Query ANN results, excluding liked items
    max_likes = max([len(user_liked) for user_liked in users_liked])
    if allow_click is True:
        count = k
    else:
        count = k + max_likes
    if count >= max_item_index_to_save + 1:
        count = max_item_index_to_save

    start_time = timeit.default_timer()
    results = get_user_results_from_i2imatrix(
        i2imatrix_csr,
        train_user_items,
        users,
        max_item_index_to_save,
        clicks_limit=None,
        recom_per_click_limit=50,
        k=k,
    )
    print("Time for function get_user_results_from_i2imatrix", timeit.default_timer() - start_time)

    predictions = []
    distances = []
    start_time = timeit.default_timer()
    for i in range(len(results[0])):
        if allow_click is True:
            users_liked[i] = []
        ids = list(
            itertools.islice((j for j in range(len(results[0][i])) if results[0][i][j] not in users_liked[i]), k)
        )
        distance = np.array(results[1][i])[ids]
        if reco_cutoff_score is not None:
            ids = np.array(ids)[distance < reco_cutoff_score]
            ids = list(ids)
        predictions.append(np.array(results[0][i], dtype=np.uint64)[ids])
        distances.append(distance)
    print("Time for for-loop in matrix_query: ", timeit.default_timer() - start_time)

    return predictions, distances


def count_zero_embeddings(embeddings):
    """Zero embeddings are discarded at ANN server side. ANN server incorrectly
    classifies vectors with all dimensional value < 1E-6 as zero vector, which
    we follow here.
    """
    if embeddings is None:
        return 0, 0
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
    users_liked,
    ground_truths,
    train_user_items,
    k=50,
    index=None,
    distance_func="cosine",
    user_type="prism",
    reco_cutoff_score=None,
    i2imatrix_csr=None,
    max_item_index_to_save=None,
):
    if i2imatrix_csr is not None:
        preds, distances = matrix_query(
            users,
            users_liked,
            i2imatrix_csr,
            train_user_items,
            max_item_index_to_save=max_item_index_to_save,
            k=k,
            reco_cutoff_score=reco_cutoff_score,
        )
        index = None
    else:
        # Get ANN query results
        preds, distances, index = ann_query(
            item_emb,
            item_labels,
            user_emb,
            users,
            users_liked,
            k=k,
            index=index,
            distance_func=distance_func,
            reco_cutoff_score=reco_cutoff_score,
        )

    start_time = timeit.default_timer()
    if len(ground_truths[0]) == 1 and len(preds[0]) == k and len(set(map(len, preds))) == 1:
        # Broadcasting possible only when there is 1 gt item per user and len(preds[0]) should be equal to k (it could be smaller "if count > len(item_emb)")
        pred_labels = np.float32(np.int64(preds) == np.int64(ground_truths))
        print(
            "Time for creating pred_labels using numpy comparison in evalution_at_k: ",
            timeit.default_timer() - start_time,
        )
    else:
        pred_labels = np.zeros((len(preds), k), dtype=np.float32)
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                pred_labels[i, j] = 1.0 if preds[i][j] in ground_truths[i] else 0.0
        print(
            "Time for creating pred_labels using nested for-loop in evalution_at_k: ",
            timeit.default_timer() - start_time,
        )

    if user_type == "prism":
        mrr = round(mrr_at_k(pred_labels, k), 4)
        recall = round(recall_at_k(preds, ground_truths, k), 4)
        precision = round(precision_at_k(preds, ground_truths, k), 4)
        fmeasure = round(fmeasure_at_k(recall, precision), 4)
        ndcg = round(ndcg_at_k(pred_labels, ground_truths, k), 4)
    else:
        mrr = recall = precision = fmeasure = ndcg = None

    # Count zero embeddings
    zero_cnt, zero_pct = count_zero_embeddings(item_emb)
    results = {
        f"recall_{k}": recall,
        f"precision_{k}": precision,
        f"fmeasure_{k}": fmeasure,
        f"mrr_{k}": mrr,
        f"ndcg_{k}": ndcg,
        "item_zero_cnt": zero_cnt,
        "item_zero_pct": zero_pct,
    }

    return results, index, preds, distances


def evaluation_at_k_additional(
    results,
    preds,
    distances,
    users,
    ground_truths,
    train_user_items,
    k=50,
    index=None,
    user_type="prism",
    doc_embeddings_dict=None,
    report_diversity=False,
    report_relevancy=False,
    report_soft_recall=False,
):

    novelty = round(novelty_at_k(preds, k, train_user_items), 4)
    popularity = round(popularity_at_k(preds, k, train_user_items), 4)
    avg_num_predictions = round(avg_num_predictions_at_k(preds), 4)
    user_coverage = round(user_coverage_at_k(preds), 4)

    # Compute metrics
    predictions_vec_dict = dict()
    for mode in doc_embeddings_dict:
        predictions_vec_dict[mode] = doc_embeddings_dict[mode].get_eval_embedding(indices=preds)

    diversity_dict = dict()
    diversity_user_dict = dict()
    for mode in doc_embeddings_dict:
        for stat in ["average", "median"]:
            diversity_dict[mode, stat], diversity_user_dict[mode, stat] = diversity_at_k(
                predictions_vec_dict[mode], stat, report_diversity
            )

    relevancy_dict = dict()
    relevancy_user_dict = dict()
    for mode in doc_embeddings_dict:
        for metric in ["history_coverage", "recommendation_relevance"]:
            relevancy_dict[mode, metric], relevancy_user_dict[mode, metric] = relevancy_at_k(
                predictions_vec_dict[mode],
                doc_embeddings_dict[mode],
                metric,
                train_user_items,
                users,
                report_relevancy,
            )

    soft_recall_dict = dict()
    soft_recall_user_dict = dict()
    for mode in doc_embeddings_dict:
        soft_recall_dict[mode], soft_recall_user_dict[mode] = soft_recall_at_k(
            predictions_vec_dict[mode], doc_embeddings_dict[mode], ground_truths, report_soft_recall, user_type
        )

    user_doc_preds = extract_user_doc_preds(
        users, preds, distances, k, diversity_user_dict, relevancy_user_dict, soft_recall_user_dict
    )

    results.update(
        {
            f"novelty_{k}": novelty,
            f"popularity_{k}": popularity,
            f"avg_num_predictions_{k}": avg_num_predictions,
            f"user_coverage_{k}": user_coverage,
        }
    )
    for mode, stat in diversity_dict:
        results.update({f"diversity_{mode}_{stat}_{k}": diversity_dict[mode, stat]})
    for mode, metric in relevancy_dict:
        results.update({f"{mode}_{metric}_{k}": relevancy_dict[mode, metric]})
    for mode in soft_recall_dict:
        results.update({f"soft_recall_{mode}_{k}": soft_recall_dict[mode]})

    return results, user_doc_preds, index


def extract_user_doc_preds(
    users, preds, distances, k, diversity_user_dict, relevancy_user_dict, soft_recall_user_dict
):
    user_doc_preds = {}
    for i, u in enumerate(users):
        recommendation_diversity = ""
        if diversity_user_dict is not None:
            for mode, stat in diversity_user_dict:
                if diversity_user_dict[mode, stat] is not None and not np.isnan(diversity_user_dict[mode, stat][i]):
                    recommendation_diversity += (
                        " " + f"diversity_{mode}_{stat}_{k}:" + str(round(diversity_user_dict[mode, stat][i], 2))
                    )
        recommendation_relevancy = ""
        if relevancy_user_dict is not None:
            for mode, metric in relevancy_user_dict:
                if relevancy_user_dict[mode, metric] is not None and not np.isnan(
                    relevancy_user_dict[mode, metric][i]
                ):
                    recommendation_relevancy += (
                        " " + f"{mode}_{metric}_{k}:" + str(round(relevancy_user_dict[mode, metric][i], 2))
                    )
        recommendation_soft_recall = ""
        if soft_recall_user_dict is not None:
            for mode in soft_recall_user_dict:
                if soft_recall_user_dict[mode] is not None and not np.isnan(soft_recall_user_dict[mode][i]):
                    recommendation_soft_recall += (
                        " " + f"soft_recall{mode}_{k}:" + str(round(soft_recall_user_dict[mode][i], 2))
                    )
        user_doc_preds[u] = [
            preds[i][0:k],
            distances[i][0:k],
            recommendation_diversity,
            recommendation_relevancy,
            recommendation_soft_recall,
        ]
    return user_doc_preds


def l2_dist(x, y):
    dist = np.sum([np.square(x[i] - y[i]) for i in range(len(x))])
    return dist
