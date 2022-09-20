#!/usr/bin/python3
import copy
import gc
import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
from mlflow import log_metric
from scipy.sparse import coo_matrix
from util.helper import (
    DocEmbeddings,
    load_results,
    load_test_data,
    load_train_data,
    load_train_data_weighted_by_time,
    write_user_doc_preds,
)
from util.metrics import evaluation_at_k, evaluation_at_k_additional


logger = logging.getLogger(__name__)


def evaluate(
    args,
    test_path,
    input_path,
    user_type="prism",
    filename="user_doc_preds.tsv",
    index=None,
    max_item_index_to_save=None,
    logger_in=None,
    i2imatrix_csr=None,
):

    if logger_in:
        global logger
        logger = logger_in

    # Load test dataset.
    test_user_items, test_df = load_test_data(test_path, args.input_delimiter, logger_in=logger)

    # load training dataset
    if args.load_training_method == "load_train_data":
        train_user_items, train_df, _, _, _, _ = load_train_data(
            args, input_path, args.input_delimiter, logger_in=logger
        )
    elif args.load_training_method == "load_train_data_weighted_by_time":
        train_user_items, train_df, _, _, _, _ = load_train_data_weighted_by_time(
            args, input_path, args.input_delimiter, logger_in=logger
        )

    if i2imatrix_csr is not None:
        results, index, preds, distances, users, ground_truths = test_i2imatrix(
            i2imatrix_csr,
            train_user_items,
            test_user_items,
            k=args.user_top_k,
            user_type=user_type,
            reco_cutoff_score=args.reco_cutoff_score,
            max_item_index_to_save=max_item_index_to_save,
        )
    else:
        # Load embeddings.
        _, user_embeddings, _, item_embeddings = load_results(
            args.model_path, args.factors, args.load_format, load_weights="is_eals" in args and args.is_eals
        )
        results, index, preds, distances, users, ground_truths = test_embeddings(
            args.epoch,
            user_embeddings,
            item_embeddings,
            train_user_items,
            test_user_items,
            item_labels=None,
            k=args.user_top_k,
            distance_func=args.distance_func,
            user_type=user_type,
            reco_cutoff_score=args.reco_cutoff_score,
            index=index,
        )
        del user_embeddings, item_embeddings
        gc.collect()

    doc_embeddings_dict = dict()
    if args.report_relevancy is True or args.report_diversity is True or args.report_soft_recall is True:
        doc_embeddings_dict["visual"] = DocEmbeddings(
            args.doc_embeddings_visual_path,
            normalized_factor={"factor": 1 / 127.5, "scale": 1},
            delimiter="\t",
            embedding_delimiter=",",
            max_item_index_to_load=max_item_index_to_save
            if args.report_relevancy is False and args.report_soft_recall is False
            else None,
            item_labels=None,
            logger_in=logger_in,
        )
        doc_embeddings_dict["text"] = DocEmbeddings(
            args.doc_embeddings_text_path,
            normalized_factor={"factor": 1, "scale": 0},
            delimiter="\t",
            embedding_delimiter=" ",
            max_item_index_to_load=max_item_index_to_save
            if args.report_relevancy is False and args.report_soft_recall is False
            else None,
            item_labels=None,
            logger_in=logger_in,
        )

    results, user_doc_preds, index = evaluation_at_k_additional(
        results,
        preds,
        distances,
        users,
        ground_truths,
        train_user_items,
        k=args.user_top_k,
        index=index,
        user_type=user_type,
        doc_embeddings_dict=doc_embeddings_dict,
        report_diversity=args.report_diversity,
        report_relevancy=args.report_relevancy,
        report_soft_recall=args.report_soft_recall,
    )

    logger.info("test @ epoch %s results: %s", args.epoch, results)

    if args.mlflow:
        try:
            for metric, value in results.items():
                log_metric(f"{metric}", value, step=args.epoch)
        except Exception:
            logger.exception("failed to publish to mlflow")

    if args.save_user_doc_preds is True:
        write_user_doc_preds(user_doc_preds, train_df, test_df, output_dir=args.output_dir, filename=filename)

    results["item_zero_cnt"] = float(results["item_zero_cnt"])

    with open(os.path.join(args.output_dir, "metrics.txt"), "w" if user_type == "prism" else "a") as fp:
        fp.write("\n\n\n user_type: " + user_type + "\n")
        json.dump(results, fp, indent=4)

    return index


def test_i2imatrix(
    i2imatrix_csr,
    train_user_items,
    test_user_items,
    k=50,
    user_type="prism",
    reco_cutoff_score=None,
    max_item_index_to_save=None,
):

    logger.info("test starting")
    users = np.unique(coo_matrix(test_user_items).row)

    m = train_user_items[users].tocsr()
    users_liked = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    m = test_user_items[users].tocsr()
    ground_truths = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    result, index, preds, distances = evaluation_at_k(
        item_emb=None,
        item_labels=None,
        user_emb=None,
        users=users,
        users_liked=users_liked,
        ground_truths=ground_truths,
        train_user_items=train_user_items,
        k=k,
        index=None,
        distance_func=None,
        user_type=user_type,
        reco_cutoff_score=reco_cutoff_score,
        i2imatrix_csr=i2imatrix_csr,
        max_item_index_to_save=max_item_index_to_save,
    )

    return result, index, preds, distances, users, ground_truths


def test_embeddings(
    epoch,
    user_embeddings,
    item_embeddings,
    train_user_items,
    test_user_items,
    item_labels=None,
    k=50,
    distance_func="cosine",
    user_type="prism",
    reco_cutoff_score=None,
    index=None,
):

    logger.info("test @ epoch %s starting", epoch)
    users = np.unique(coo_matrix(test_user_items).row)

    m = train_user_items[users].tocsr()
    users_liked = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    m = test_user_items[users].tocsr()
    ground_truths = [m.indices[range(m.indptr[u], m.indptr[u + 1])] for u in range(len(users))]

    if item_labels is not None:
        item_embeddings = item_embeddings[item_labels]

    result, index, preds, distances = evaluation_at_k(
        item_embeddings,
        item_labels,
        user_embeddings,
        users,
        users_liked,
        ground_truths,
        train_user_items=train_user_items,
        k=k,
        index=index,
        distance_func=distance_func,
        user_type=user_type,
        reco_cutoff_score=reco_cutoff_score,
        i2imatrix_csr=None,
    )

    return result, index, preds, distances, users, ground_truths


class ConvergenceTest(object):
    def __init__(self, epsilon=None, epsilon_ts=None, max_time=None, return_model="best"):
        """Test when to stop training.

        Args:
            epsilon (float): Gain percent threshold in convergence test.
            epsilon_ts (int): Timespan threshold for gain percent in
                convergence test.
            max_time (int): Max allowed training time before termination.
        """
        self.epsilon = epsilon
        self.epsilon_ts = epsilon_ts
        self.max_time = max_time

        self.start_time = time.time()

        self.best_epoch = None
        self.best_model = None
        self.best_metric = None

        self.best_epsilon_epoch = None
        self.best_epsilon_metric = None
        self.best_epsilon_time = None

        self.return_model = return_model

    def update_best(self, epoch, model, metric, user_doc_preds):
        if self.best_epoch is None or metric > self.best_metric:
            logger.info("update best metric/model at epoch %s", epoch)
            self.best_epoch = epoch

            if self.return_model == "final":
                self.best_model = model
            else:
                self.best_model = copy.deepcopy(model)

            self.best_model.fit_callback = None
            self.best_model.user_doc_preds = user_doc_preds
            self.best_metric = metric

    def update_best_epsilon(self, epoch, metric, cur_time):
        if self.epsilon is not None and self.epsilon_ts is not None:
            if (
                self.best_epsilon_epoch is None
                or (self.best_epsilon_metric == 0 and metric >= self.best_epsilon_metric)
                or (metric - self.best_epsilon_metric) / self.best_epsilon_metric > self.epsilon
            ):
                logger.info("update best epsilon metric at epoch %s", epoch)
                self.best_epsilon_epoch = epoch
                self.best_epsilon_metric = metric
                self.best_epsilon_time = cur_time

    def exceed_epsilon_ts(self, epoch, cur_time):
        if self.epsilon_ts is None:
            return False

        elapsed = cur_time - self.best_epsilon_time
        if cur_time - self.best_epsilon_time > self.epsilon_ts:
            logger.info(
                "elapsed time from best epsilon %ss (epoch %s) exceeds " "epsilon timespan %ss at epoch %s",
                elapsed,
                self.best_epsilon_epoch,
                self.epsilon_ts,
                epoch,
            )
            return True
        else:
            return False

    def exceed_max_time(self, epoch, cur_time):
        if self.max_time is None:
            return False

        elapsed = cur_time - self.start_time
        if elapsed > self.max_time:
            logger.info("running time %ss exceeds max time %ss at epoch %s", elapsed, self.max_time, epoch)
            return True
        else:
            return False

    def __call__(self, epoch, model, metric, user_doc_preds):
        cur_time = time.time()

        self.update_best(epoch, model, metric, user_doc_preds)
        self.update_best_epsilon(epoch, metric, cur_time)

        return self.exceed_epsilon_ts(epoch, cur_time) or self.exceed_max_time(epoch, cur_time)


class TestRunner:
    def __init__(
        self,
        model,
        train_user_items,
        test_user_items,
        item_labels,
        k,
        distance_func="cosine",
        mlflow=False,
        epsilon=None,
        epsilon_ts=None,
        max_time=None,
        test_rand_embedding=False,
        conv_metric="recall_50",
        logging_steps=5,
        reco_cutoff_score=None,
        tensorboard=None,
        writer_in=None,
        logger_in=None,
        return_model="best",
        doc_embeddings_dict=None,
        report_diversity=False,
        report_relevancy=False,
        report_soft_recall=False,
    ):
        self.model = model
        self.train_user_items = train_user_items
        self.test_user_items = test_user_items
        self.item_labels = item_labels
        self.k = k
        self.distance_func = distance_func
        self.mlflow = mlflow
        self.reco_cutoff_score = reco_cutoff_score
        # np.random.seed(12)

        self.last_test_results = None

        self.conv_test = ConvergenceTest(epsilon, epsilon_ts, max_time, return_model)
        self.conv_metric = conv_metric
        self.test_rand_embedding = test_rand_embedding
        self.logging_steps = logging_steps
        self.tensorboard = tensorboard
        self.user_doc_preds = None
        self.doc_embeddings_dict = doc_embeddings_dict
        self.report_diversity = report_diversity
        self.report_relevancy = report_relevancy
        self.report_soft_recall = report_soft_recall

        if writer_in:
            global writer
            writer = writer_in
        if logger_in:
            global logger
            logger = logger_in

    def __call__(self, epoch, elapsed):
        return self.run(epoch)

    def get_rand_factors(self, users, items, size):
        user_factors = np.random.rand(users, size)
        item_factos = np.random.rand(items, size)
        return user_factors, item_factos

    """
    def store_loss(self,model, name):
        def inner(iteration, elapsed):
            loss = calculate_loss(train_item_users, model.item_factors, model.user_factors, 0)
            print("model %s iteration %i loss %.5f" % (name, iteration, loss))
            self.output[name].append(loss)

        return inner
    """

    def run(self, epoch):
        terminate = False
        self.output = defaultdict(list)
        # self.store_loss(self.model, "cg%i" % epoch)
        print("epochs:", epoch)
        if (epoch + 1) % self.logging_steps != 0:
            return
        if self.test_user_items is not None:
            result, user_doc_preds, _ = test_embeddings(
                epoch + 1,
                self.model.get_user_embeddings(),
                self.model.get_item_embeddings(),
                self.train_user_items,
                self.test_user_items,
                self.item_labels,
                self.k,
                self.distance_func,
                mlflow=self.mlflow,
                reco_cutoff_score=self.reco_cutoff_score,
                doc_embeddings_dict=self.doc_embeddings_dict,
            )
            if self.test_rand_embedding:
                user_count, embedding_size = self.model.user_factors.shape
                item_count, _ = self.model.item_factors.shape
                user_rand_factors, item_rand_factors = self.get_rand_factors(user_count, item_count, embedding_size)
                logger.info("test-random-embedding")
                test_embeddings(
                    epoch + 1,
                    user_rand_factors,
                    item_rand_factors,
                    self.train_user_items,
                    self.test_user_items,
                    self.item_labels,
                    self.k,
                    self.distance_func,
                    mlflow=self.mlflow,
                )
                logger.info("rand-embedding done!---")

            metric = result.get(self.conv_metric)
            self.last_test_results = result
            self.user_doc_preds = user_doc_preds
            if metric is not None:
                terminate = self.conv_test(epoch + 1, self.model, metric, user_doc_preds)
                if self.mlflow:
                    log_metric(f"best-{self.conv_metric}", self.conv_test.best_metric, step=epoch + 1)
        print(self.output)
        if self.tensorboard:
            for metric_name, metric_val in self.last_test_results.items():
                if metric_val is not None:
                    writer.add_scalar(f"test/{metric_name}", metric_val, epoch + 1)
        return terminate
