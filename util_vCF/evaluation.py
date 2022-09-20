#!/usr/bin/python3
import copy
import json
import logging
import os
import time

import numpy as np
from scipy.sparse import coo_matrix

from util import mlflow_utils
from util.helper import load_results, load_test_data, load_train_data, prepare_test_complementary
from util.metrics import evaluation_at_k


logger = logging.getLogger(__name__)


def evaluate(args):
    # Load embeddings.
    user_ids, user_embeddings, item_ids, item_embeddings = load_results(args.model_path, args.factors, args.load_text)
    # Load test dataset.
    test_user_items, _ = load_test_data(args.test_path, args.input_delimiter, user_ids, item_ids)
    # train_user_items = coo_matrix((test_user_items.shape[0], test_user_items.shape[1])).tocsr()
    # load training dataset
    train_user_items, train_df, _, idfs, item_sums, _ = load_train_data(
        args.input_path, args.input_delimiter, args.rating_scale, args.rating_offset
    )
    click_in_train, users_liked, ground_truths, complementary_ground_truths = prepare_test_complementary(
        train_user_items, test_user_items, None
    )

    result, found = test_embeddings(
        0,
        user_embeddings,
        item_embeddings,
        test_user_items,
        click_in_train,
        None,
        users_liked,
        ground_truths,
        complementary_ground_truths,
        k=args.user_top_k,
        knn=args.test_knn,
        distance_func=args.distance_func,
        mlflow=args.mlflow,
    )
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as fp:
        json.dump(result, fp, indent=4)
    with open(os.path.join(args.output_dir, "testset_found.tsv"), "w") as out_file:
        for k in found:
            out_file.write(k + "\n")


def test_embeddings(
    epoch,
    user_embeddings,
    item_embeddings,
    test_user_items,
    click_in_train,
    item_labels=None,
    users_liked=None,
    ground_truths=None,
    complementary_ground_truths=None,
    k=50,
    knn=False,
    distance_func="cosine",
    mlflow=False,
):
    logger.info("test @ epoch %s starting", epoch)

    users = np.unique(coo_matrix(test_user_items).row)

    if item_labels is not None:
        item_embeddings = item_embeddings[item_labels]

    logger.info("test index size: %d", item_embeddings.shape[0])

    result, found = evaluation_at_k(
        item_embeddings,
        item_labels,
        user_embeddings,
        users,
        click_in_train,
        users_liked,
        ground_truths,
        complementary_ground_truths,
        k=k,
        knn=knn,
        index=None,
        distance_func=distance_func,
    )

    logger.info("test @ epoch %s result: %s", epoch, result)
    mlflow_utils.mlflow_log_metrics(metrics_dict=result, step=epoch, prefix="test")

    return result, found


class ConvergenceTest(object):
    def __init__(self, epsilon=None, epsilon_ts=None, max_time=None, return_model="final"):
        """Test when to stop training.

        Args:
            epsilon (float): Gain percent threshold in convergence test.
            epsilon_ts (int): Timespan threshold for gain percent in
                convergence test.
            max_time (int): Max allowed training time before termination.
            return_model (str): The argument to decide whether to keep current best model on dev set.
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

    def update_best(self, epoch, model, metric):
        if self.best_epoch is None or metric > self.best_metric:
            logger.info("update best metric/model at epoch %s", epoch)
            self.best_epoch = epoch

            if self.return_model == "final":
                self.best_model = model
            else:
                self.best_model = copy.deepcopy(model)

            self.best_model.fit_callback = None
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

    def __call__(self, epoch, model, metric):
        cur_time = time.time()

        self.update_best(epoch, model, metric)
        self.update_best_epsilon(epoch, metric, cur_time)

        return self.exceed_epsilon_ts(epoch, cur_time) or self.exceed_max_time(epoch, cur_time)


class TestRunner:
    def __init__(
        self,
        model,
        click_in_train,
        users_liked,
        ground_truths,
        complementary_ground_truths,
        test_user_items,
        item_labels,
        k,
        useKNN,
        distance_func="cosine",
        mlflow=False,
        epsilon=None,
        epsilon_ts=None,
        max_time=None,
        conv_metric="recall_50",
        return_model="final",
    ):
        self.model = model
        self.test_user_items = test_user_items
        self.item_labels = item_labels
        self.k = k
        self.knn = useKNN
        self.distance_func = distance_func
        self.mlflow = mlflow
        self.click_in_train = click_in_train
        self.users_liked = users_liked
        self.ground_truths = ground_truths
        self.complementary_ground_truths = complementary_ground_truths

        self.epsilon = epsilon
        self.epsilon_ts = epsilon_ts
        self.max_time = max_time
        self.conv_metric = conv_metric
        self.return_model = return_model
        self.reset_training_state(new_model=model)
        self.terminate = False

    def __call__(self, epoch, elapsed):
        return self.run(epoch)

    def run(self, epoch):
        terminate = False
        metric = epoch  # default metric incase metric is None
        if self.test_user_items is not None:
            result, found = test_embeddings(
                epoch + 1,
                self.model.get_user_embeddings(),
                self.model.get_item_embeddings(),
                self.test_user_items,
                self.click_in_train,
                self.item_labels,
                self.users_liked,
                self.ground_truths,
                self.complementary_ground_truths,
                self.k,
                self.knn,
                self.distance_func,
                mlflow=self.mlflow,
            )
            metric = result.get(self.conv_metric)

            if metric is not None:
                terminate = self.conv_test(epoch + 1, self.model, metric)
                if self.mlflow:
                    mlflow_utils.mlflow_log_metric(
                        f"best-{self.conv_metric}", self.conv_test.best_metric, step=epoch + 1
                    )

        self.terminate = terminate
        return metric, result

    def reset_training_state(self, new_model=None, new_conv_metric=None):
        """Resets the test runner's training state, without resetting test set pre-processing.

        This method is useful for situations like auto-tuning when we want to reuse the test set
        between many training runs.

        Args:
            new_model (optional): Now use this model to fetch user/item embeddings. Defaults to None.
            new_conv_metric (str, optional): Now use this metric to determine convergence. Defaults to None.
        """
        if new_model is not None:
            self.model = new_model

        if new_conv_metric is not None:
            self.conv_metric = new_conv_metric

        self.conv_test = ConvergenceTest(self.epsilon, self.epsilon_ts, self.max_time, self.return_model)

    def get_model(self):
        return self.model

    def should_stop(self):
        return self.terminate
