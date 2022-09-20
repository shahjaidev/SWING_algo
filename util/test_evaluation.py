#!/usr/bin/python3
import os
import sys
import time
import unittest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.evaluation import ConvergenceTest  # noqa: E402


class DummyModel(object):
    def __init__(self):
        self.name = "dummy"


class TestConvergenceTest(unittest.TestCase):
    def setUp(self):
        self.epochs = list(range(10))
        self.metrics = [0.01, 0.1, 0.11, 0.12, 0.1, 0.11, 0.13, 0.11, 0.11, 0.11]
        self.model = DummyModel()

    def test_none_config(self):
        conv_test = ConvergenceTest()
        for epoch, metric in zip(self.epochs, self.metrics):
            terminate = conv_test(epoch, self.model, metric)
            self.assertFalse(terminate)
        self.assertEqual(conv_test.best_epoch, self.epochs[6])

    def test_exceed_max_time(self):
        conv_test = ConvergenceTest(max_time=0)
        time.sleep(0.001)
        self.assertTrue(conv_test(self.epochs[0], self.model, self.metrics[0]))
        self.assertEqual(conv_test.best_epoch, self.epochs[0])

    def test_exceed_epsilon_ts(self):
        conv_test = ConvergenceTest(epsilon=1000, epsilon_ts=-1)
        self.assertTrue(conv_test(self.epochs[0], self.model, self.metrics[0]))
        self.assertEqual(conv_test.best_epoch, self.epochs[0])

    def test_normal_run(self):
        conv_test = ConvergenceTest(epsilon=0, epsilon_ts=1, max_time=100)

        for epoch, metric in zip(self.epochs[:4], self.metrics[:4]):
            terminate = conv_test(epoch, self.model, metric)
            self.assertFalse(terminate)
        conv_test.best_epsilon_time = time.time() - 2
        self.assertTrue(conv_test(self.epochs[4], self.model, self.metrics[4]))
        self.assertEqual(conv_test.best_epoch, self.epochs[3])

    def test_zero_metric(self):
        conv_test = ConvergenceTest(epsilon=0.001, epsilon_ts=10, max_time=100)
        self.epochs = list(range(14))
        self.metrics = [0.0, 0.0, 0.01, 0.0, 0.0, 0.1, 0.11, 0.12, 0.1, 0.11, 0.13, 0.11, 0.11, 0.11]
        for epoch, metric in zip(self.epochs, self.metrics):
            terminate = conv_test(epoch, self.model, metric)
            self.assertFalse(terminate)
        self.assertEqual(conv_test.best_epoch, self.epochs[10])


if __name__ == "__main__":
    unittest.main()
