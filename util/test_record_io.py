#!/usr/bin/python3
import os
import random
import shutil
import sys
import unittest
import uuid

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from util.record_io import RecordFormatter, RecordIO  # noqa: E402


class TestRecordIO(unittest.TestCase):
    def setUp(self):
        self.dirname = f"/tmp/test_record_io-{str(uuid.uuid4())}"
        self.filename = os.path.join(self.dirname, str(uuid.uuid4()))
        os.makedirs(self.dirname)

    def tearDown(self):
        try:
            shutil.rmtree(self.dirname)
        except Exception:
            pass

    def test_sequential_1(
        self,
    ):
        records = 100
        factors = 96
        desc = [["char", 32], ["float", factors], ["double", 1]]
        arr1 = ["dummy_%d" % i for i in range(records)]
        arr2 = np.asarray(np.random.random([records, factors]), dtype=np.float32)
        arr3 = [random.uniform(0.1, 10.0) for i in range(records)]

        formatter = RecordFormatter(desc)
        record_io = RecordIO(formatter)

        # write text and read text
        record_io.write_text(self.filename, arr1, arr2, arr3)
        result = record_io.read_text(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)

        # write binary and read binary
        record_io.write_binary(self.filename, arr1, arr2, arr3)
        result = record_io.read_binary(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)

    def test_sequential_2(self):
        records = 100
        factors = 96
        f_2 = 10
        desc = [["char", 32], ["float", factors], ["double", 1], ["double", 1], ["double", f_2], ["char", 20]]
        arr1 = ["dummy_%d" % i for i in range(records)]
        arr2 = np.asarray(np.random.random([records, factors]), dtype=np.float32)
        arr3 = [random.uniform(0.1, 10.0) for i in range(records)]
        arr4 = [random.uniform(0.1, 100.0) for i in range(records)]
        arr5 = np.random.random([records, f_2])
        arr6 = ["A%d" % i for i in range(records)]

        formatter = RecordFormatter(desc)
        record_io = RecordIO(formatter)

        # write text and read text
        record_io.write_text(self.filename, arr1, arr2, arr3, arr4, arr5, arr6)
        result = record_io.read_text(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)
        self.assertEqual(result[3], arr4)
        self.assertTrue((np.asarray(result[4]) == arr5).all())
        self.assertEqual(result[5], arr6)

        # write binary and read binary
        record_io.write_binary(self.filename, arr1, arr2, arr3, arr4, arr5, arr6)
        result = record_io.read_binary(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)
        self.assertEqual(result[3], arr4)
        self.assertTrue((np.asarray(result[4]) == arr5).all())
        self.assertEqual(result[5], arr6)

    def test_parallel_1(self):
        records = 100
        factors = 96
        desc = [["char", 32], ["float", factors], ["double", 1]]
        arr1 = ["dummy_%d" % i for i in range(records)]
        arr2 = np.asarray(np.random.random([records, factors]), dtype=np.float32)
        arr3 = [random.uniform(0.1, 10.0) for i in range(records)]

        formatter = RecordFormatter(desc)
        record_io = RecordIO(formatter)

        # write text and read text
        record_io.write_text_in_parallel(self.filename, 8, arr1, arr2, arr3)
        result = record_io.read_text(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)

        # write binary and read binary
        record_io.write_binary_in_parallel(self.filename, 8, arr1, arr2, arr3)
        result = record_io.read_binary(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)

    def test_parallel_2(self):
        records = 100
        factors = 96
        f_2 = 10
        desc = [["char", 32], ["float", factors], ["double", 1], ["double", 1], ["double", f_2], ["char", 20]]
        arr1 = ["dummy_%d" % i for i in range(records)]
        arr2 = np.asarray(np.random.random([records, factors]), dtype=np.float32)
        arr3 = [random.uniform(0.1, 10.0) for i in range(records)]
        arr4 = [random.uniform(0.1, 100.0) for i in range(records)]
        arr5 = np.random.random([records, f_2])
        arr6 = ["A%d" % i for i in range(records)]

        formatter = RecordFormatter(desc)
        record_io = RecordIO(formatter)

        # write text and read text
        record_io.write_text_in_parallel(self.filename, 8, arr1, arr2, arr3, arr4, arr5, arr6)
        result = record_io.read_text(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)
        self.assertEqual(result[3], arr4)
        self.assertTrue((np.asarray(result[4]) == arr5).all())
        self.assertEqual(result[5], arr6)

        # write binary and read binary
        record_io.write_binary_in_parallel(self.filename, 8, arr1, arr2, arr3, arr4, arr5, arr6)
        result = record_io.read_binary(self.filename)

        self.assertEqual(result[0], arr1)
        self.assertTrue((np.asarray(result[1]) == arr2).all())
        self.assertEqual(result[2], arr3)
        self.assertEqual(result[3], arr4)
        self.assertTrue((np.asarray(result[4]) == arr5).all())
        self.assertEqual(result[5], arr6)


if __name__ == "__main__":
    unittest.main()
