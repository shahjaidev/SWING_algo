#!/usr/bin/python3
import functools
import logging
import time
from collections.abc import Iterable
from multiprocessing import Pool
from struct import pack, unpack

import numpy as np


logger = logging.getLogger(__name__)

BYTE = 1
ENCODING = "utf-8"


class Counter(object):
    def __init__(self):
        self.val = 0

    def incr(self):
        self.val += 1


def time_io(func):
    """Wraps a class method to insert timing and throughput info."""

    @functools.wraps(func)
    def time_io_wrapper(*args, **kwargs):
        # Start timer
        start = time.time()

        # Record the total number of processed records
        counter = Counter()
        kwargs["counter"] = counter

        # Call function with counter
        result = func(*args, **kwargs)

        # Stop timer and calculate throughput
        elapsed = time.time() - start
        tput = counter.val / elapsed

        logger.info(
            "%s(%s) time: %.2f min. total records %s, throughput %s records/s",
            func.__name__,
            args[1],
            elapsed / 60,
            counter.val,
            tput,
        )

        return result

    return time_io_wrapper


class RecordFormatter(object):

    type_letter = {
        "char": "s",
        "float": "f",
        "double": "d",
    }

    type_size = {
        "char": 1 * BYTE,
        "float": 4 * BYTE,
        "double": 8 * BYTE,
    }

    def __init__(self, desc, text_delimiter="\t", little_endian=False):
        """Transforms sequence to text/bin record, and vice versa.

        Args:
            desc (:obj:`list` of :obj:`str`): Description of a record.
                e.g.
                [
                    ["char", 32],
                    ["float", 1],
                    ["double", 1],
                ]
            text_delimiter (str): Delimiter for text record.
            little_endian (bool): Use little endian or not.
        """
        self.desc = desc
        self.text_delimiter = text_delimiter
        self.little_endian = little_endian
        self.fmt = self._calc_fmt()
        self.size = self._calc_size()

        self._type_check()

    def _calc_fmt(self):
        """Get record format."""
        endian = "<" if self.little_endian else ">"
        data_fmt = "".join(["%d%s" % (d[1], self.type_letter[d[0]]) for d in self.desc])
        return endian + data_fmt

    def _calc_size(self):
        """Get size of binary encoding."""
        s = 0
        for d in self.desc:
            s += self.type_size[d[0]] * d[1]
        return s

    def _type_check(self):
        for d in self.desc:
            assert d[0] in ("char", "float", "double")

    def data_to_text(self, data):
        """Convert a list of objects to a text line."""
        parts = []
        for i in range(len(self.desc)):
            if self.desc[i][0] == "char":
                parts.append(data[i])
            elif isinstance(data[i], Iterable):
                parts.append(",".join([repr(e) for e in data[i]]))
            else:
                parts.append(repr(data[i]))
        return self.text_delimiter.join(parts) + "\n"

    def text_to_data(self, text):
        """Convert a text line to a list of objects."""
        parts = text.strip().split(self.text_delimiter)
        for i in range(len(self.desc)):
            if self.desc[i][0] == "char":
                parts[i] = parts[i].strip()
            elif self.desc[i][1] > 1:
                dtype = np.float32 if self.desc[i][0] == "float" else np.float64
                parts[i] = np.array(parts[i].split(","), dtype=dtype)
            else:
                parts[i] = float(parts[i])
        return parts

    def data_to_binary(self, data):
        """Convert data to binary record"""
        pack_in = []
        for i in range(len(self.desc)):
            if self.desc[i][0] == "char":
                d = data[i].ljust(self.desc[i][1])
                pack_in.append(d.encode(ENCODING))
            elif isinstance(data[i], Iterable):
                pack_in += list(data[i])
            else:
                pack_in.append(data[i])
        return pack(self.fmt, *pack_in)

    def binary_to_data(self, binary):
        """Convert binary record to data"""
        parts = []
        data = unpack(self.fmt, binary)
        begin = 0
        for i in range(len(self.desc)):
            if self.desc[i][0] == "char":
                parts.append(data[begin].decode(ENCODING).strip())
                begin += 1
            elif self.desc[i][1] > 1:
                dtype = np.float32 if self.desc[i][0] == "float" else np.float64
                end = begin + self.desc[i][1]
                parts.append(np.array(data[begin:end], dtype=dtype))
                begin = end
            else:
                parts.append(data[begin])
                begin += 1
        return parts


class RecordIO(object):
    def __init__(self, formatter):
        self.formatter = formatter

    @time_io
    def write_text(self, filename, *args, **context):
        with open(filename, "w") as f:
            for i, data in enumerate(zip(*args)):
                try:
                    f.write(self.formatter.data_to_text(data))
                except Exception:
                    logger.exception("failed to write text data %s: %s", i, data)
                context["counter"].incr()
                if (i + 1) % 10000 == 0:
                    logger.info(f"{i+1} records processed")

    @time_io
    def write_text_in_parallel(self, filename, workers, *args, **context):
        with Pool(processes=workers) as pool:
            pool_iter = pool.imap(self.formatter.data_to_text, zip(*args), chunksize=256)

            with open(filename, "w") as f:
                for i, data in enumerate(pool_iter):
                    try:
                        f.write(data)
                    except Exception:
                        logger.exception("fail to write text data %s: %s", i, data)
                    context["counter"].incr()

    @time_io
    def read_text(self, filename, **context):
        result = [[] for _ in range(len(self.formatter.desc))]

        with open(filename, "r") as f:
            for i, line in enumerate(f):
                try:
                    data = self.formatter.text_to_data(line)
                    for j, d in enumerate(data):
                        result[j].append(d)
                except Exception:
                    logger.exception("fail to read text data %s", i)
                context["counter"].incr()

        return result

    @time_io
    def write_binary(self, filename, *args, **context):
        with open(filename, "wb") as f:
            for i, data in enumerate(zip(*args)):
                try:
                    f.write(self.formatter.data_to_binary(data))
                except Exception:
                    logger.exception("failed to write binary data %s: %s", i, data)
                context["counter"].incr()

    @time_io
    def write_binary_in_parallel(self, filename, workers, *args, **context):
        with Pool(processes=workers) as pool:
            pool_iter = pool.imap(self.formatter.data_to_binary, zip(*args), chunksize=256)

            with open(filename, "wb") as f:
                for i, data in enumerate(pool_iter):
                    try:
                        f.write(data)
                    except Exception:
                        logger.exception("fail to write text data %s: %s", i, data)
                    context["counter"].incr()

    @time_io
    def read_binary(self, filename, **context):
        result = [[] for _ in range(len(self.formatter.desc))]
        size = self.formatter.size

        cnt = 0
        with open(filename, "rb") as f:
            binary = f.read(size)
            while binary:
                try:
                    data = self.formatter.binary_to_data(binary)
                    for j, d in enumerate(data):
                        result[j].append(d)
                except Exception:
                    logger.exception("fail to read binary data %s", cnt)
                cnt += 1
                binary = f.read(size)

        context["counter"].val = cnt

        return result


if __name__ == "__main__":
    console_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s - %(levelname)s] %(message)s", handlers=[console_handler]
    )

    filename = "output.txt"
    arr1 = ["user1", "user2", "user3", "user4", "user5"]
    arr2 = np.array([[2.0, 2.1], [2.1, 2.2], [2.2, 2.3], [2.3, 2.4], [2.4, 2.5]], dtype=np.float32)
    arr3 = [3.0, 3.0, 3.0, 4.5, 5.6]
    r_fmt = RecordFormatter([["char", 8], ["float", 2], ["double", 1]])
    record_io = RecordIO(r_fmt)
    logger.info("writing to text ...")
    record_io.write_text(filename, arr1, arr2, arr3)
    logger.info("reading from text ...")
    result = record_io.read_text(filename)
    logger.info("result: %s", result)

    workers = 3
    logger.info("writing to text in parallel ...")
    record_io.write_text_in_parallel(filename, workers, arr1, arr2, arr3)
    logger.info("reading from text ...")
    result = record_io.read_text(filename)
    logger.info("result: %s", result)

    filename = "output.bin"
    logger.info("writing to binary ...")
    record_io.write_binary(filename, arr1, arr2, arr3)
    logger.info("reading from binary ...")
    result = record_io.read_binary(filename)
    logger.info("result: %s", result)

    logger.info("writing to binary in parallel ...")
    record_io.write_binary_in_parallel(filename, workers, arr1, arr2, arr3)
    logger.info("reading from binary ...")
    result = record_io.read_binary(filename)
    logger.info("result: %s", result)
