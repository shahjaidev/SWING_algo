#!/usr/bin/python3
import logging
import time
from multiprocessing import Pool
from struct import pack, unpack

import numpy as np


logger = logging.getLogger(__name__)


BYTE = 1


def export_text(data):
    u_id, emb, val = data
    return u_id + "\t" + ",".join([repr(e) for e in emb]) + "\t" + str(val) + "\n"


def load_text(data, dtype=np.float32):
    u_id, emb, val = data.split("\t")
    emb = np.array(emb.split(","), dtype=dtype)
    val = float(val)
    return u_id, emb, val


def export_binary(data):
    u_id, emb, val, id_len = data
    return pack(
        ">%ds%df1d" % (id_len, len(emb)),
        u_id.ljust(id_len).encode("utf-8"),  # pad with whitespace and encode
        *emb,
        val
    )


def load_binary(data, dtype=np.float32):
    record, id_len, factors = data
    unpacked = unpack(">%ds%df1d" % (id_len, factors), record)

    u_id = unpacked[0].decode("utf-8").strip()
    emb = np.asarray(unpacked[1 : (factors + 1)], dtype=dtype)
    val = unpacked[-1]
    return u_id, emb, val


def write_text(filename, ids, embs, vals):
    start_time = time.time()

    with open(filename, "w") as f:
        for i in range(len(ids)):
            try:
                data = (ids[i], embs[i], vals[i])
                f.write(export_text(data))
            except Exception:
                logger.warning("fail to write emb for %s", ids[i])

    elapsed = time.time() - start_time
    logger.info("saved text to %s in %.2f min. throughput %.2f records/s", filename, elapsed / 60, len(ids) / elapsed)


def read_text(filename):
    start_time = time.time()

    ids = []
    embs = []
    vals = []

    cnt = 0
    with open(filename, "r") as f:
        for line in f:
            cnt += 1
            try:
                u_id, emb, val = load_text(line)
                ids.append(u_id)
                embs.append(emb)
                vals.append(val)
            except Exception:
                logger.warning("fail to read line %s", line)

    elapsed = time.time() - start_time
    logger.info(
        "read text from %s in %.2f min. total %s records. throughput %.2f records/s",
        filename,
        elapsed / 60,
        cnt,
        cnt / elapsed,
    )

    return ids, np.asarray(embs), vals


def write_text_in_parallel(filename, ids, embs, vals, workers=8):
    start_time = time.time()

    with Pool(processes=workers) as pool:
        pool_iter = pool.imap(export_text, zip(ids, embs, vals), chunksize=256)

        with open(filename, "w") as f:
            for line in pool_iter:
                try:
                    f.write(line)
                except Exception:
                    logger.warning("fail to write line %s", line)

    elapsed = time.time() - start_time
    logger.info(
        "saved text to %s in %.2f min. total %s records. throughput %.2f records/s",
        filename,
        elapsed / 60,
        len(ids),
        len(ids) / elapsed,
    )


def write_binary(filename, ids, embs, vals, id_len=32):
    start_time = time.time()

    with open(filename, "wb") as f:
        for i in range(len(ids)):
            try:
                data = (ids[i], embs[i], vals[i], id_len)
                f.write(export_binary(data))
            except Exception:
                logger.warning("fail to write emb for %s", ids[i])

    elapsed = time.time() - start_time
    logger.info(
        "saved binary to %s in %.2f min. total %s records. throughput %.2f records/s",
        filename,
        elapsed / 60,
        len(ids),
        len(ids) / elapsed,
    )


def read_binary(filename, id_len=32, factors=96):
    start_time = time.time()

    ids = []
    embs = []
    vals = []

    cnt = 0
    record_size = id_len * BYTE + factors * 4 * BYTE + 1 * 8 * BYTE
    with open(filename, "rb") as f:
        record_bin = f.read(record_size)

        while record_bin:
            cnt += 1
            try:
                u_id, emb, val = load_binary((record_bin, id_len, factors))
                ids.append(u_id)
                embs.append(emb)
                vals.append(val)
            except Exception:
                logger.exception("fail to read record")

            record_bin = f.read(record_size)

    elapsed = time.time() - start_time
    logger.info(
        "read binary from %s in %.2f min. total %s records. throughput %.2f records/s",
        filename,
        elapsed / 60,
        cnt,
        cnt / elapsed,
    )

    return ids, np.asarray(embs), vals


def write_binary_in_parallel(filename, ids, embs, vals, workers=8, id_len=32):
    start_time = time.time()

    id_lens = [id_len] * len(ids)

    with Pool(processes=workers) as pool:
        pool_iter = pool.imap(export_binary, zip(ids, embs, vals, id_lens), chunksize=256)
        with open(filename, "wb") as f:
            for record in pool_iter:
                try:
                    f.write(record)
                except Exception:
                    logger.warning("fail to write record %s", record.decode())

    elapsed = time.time() - start_time
    logger.info(
        "saved binary to %s in %.2f min. total %s records. throughput %.2f records/s",
        filename,
        elapsed / 60,
        len(ids),
        len(ids) / elapsed,
    )
