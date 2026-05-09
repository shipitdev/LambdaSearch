import re
import random
import pytest
from to_libsvm import compute_labels, to_libsvm_row, build_dataset, split_and_write


SAMPLE_CLICK_LOG = [
    {"query": "headphones", "item_id": "1", "position": 0, "clicked": True,  "bm25_score": 4.2},
    {"query": "headphones", "item_id": "2", "position": 1, "clicked": False, "bm25_score": 3.1},
    {"query": "headphones", "item_id": "3", "position": 2, "clicked": False, "bm25_score": 2.5},
    {"query": "shoes",      "item_id": "4", "position": 0, "clicked": True,  "bm25_score": 5.0},
    {"query": "shoes",      "item_id": "5", "position": 1, "clicked": False, "bm25_score": 3.8},
]

SAMPLE_CATALOG = [
    {"item_id": "1", "title": "A", "category": "electronics", "brand": "Sony",   "price": 100.0},
    {"item_id": "2", "title": "B", "category": "electronics", "brand": "LG",     "price": 200.0},
    {"item_id": "3", "title": "C", "category": "electronics", "brand": "Apple",  "price": 150.0},
    {"item_id": "4", "title": "D", "category": "sports",      "brand": "Nike",   "price": 80.0},
    {"item_id": "5", "title": "E", "category": "sports",      "brand": "Adidas", "price": 60.0},
]


def test_compute_labels_clicked():
    labels = compute_labels(SAMPLE_CLICK_LOG)
    assert labels[("headphones", "1")] == 2
    assert labels[("shoes", "4")] == 2


def test_compute_labels_not_clicked():
    labels = compute_labels(SAMPLE_CLICK_LOG)
    assert labels[("headphones", "2")] == 1
    assert labels[("shoes", "5")] == 1


def test_libsvm_row_format():
    row = to_libsvm_row(2, 1, [2.0, 99.99, 1.5, 4.2])
    assert re.match(r"^2 qid:1 0:\d+\.\d+ 1:\d+\.\d+ 2:\d+\.\d+ 3:\d+\.\d+$", row), \
        f"Bad format: {row}"


def test_libsvm_row_label_and_qid():
    row = to_libsvm_row(1, 42, [1.0, 2.0, 3.0, 4.0])
    parts = row.split()
    assert parts[0] == "1"
    assert parts[1] == "qid:42"


def test_qids_contiguous():
    rows = build_dataset(SAMPLE_CLICK_LOG, SAMPLE_CATALOG)
    qids = [int(r.split()[1].split(":")[1]) for r in rows]
    seen = []
    for qid in qids:
        if not seen or seen[-1] != qid:
            seen.append(qid)
    assert len(seen) == len(set(qids)), "Same qid appeared non-contiguously"


def test_no_query_in_both_splits():
    rows = build_dataset(SAMPLE_CLICK_LOG, SAMPLE_CATALOG)
    split_and_write(rows, SAMPLE_CLICK_LOG,
                    holdout_ratio=0.5, seed=42)
    with open("data/train.libsvm") as f:
        train_qids = {int(r.split()[1].split(":")[1]) for r in f if r.strip()}
    with open("data/holdout.libsvm") as f:
        holdout_qids = {int(r.split()[1].split(":")[1]) for r in f if r.strip()}
    assert train_qids.isdisjoint(holdout_qids), "Query leaked between train and holdout"


def test_feature_count_per_row():
    rows = build_dataset(SAMPLE_CLICK_LOG, SAMPLE_CATALOG)
    for row in rows:
        parts = row.split()
        features = [p for p in parts if re.match(r"^\d+:", p)]
        assert len(features) == 4, f"Expected 4 features, got {len(features)}"