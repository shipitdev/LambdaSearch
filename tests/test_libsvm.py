import re

from to_libsvm import build_dataset, to_libsvm_row


CATALOG = [
    {"item_id": "1", "category": "electronics", "price": 100.0},
    {"item_id": "2", "category": "electronics", "price": 200.0},
]
CASES = [{"query": "wireless headphones", "judgments": {"1": 3, "2": 0}}]
HITS = [{"_score": 2.0, "_source": CATALOG[0]}, {"_score": 1.0, "_source": CATALOG[1]}]


def test_libsvm_row_format():
    row = to_libsvm_row(2, 1, [2.0, 99.99, 1.5, 4.2])
    assert re.match(r"^2 qid:1 0:\d+\.\d+ 1:\d+\.\d+ 2:\d+\.\d+ 3:\d+\.\d+$", row)


def test_dataset_uses_independent_judgments_and_hybrid_candidates():
    rows = build_dataset(CASES, CATALOG, lambda query: HITS)
    assert [row.split()[0] for row in rows] == ["3", "0"]
    assert {row.split()[1] for row in rows} == {"qid:1"}


def test_dataset_has_the_expected_feature_count():
    rows = build_dataset(CASES, CATALOG, lambda query: HITS)
    assert all(len([part for part in row.split() if re.match(r"^\d+:", part)]) == 4 for row in rows)
