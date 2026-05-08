from shared.features import (
    query_length,
    price_feature,
    compute_category_stats,
    price_vs_category_avg,
    bm25_score,
    feature_vector,
)

SAMPLE_CATALOG = [
    {"item_id": "1", "title": "A", "category": "electronics", "brand": "Sony", "price": 100.0},
    {"item_id": "2", "title": "B", "category": "electronics", "brand": "LG", "price": 200.0},
    {"item_id": "3", "title": "C", "category": "books", "brand": "Penguin", "price": 20.0},
    {"item_id": "4", "title": "D", "category": "books", "brand": "Oxford", "price": 40.0},
]

CATEGORY_STATS = compute_category_stats(SAMPLE_CATALOG)


def test_query_length_single():
    assert query_length("shoes") == 1


def test_query_length_multi():
    assert query_length("wireless noise cancelling headphones") == 4


def test_query_length_extra_spaces():
    assert query_length("running  shoes") == 2


def test_price_feature_normal():
    item = {"price": 49.99}
    assert price_feature(item) == 49.99


def test_price_feature_missing():
    assert price_feature({}) == 0.0


def test_category_stats_values():
    assert CATEGORY_STATS["electronics"] == 150.0
    assert CATEGORY_STATS["books"] == 30.0


def test_price_vs_category_avg_above():
    item = {"category": "electronics", "price": 300.0}
    result = price_vs_category_avg(item, CATEGORY_STATS)
    assert abs(result - 2.0) < 0.001


def test_price_vs_category_avg_below():
    item = {"category": "electronics", "price": 75.0}
    result = price_vs_category_avg(item, CATEGORY_STATS)
    assert abs(result - 0.5) < 0.001


def test_price_vs_category_avg_unknown_category():
    item = {"category": "unknown", "price": 50.0}
    assert price_vs_category_avg(item, CATEGORY_STATS) == 1.0


def test_bm25_score_normal():
    event = {"bm25_score": 3.75}
    assert bm25_score(event) == 3.75


def test_bm25_score_missing():
    assert bm25_score({}) == 0.0


def test_feature_vector_length():
    item = {"category": "electronics", "price": 100.0}
    vec = feature_vector("wireless headphones", item, 4.2, CATEGORY_STATS)
    assert len(vec) == 4


def test_feature_vector_order():
    item = {"category": "electronics", "price": 100.0}
    vec = feature_vector("wireless headphones", item, 4.2, CATEGORY_STATS)
    assert vec[0] == 2.0                          # query_length
    assert vec[1] == 100.0                        # price
    assert abs(vec[2] - (100.0 / 150.0)) < 0.001 # price_vs_category_avg
    assert vec[3] == 4.2                          # bm25_score


def test_feature_vector_all_floats():
    item = {"category": "books", "price": 20.0}
    vec = feature_vector("book", item, 1.1, CATEGORY_STATS)
    assert all(isinstance(v, float) for v in vec)