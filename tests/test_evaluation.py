import math
from evaluate import (
    dcg,
    ideal_dcg,
    ndcg_at_k,
    mrr,
    evaluate_ranking,
    paired_bootstrap_ci,
    recall_at_k,
    strongest_baseline,
    _metrics_for_case,
)



def test_ndcg_perfect_ranking():
    relevances = [3, 2, 1, 0]
    assert ndcg_at_k(relevances, 4) == 1.0


def test_ndcg_worst_ranking():
    relevances = [0, 0, 0, 3]
    score = ndcg_at_k(relevances, 4)
    assert score < 0.5


def test_ndcg_all_zeros():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_single_relevant():
    relevances = [2, 0, 0]
    assert ndcg_at_k(relevances, 3) == 1.0


def test_missing_relevant_product_lowers_case_ndcg():
    case = {"judgments": {"found": 3, "missed": 3}}
    hits = [{"_source": {"item_id": "found"}}]

    assert _metrics_for_case(case, hits)["ndcg@10"] == dcg([3], 10) / ideal_dcg([3, 3], 10)


def test_mrr_first_position():
    assert mrr([1, 0, 0]) == 1.0


def test_mrr_second_position():
    assert abs(mrr([0, 1, 0]) - 0.5) < 0.001


def test_mrr_no_relevant():
    assert mrr([0, 0, 0]) == 0.0


def test_dcg_known_values():
    relevances = [3, 2, 1]
    expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert abs(dcg(relevances, 3) - expected) < 0.001


def test_evaluate_ranking_structure():
    groups = {
        1: [(2, 5.0), (1, 3.0), (0, 1.0)],
        2: [(0, 4.0), (2, 2.0), (1, 1.0)],
    }
    result = evaluate_ranking(groups)
    assert "ndcg@10" in result
    assert "ndcg@5" in result
    assert "mrr" in result
    assert "recall@100" in result
    assert 0.0 <= result["ndcg@10"] <= 1.0
    assert 0.0 <= result["ndcg@5"] <= 1.0
    assert 0.0 <= result["mrr"] <= 1.0
    assert 0.0 <= result["recall@100"] <= 1.0


def test_recall_counts_relevant_items_retrieved():
    assert recall_at_k([3, 0, 2], {"a": 3, "b": 2, "c": 2}, 3) == 2 / 3


def test_paired_bootstrap_ci_detects_consistent_lift():
    ci = paired_bootstrap_ci([0.4, 0.5, 0.6], [0.3, 0.4, 0.5], samples=500, seed=42)
    assert ci["lower"] > 0
    assert ci["mean"] == 0.1


def test_strongest_baseline_is_selected_from_per_query_scores():
    mode_rows = {
        "bm25": [{"ndcg@10": 0.7}],
        "semantic": [{"ndcg@10": 0.9}],
        "hybrid": [{"ndcg@10": 0.8}],
    }
    assert strongest_baseline(mode_rows) == "semantic"
