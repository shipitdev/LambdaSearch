from shared.reranking import rerank_hits


def test_reranker_orders_hybrid_candidates_by_model_prediction():
    hits = [
        {"_score": 4.0, "_source": {"item_id": "a", "category": "electronics", "price": 100.0}},
        {"_score": 2.0, "_source": {"item_id": "b", "category": "electronics", "price": 200.0}},
    ]

    reranked = rerank_hits("wireless headphones", hits, {"electronics": 150.0}, lambda rows: [0.1, 0.9])

    assert [hit["_source"]["item_id"] for hit in reranked] == ["b", "a"]
    assert reranked[0]["_score"] == 0.9
