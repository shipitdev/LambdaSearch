"""Shared online reranking logic; feature construction matches training."""

from collections.abc import Callable, Sequence

from shared.features import feature_vector


def rerank_hits(
    query: str,
    hits: list[dict],
    category_stats: dict,
    predict: Callable[[list[list[float]]], Sequence[float]],
) -> list[dict]:
    rows = [feature_vector(query, hit["_source"], hit.get("_score", 0.0), category_stats) for hit in hits]
    scores = predict(rows)
    ranked = []
    for hit, score in zip(hits, scores):
        ranked.append({**hit, "_score": float(score)})
    return sorted(ranked, key=lambda hit: hit["_score"], reverse=True)
