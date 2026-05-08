import json


def query_length(query: str) -> int:
    return len(query.split())


def price_feature(item: dict) -> float:
    return float(item.get("price", 0.0))


def compute_category_stats(catalog: list[dict]) -> dict:
    totals = {}
    counts = {}
    for item in catalog:
        cat = item["category"]
        totals[cat] = totals.get(cat, 0.0) + item["price"]
        counts[cat] = counts.get(cat, 0) + 1
    return {cat: totals[cat] / counts[cat] for cat in totals}


def price_vs_category_avg(item: dict, category_stats: dict) -> float:
    avg = category_stats.get(item["category"])
    if avg is None or avg == 0:
        return 1.0
    return float(item["price"]) / avg


def bm25_score(hit: dict) -> float:
    return float(hit.get("bm25_score", 0.0))


def feature_vector(query: str, item: dict, bm25: float, category_stats: dict) -> list[float]:
    return [
        float(query_length(query)),
        price_feature(item),
        price_vs_category_avg(item, category_stats),
        bm25,
    ]