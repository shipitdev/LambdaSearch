import json
import os
import random
from shared.es_client import get_client

random.seed(42)


def position_bias(rank: int) -> float:
    return 1.0 / (1.0 + rank)


def load_popular_items(catalog_path: str = "data/catalog.json", fraction: float = 0.1) -> set:
    with open(catalog_path) as f:
        catalog = json.load(f)
    random.seed(42)
    n_popular = int(len(catalog) * fraction)
    popular_ids = random.sample([item["item_id"] for item in catalog], n_popular)
    return set(popular_ids)


def build_queries_from_catalog(catalog_path: str = "data/catalog.json") -> list[str]:
    with open(catalog_path) as f:
        catalog = json.load(f)

    queries = set()

    # brand names — guaranteed to appear in titles
    brands = set(item["brand"] for item in catalog)
    queries.update(brands)

    # categories — always keyword-matchable
    categories = set(item["category"] for item in catalog)
    queries.update(categories)

    # brand + category combos for variety
    for item in random.sample(catalog, min(30, len(catalog))):
        queries.add(f"{item['brand']} {item['category']}")

    return list(queries)


def prefetch_query_results(es, queries: list[str], n_results: int = 20) -> dict:
    cache = {}
    print(f"Pre-fetching results for {len(queries)} queries...")
    for i, query in enumerate(queries):
        resp = es.search(
            index="catalog",
            body={
                "size": n_results,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "description"],
                    }
                },
            },
        )
        hits = resp["hits"]["hits"]
        if hits:
            cache[query] = [
                {
                    "item_id": h["_source"].get("item_id"),
                    "bm25_score": h["_score"],
                }
                for h in hits
            ]
        print(f"  [{i+1}/{len(queries)}] '{query}' → {len(hits)} results")
    return cache


def simulate_session(query: str, cached_results: list, popular_items: set) -> list[dict]:
    events = []
    for rank, hit in enumerate(cached_results):
        item_id = hit["item_id"]
        bm25_score = hit["bm25_score"]

        prob = position_bias(rank)
        if item_id in popular_items:
            prob = min(prob * 2.0, 1.0)

        clicked = random.random() < prob

        events.append({
            "query": query,
            "item_id": item_id,
            "position": rank,
            "clicked": clicked,
            "bm25_score": bm25_score,
        })
    return events


def run_simulation(es, n_sessions: int = 3000) -> list[dict]:
    popular_items = load_popular_items()
    queries = build_queries_from_catalog()
    query_cache = prefetch_query_results(es, queries)

    valid_queries = [q for q in queries if q in query_cache and query_cache[q]]
    print(f"{len(valid_queries)}/{len(queries)} queries returned results from ES")

    if not valid_queries:
        raise RuntimeError("No queries returned results — is the catalog indexed?")

    random.seed(42)
    all_events = []
    for _ in range(n_sessions):
        query = random.choice(valid_queries)
        events = simulate_session(query, query_cache[query], popular_items)
        all_events.extend(events)
    return all_events


def save_click_log(events: list[dict], path: str = "data/click_log.json") -> None:
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved {len(events)} click events to {path}.")


if __name__ == "__main__":
    es = get_client()
    events = run_simulation(es)
    save_click_log(events)
    clicked = sum(1 for e in events if e["clicked"])
    print(f"Simulated {len(events)} events, {clicked} clicks ({clicked/len(events)*100:.1f}% CTR).")