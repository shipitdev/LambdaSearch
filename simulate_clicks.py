import json
import os
import random
from shared.es_client import get_client

random.seed(42)

QUERIES = [
    "wireless headphones", "running shoes", "coffee maker", "yoga mat",
    "laptop bag", "winter jacket", "bluetooth speaker", "office chair",
    "kitchen knife set", "desk lamp", "hiking boots", "phone charger",
    "blender", "backpack", "sunglasses", "water bottle",
    "fitness tracker", "throw pillow", "table lamp", "rain jacket",
    "wireless mouse", "gaming keyboard", "air fryer", "yoga pants",
    "bookshelf", "tennis racket", "camping tent", "electric kettle",
    "novel paperback", "children's book", "cookbook", "notebook",
    "winter gloves", "summer dress", "denim jeans", "leather wallet",
    "smart watch", "tablet stand", "vacuum cleaner", "shower curtain",
    "bath towel set", "alarm clock", "wall art", "candle set",
    "board game", "puzzle 1000 piece", "basketball", "soccer ball",
    "dumbbell set", "resistance bands",
]


def position_bias(rank: int) -> float:
    return 1.0 / (1.0 + rank)


def load_popular_items(catalog_path: str = "data/catalog.json", fraction: float = 0.1) -> set:
    with open(catalog_path) as f:
        catalog = json.load(f)
    n_popular = int(len(catalog) * fraction)
    popular_ids = random.sample([item["item_id"] for item in catalog], n_popular)
    return set(popular_ids)


def simulate_session(es, query: str, popular_items: set, n_results: int = 10) -> list[dict]:
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

    events = []
    for rank, hit in enumerate(hits):
        item_id = hit["_source"].get("item_id")
        bm25_score = hit["_score"]

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


def run_simulation(es, n_sessions: int = 500) -> list[dict]:
    popular_items = load_popular_items()
    all_events = []
    for _ in range(n_sessions):
        query = random.choice(QUERIES)
        events = simulate_session(es, query, popular_items)
        all_events.extend(events)
    return all_events


def save_click_log(events: list[dict], path: str = "data/click_log.json") -> None:
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(events, f, indent=2)


if __name__ == "__main__":
    es = get_client()
    events = run_simulation(es)
    save_click_log(events)
    clicked = sum(1 for e in events if e["clicked"])
    print(f"Simulated {len(events)} events, {clicked} clicks ({clicked/len(events)*100:.1f}% CTR).")