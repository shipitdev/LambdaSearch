"""Deterministic catalog and relevance data for the ranking benchmark."""

from __future__ import annotations

import random
from collections import defaultdict


PRODUCTS = (
    ("electronics", "headphones", ("wireless", "noise cancelling", "bluetooth"), ("travel", "commuting", "music")),
    ("electronics", "speaker", ("portable", "bluetooth", "waterproof"), ("outdoors", "parties", "travel")),
    ("electronics", "smartwatch", ("fitness", "gps", "heart rate"), ("running", "workouts", "health")),
    ("clothing", "running shoes", ("lightweight", "breathable", "cushioned"), ("running", "marathons", "training")),
    ("clothing", "winter jacket", ("waterproof", "insulated", "hooded"), ("hiking", "cold weather", "travel")),
    ("home", "coffee maker", ("programmable", "thermal", "compact"), ("breakfast", "office", "home")),
    ("home", "air purifier", ("quiet", "hepa", "compact"), ("allergies", "bedroom", "home")),
    ("sports", "yoga mat", ("non slip", "thick", "portable"), ("yoga", "stretching", "home workouts")),
    ("sports", "camping tent", ("waterproof", "lightweight", "two person"), ("camping", "hiking", "outdoors")),
    ("books", "python book", ("beginner", "practical", "programming"), ("learning", "coding", "career")),
)

BRANDS = ("Aster", "Boreal", "Cedar", "Drift", "Ember", "Fjord", "Grove", "Harbor")
PRICE_RANGES = {
    "electronics": (39.0, 499.0),
    "clothing": (29.0, 249.0),
    "home": (25.0, 399.0),
    "sports": (20.0, 299.0),
    "books": (12.0, 75.0),
}


def generate_products(n: int = 3000, seed: int = 42) -> list[dict]:
    """Create products whose relevance can be judged without retrieval scores."""
    rng = random.Random(seed)
    products = []
    for item_id in range(n):
        category, product_type, attributes, use_cases = PRODUCTS[item_id % len(PRODUCTS)]
        brand = BRANDS[(item_id // len(PRODUCTS)) % len(BRANDS)]
        low, high = PRICE_RANGES[category]
        selected_attributes = rng.sample(attributes, k=2)
        selected_uses = rng.sample(use_cases, k=2)
        title = f"{brand} {' '.join(selected_attributes)} {product_type}"
        search_text = " ".join((title, category, *selected_attributes, *selected_uses))
        products.append(
            {
                "item_id": str(item_id),
                "title": title,
                "description": f"A {product_type} for {' and '.join(selected_uses)}.",
                "category": category,
                "brand": brand,
                "product_type": product_type,
                "attributes": selected_attributes,
                "use_cases": selected_uses,
                "search_text": search_text,
                "price": round(rng.uniform(low, high), 2),
            }
        )
    return products


def _case(query: str, cohort: str, products: list[dict], predicate) -> dict:
    judgments = {}
    for product in products:
        label = predicate(product)
        if label:
            judgments[product["item_id"]] = label
    return {"query": query, "cohort": cohort, "judgments": judgments}


def _query_cases(products: list[dict]) -> list[dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for product in products:
        by_type[product["product_type"]].append(product)

    cases = []
    for product_type, matches in by_type.items():
        representative = matches[0]
        brand = representative["brand"]
        attribute = representative["attributes"][0]
        use_case = representative["use_cases"][0]
        cases.extend(
            (
                _case(
                    f"{brand} {product_type}",
                    "lexical",
                    products,
                    lambda p, product_type=product_type, brand=brand: 3 if p["product_type"] == product_type and p["brand"] == brand else (2 if p["product_type"] == product_type else 0),
                ),
                _case(
                    f"{attribute} {product_type}",
                    "attribute",
                    products,
                    lambda p, product_type=product_type, attribute=attribute: 3 if p["product_type"] == product_type and attribute in p["attributes"] else (2 if p["product_type"] == product_type else 0),
                ),
                _case(
                    f"something for {use_case}",
                    "semantic",
                    products,
                    lambda p, use_case=use_case: 3 if use_case in p["use_cases"] else 0,
                ),
            )
        )
    return cases


def build_benchmark(products: list[dict], seed: int = 42, queries_per_cohort: int = 50) -> dict:
    """Return query-disjoint train/dev/test splits with metadata-derived labels."""
    rng = random.Random(seed)
    cases_by_cohort: dict[str, list[dict]] = defaultdict(list)
    for case in _query_cases(products):
        cases_by_cohort[case["cohort"]].append(case)

    splits = {"train": [], "dev": [], "test": []}
    for cohort, cases in cases_by_cohort.items():
        rng.shuffle(cases)
        selected = cases[: min(queries_per_cohort, len(cases))]
        train_end = max(1, len(selected) // 2)
        dev_end = min(len(selected) - 1, max(train_end, (len(selected) * 3) // 4))
        splits["train"].extend(selected[:train_end])
        splits["dev"].extend(selected[train_end:dev_end])
        splits["test"].extend(selected[dev_end:])
    return splits
