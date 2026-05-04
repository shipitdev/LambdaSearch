import json
import os
import random

from faker import Faker
from shared.es_client import get_client
from shared.indexer import bulk_index

fake = Faker()
Faker.seed(42)
random.seed(42)

CATEGORIES = {
    "electronics": {
        "weight": 0.35,
        "price_range": (29.99, 1299.99),
        "brands": ["Sony", "Samsung", "Apple", "Bose", "LG",
                   "JBL", "Anker", "Logitech", "Razer", "Jabra"],
    },
    "clothing": {
        "weight": 0.25,
        "price_range": (9.99, 299.99),
        "brands": ["Nike", "Adidas", "Zara", "H&M", "Levi's",
                   "Uniqlo", "Gap", "Puma", "Reebok", "Calvin Klein"],
    },
    "home": {
        "weight": 0.20,
        "price_range": (14.99, 599.99),
        "brands": ["IKEA", "Dyson", "KitchenAid", "Cuisinart", "Philips",
                   "Black+Decker", "Instant Pot", "OXO", "Rubbermaid", "Vitamix"],
    },
    "sports": {
        "weight": 0.12,
        "price_range": (12.99, 499.99),
        "brands": ["Nike", "Adidas", "Under Armour", "Wilson", "Callaway",
                   "Yeti", "CamelBak", "Garmin", "Fitbit", "Patagonia"],
    },
    "books": {
        "weight": 0.08,
        "price_range": (4.99, 59.99),
        "brands": ["Penguin", "HarperCollins", "Random House", "Scholastic",
                   "Wiley", "O'Reilly", "Springer", "MIT Press", "Oxford", "Cambridge"],
    },
}

CATEGORY_NAMES = list(CATEGORIES.keys())
CATEGORY_WEIGHTS = [CATEGORIES[c]["weight"] for c in CATEGORY_NAMES]

CATALOG_MAPPING = {
    "mappings": {
        "properties": {
            "title:": {"type": "text"},
            "description": {"type": "text"},
            "category": {"type": "keyword"},
            "brand": {"type": "keyword"},
            "price": {"type": "float"},
        }
    }
}

def generate_item(item_id: int) -> dict:
    category = random.choices(CATEGORY_NAMES, weights=CATEGORY_WEIGHTS, k=1)[0]
    cfg = CATEGORIES[category]
    brand = random.choice(cfg["brands"])
    low,high = cfg["price_range"]
    price = round(random.uniform(low, high), 2)
    title = f"{brand} {fake.catch_phrase()}"
    description = fake.paragraph(nb_sentences=5)
    return {
        "item_id": str(item_id),
        "title": title,
        "description": description,
        "category": category,
        "brand": brand,
        "price": price,
    }

def generate_catalog(n: int=3000) -> list[dict]:
    Faker.seed(42)
    random.seed(42)
    return [generate_item(i) for i in range(n)]

def create_index(es) -> None:
    if es.indices.exists(index="catalog"):
        print("Index 'catalog' already exists. skipping creating it.")
        return
    es.indices.create(index="catalog", body=CATALOG_MAPPING)
    print('Index "catalog" created successfully.')

def save_catalog(items: list[dict], path: str="data/catalog.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)
        print(f"Saved {len(items)} items to {path}")

def load_catalog(path: str="data/catalog.json") -> list[dict]:
    with open(path,"r") as f:
        return json.load(f)
    
if __name__ == "__main__":
    items = generate_catalog()
    save_catalog(items)

    es = get_client()
    create_index(es)
    indexed = bulk_index(es, "catalog", items)
    print(f"Indexed {indexed} items into Elasticsearch.")