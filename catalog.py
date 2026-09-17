import json
import os
from shared.es_client import get_client
from shared.indexer import bulk_index
from shared.benchmark_data import generate_products

CATALOG_MAPPING = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "description": {"type": "text"},
            "search_text": {"type": "text"},
            "category": {"type": "keyword"},
            "brand": {"type": "keyword"},
            "product_type": {"type": "keyword"},
            "attributes": {"type": "keyword"},
            "use_cases": {"type": "keyword"},
            "price": {"type": "float"},
        }
    }
}

def generate_item(item_id: int) -> dict:
    return generate_products(item_id + 1)[item_id]

def generate_catalog(n: int=3000) -> list[dict]:
    return generate_products(n)

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
