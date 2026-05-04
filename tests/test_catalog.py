import json
import os
import pytest
from catalog import generate_catalog, generate_item, load_catalog, create_index
from shared.es_client import get_client
from shared.indexer import bulk_index


def test_catalog_size():
    items = generate_catalog(100)
    assert len(items) == 100


def test_item_fields():
    item = generate_item(0)
    for field in ["item_id", "title", "description", "category", "brand", "price"]:
        assert field in item, f"Missing field: {field}"


def test_no_bad_values():
    items = generate_catalog(200)
    for item in items:
        assert item["price"] > 0
        assert len(item["title"]) > 0
        assert len(item["description"]) > 0
        assert item["category"] in ["electronics", "clothing", "home", "sports", "books"]


def test_category_distribution():
    items = generate_catalog(3000)
    categories = [i["category"] for i in items]
    electronics_ratio = categories.count("electronics") / len(items)
    books_ratio = categories.count("books") / len(items)
    assert 0.25 < electronics_ratio < 0.45
    assert 0.04 < books_ratio < 0.15


def test_deterministic_with_seed():
    items_a = generate_catalog(10)
    items_b = generate_catalog(10)
    assert items_a[0]["title"] == items_b[0]["title"]
    assert items_a[0]["price"] == items_b[0]["price"]


@pytest.mark.skipif(not os.getenv("ES_URL"), reason="ES_URL not set")
def test_bulk_index_and_retrieve():
    es = get_client()
    test_items = generate_catalog(10)
    for item in test_items:
        item["item_id"] = f"test_{item['item_id']}"

    create_index(es)
    count = bulk_index(es, "catalog", test_items)
    assert count == 10

    es.indices.refresh(index="catalog")
    result = es.count(index="catalog", body={
        "query": {"prefix": {"item_id.keyword": "test_"}}
    })
    assert result["count"] >= 10

    es.delete_by_query(index="catalog", body={
        "query": {"prefix": {"item_id.keyword": "test_"}}
    })