import os
import pytest
from fastapi.testclient import TestClient
from api import app
from shared.es_client import get_client
from shared.indexer import bulk_index
from catalog import generate_catalog, create_index

client = TestClient(app)

KNOWN_ITEMS = [
    {
        "item_id": "test_search_001",
        "title": "Sony Wireless Headphones",
        "description": "Premium noise cancelling wireless headphones",
        "category": "electronics",
        "brand": "Sony",
        "price": 299.99,
    },
    {
        "item_id": "test_search_002",
        "title": "Nike Running Shoes",
        "description": "Lightweight shoes for marathon training",
        "category": "sports",
        "brand": "Nike",
        "price": 129.99,
    },
    {
        "item_id": "test_search_003",
        "title": "Coffee Maker Deluxe",
        "description": "12 cup programmable coffee maker for home use",
        "category": "home",
        "brand": "Cuisinart",
        "price": 79.99,
    },
]


@pytest.fixture(scope="module")
def es_with_test_data():
    if not os.getenv("ES_URL"):
        pytest.skip("ES_URL not set")
    es = get_client()
    create_index(es)
    bulk_index(es, "catalog", KNOWN_ITEMS)
    es.indices.refresh(index="catalog")
    yield es
    es.delete_by_query(
        index="catalog",
        body={"query": {"terms": {"item_id.keyword": [
            "test_search_001", "test_search_002", "test_search_003"
        ]}}},
    )


def test_search_response_shape(es_with_test_data):
    resp = client.get("/search?q=headphones")
    assert resp.status_code == 200
    data = resp.json()
    assert "query" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_result_fields(es_with_test_data):
    resp = client.get("/search?q=headphones")
    data = resp.json()
    assert len(data["results"]) > 0
    first = data["results"][0]
    for field in ["item_id", "title", "category", "brand", "price", "score"]:
        assert field in first, f"Missing field: {field}"


def test_relevant_result_ranks_high(es_with_test_data):
    resp = client.get("/search?q=wireless headphones")
    data = resp.json()
    titles = [r["title"] for r in data["results"]]
    assert any("Headphones" in t or "headphones" in t for t in titles[:3]), \
        "Expected headphones result in top 3"


def test_pagination_no_overlap(es_with_test_data):
    page1 = client.get("/search?q=shoes&page=1&page_size=5").json()
    page2 = client.get("/search?q=shoes&page=2&page_size=5").json()
    ids1 = {r["item_id"] for r in page1["results"]}
    ids2 = {r["item_id"] for r in page2["results"]}
    assert ids1.isdisjoint(ids2), "Pages should not share results"