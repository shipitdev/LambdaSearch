import os
import json
import pytest
from simulate_clicks import position_bias, load_popular_items, simulate_session, run_simulation
from shared.es_client import get_client


def test_position_bias_strictly_decreasing():
    values = [position_bias(r) for r in range(10)]
    for i in range(len(values) - 1):
        assert values[i] > values[i + 1], f"Bias should decrease: rank {i} vs {i+1}"


def test_position_bias_bounds():
    assert position_bias(0) == 1.0
    assert 0 < position_bias(9) < 1.0


@pytest.mark.skipif(not os.path.exists("data/catalog.json"), reason="catalog.json not generated yet")
def test_popular_items_fraction():
    popular = load_popular_items(fraction=0.1)
    with open("data/catalog.json") as f:
        catalog = json.load(f)
    expected = int(len(catalog) * 0.1)
    assert abs(len(popular) - expected) <= 1


@pytest.mark.skipif(not os.getenv("ES_URL"), reason="ES_URL not set")
@pytest.mark.skipif(not os.path.exists("data/catalog.json"), reason="catalog.json not generated yet")
def test_popular_items_higher_ctr():
    es = get_client()
    popular_items = load_popular_items(fraction=0.1)

    popular_clicks, popular_shown = 0, 0
    other_clicks, other_shown = 0, 0

    for _ in range(100):
        events = simulate_session(es, "wireless headphones", popular_items)
        for e in events:
            if e["position"] != 0:
                continue  # control for rank, only compare same position
            if e["item_id"] in popular_items:
                popular_shown += 1
                popular_clicks += int(e["clicked"])
            else:
                other_shown += 1
                other_clicks += int(e["clicked"])

    if popular_shown > 5 and other_shown > 5:
        popular_ctr = popular_clicks / popular_shown
        other_ctr = other_clicks / other_shown
        assert popular_ctr >= other_ctr, "Popular items should have CTR >= non-popular at same rank"

