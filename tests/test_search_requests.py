import pytest

from shared.search import build_search_request


def test_bm25_request_uses_lexical_fields():
    request = build_search_request("wireless headphones", "bm25", 10, 0)

    assert request["query"]["multi_match"]["fields"] == ["title^2", "description", "search_text"]


def test_semantic_request_uses_semantic_text_field():
    request = build_search_request("audio for commuting", "semantic", 10, 0)

    assert request["retriever"]["standard"]["query"]["semantic"]["field"] == "search_text"


def test_hybrid_request_fuses_lexical_and_semantic_retrievers():
    request = build_search_request("audio for commuting", "hybrid", 10, 0)

    retrievers = request["retriever"]["rrf"]["retrievers"]
    assert len(retrievers) == 2
    assert request["retriever"]["rrf"]["rank_window_size"] == 100


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_search_request("test", "unknown", 10, 0)
