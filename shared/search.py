"""Elasticsearch request builders for the ranking lab's retrieval stages."""

SEARCH_MODES = frozenset({"bm25", "semantic", "hybrid", "ltr"})
TEXT_QUERY = {"multi_match": {"fields": ["title^2", "description"]}}


def _lexical(query: str) -> dict:
    return {"multi_match": {"query": query, "fields": TEXT_QUERY["multi_match"]["fields"]}}


def _semantic(query: str) -> dict:
    return {"semantic": {"field": "search_text", "query": query}}


def build_search_request(query: str, mode: str, size: int, offset: int = 0) -> dict:
    """Build a native Elastic retriever request; LTR reuses hybrid candidates."""
    if mode not in SEARCH_MODES:
        raise ValueError(f"Unsupported search mode: {mode}")
    if mode == "bm25":
        return {"from": offset, "size": size, "query": _lexical(query)}
    if mode == "semantic":
        return {"from": offset, "size": size, "retriever": {"standard": {"query": _semantic(query)}}}

    return {
        "from": offset,
        "size": size,
        "retriever": {
            "rrf": {
                "retrievers": [
                    {"standard": {"query": _lexical(query)}},
                    {"standard": {"query": _semantic(query)}},
                ],
                "rank_window_size": 100,
                "rank_constant": 20,
            }
        },
    }
