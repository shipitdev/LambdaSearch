from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

def bulk_index(es: Elasticsearch, index: str, docs: list[dict]) -> int:
    actions = [{"_index": index, "_source": doc} for doc in docs]
    success, errors = bulk(es, actions)
    if errors:
        raise ValueError(f"Bulk indexing had {len(errors)} errors: {errors[:3]}")
    return success