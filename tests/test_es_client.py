import os
import pytest
from elasticsearch import Elasticsearch
from shared.es_client import get_client, check_connection

def test_get_client_returns_instance():
    client = get_client()
    assert isinstance(client, Elasticsearch)

def test_check_connection_fails_gracefully():
    bad_es = Elasticsearch("http://localhost:9200", request_timeout=2)
    result = check_connection(bad_es)
    assert result is False

@pytest.mark.skipif(os.getenv("ES_URL") is None, reason ="ES_URL not set in environment")

def test_check_connection_live():
    es = get_client()
    assert check_connection(es) is True