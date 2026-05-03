import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import AuthorizationException

load_dotenv()

def get_client() -> Elasticsearch:
    """
    Initializes and returns an Elasticsearch client using credentials from environment variables.
    
    Returns:
        Elasticsearch: An instance of the Elasticsearch client.
    """
    url = os.getenv("ES_URL","http://localhost:9200")
    api_key = os.getenv("ES_API_KEY")
    if api_key:
        return Elasticsearch(url, api_key=api_key)
    return Elasticsearch(url)

def check_connection(es: Elasticsearch) -> bool:
    try:
        exists = es.indices.exists(index="catalog")
        if exists:
            print("ES connected - catalog index found")
        else:
            print("ES connected - catalog index not found")
        return True
    
    except AuthorizationException:
        print("ES connection failed - authorization error")
        return False
    except Exception as e:
        print(f"ES connection failed - {e}")
        return False