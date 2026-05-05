import os
from contextlib import asynccontextmanager
from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware
from shared.es_client import get_client, check_connection

es = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global es
    es = get_client()
    if not check_connection(es):
        raise RuntimeError("Cannot establish a connection to ElasticSearch")
    yield

app = FastAPI(title="LambdaSearch", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
def search(
    q: str = Query(...,description="search query"),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1,le=50),
):
    from_ = (page - 1) * page_size

    resp = es.search(
        index="catalog",
        body = {
            "from" : from_,
            "size" : page_size,
            "query" : {
                "multi_match": {
                    "query": q,
                    "fields": ["title^2","description"],
                }
            }
        }
    )
    
    hits = resp["hits"]["hits"]
    total = resp["hits"]["total"]["value"]

    results = [
        {
            "item_id": hit["_source"].get("item_id"),
            "title": hit["_source"]["title"],
            "category": hit["_source"]["category"],
            "brand": hit["_source"]["brand"],
            "price": hit["_source"]["price"],
            "score": hit["_score"],
        }
        for hit in hits
    ]
    
    return {
        "query": q,
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": results,

    }