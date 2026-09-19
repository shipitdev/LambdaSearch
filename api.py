import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from catalog import load_catalog
from shared.es_client import check_connection, get_client
from shared.features import FEATURE_SCHEMA_VERSION, compute_category_stats
from shared.model_loader import assert_model_compatible, load_model, load_model_metadata
from shared.reranking import rerank_hits
from shared.search import build_search_request

es = None
category_stats = None
Mode = Literal["bm25", "semantic", "hybrid", "ltr"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global es, category_stats
    es = get_client()
    if not check_connection(es):
        raise RuntimeError("Cannot establish a connection to Elasticsearch")
    category_stats = compute_category_stats(load_catalog())
    yield


app = FastAPI(title="LaMDaSearch Ranking Lab", lifespan=lifespan)
origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])


@app.get("/health")
def health():
    if es is None:
        raise HTTPException(status_code=503, detail="Search backend unavailable")
    return {"status": "ok"}


def _rerank(query: str, hits: list[dict]) -> list[dict]:
    try:
        import xgboost as xgb

        assert_model_compatible(load_model_metadata(), FEATURE_SCHEMA_VERSION)
        model = load_model()
        return rerank_hits(query, hits, category_stats, lambda rows: model.predict(xgb.DMatrix(rows)))
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail="LTR model is not available") from error


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, max_length=256, description="Search query"),
    mode: Mode = "bm25",
    page: int = Query(1, ge=1, le=100),
    page_size: int = Query(10, ge=1, le=50),
):
    if es is None:
        raise HTTPException(status_code=503, detail="Search backend unavailable")
    try:
        response = es.options(request_timeout=5).search(
            index="catalog",
            body=build_search_request(q, mode, page_size, (page - 1) * page_size),
        )
    except Exception as error:
        raise HTTPException(status_code=503, detail="Search backend unavailable") from error

    hits = response["hits"]["hits"]
    if mode == "ltr":
        hits = _rerank(q, hits)
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
    return {"query": q, "mode": mode, "total": response["hits"]["total"]["value"], "page": page, "page_size": page_size, "results": results}
