# LaMDaSearch

A Learning-to-Rank (LTR) search engine for e-commerce, built to demonstrate
the two-stage retrieval architecture used in production search systems at scale.

BM25 (Elasticsearch) handles fast candidate retrieval. A LambdaMART model
trained on behavioral click signals handles re-ranking. The benchmark compares
both on NDCG@10 against a held-out query set the model never trained on.

## Architecture

```
Query
  │
  ▼
Elasticsearch BM25          ← fast candidate retrieval
  │
  ▼
Feature Extraction          ← query length, price, category avg, BM25 score
  │
  ▼
XGBoost LambdaMART          ← learned re-ranking (rank:ndcg objective)
  │
  ▼
Ranked Results
```

## Benchmark Results

| Metric  | BM25   | LambdaMART |
| ------- | ------ | ---------- |
| NDCG@10 | 1.0000 | 0.8205     |
| NDCG@5  | 1.0000 | 0.8205     |
| MRR     | 1.0000 | 1.0000     |

BM25 scores perfectly on this synthetic dataset because queries are brand and
category terms that match catalog titles exactly — the ideal case for keyword
search. LambdaMART's advantage over BM25 emerges with real user data where
queries are ambiguous, where popularity signals matter independent of text
match, and where personalization is possible. This pipeline demonstrates the
correct architecture for capturing those signals when real behavioral data is
available.

Full results: [results/benchmark.md](results/benchmark.md)

## Stack

| Layer          | Tool                             |
| -------------- | -------------------------------- |
| Search index   | Elasticsearch on Elastic Cloud   |
| API            | FastAPI                          |
| ML model       | XGBoost LambdaMART (`rank:ndcg`) |
| Synthetic data | Faker + custom click simulator   |
| Testing        | pytest                           |

## Project Structure

```
shared/
  es_client.py        # Elasticsearch connection (env-var based)
  indexer.py          # Bulk indexing with per-document error validation
  features.py         # Feature extraction: query_length, price,
                      # price_vs_category_avg, bm25_score
  model_loader.py     # XGBoost save/load (native JSON format)
catalog.py            # Generate 3000 synthetic products + index into ES
simulate_clicks.py    # Simulate user sessions with position + popularity bias
to_libsvm.py          # Convert click log to LibSVM format for XGBoost
train.py              # LambdaMART training pipeline
evaluate.py           # NDCG@10, NDCG@5, MRR benchmark
api.py                # FastAPI search endpoint (BM25)
results/
  metrics.json        # Raw benchmark numbers
  benchmark.md        # Benchmark report
```

## Setup

### Prerequisites

- Python 3.11+
- Elasticsearch on Elastic Cloud (free trial at cloud.elastic.co)

### Install

```bash
git clone https://github.com/yourusername/LaMDaSearch
cd LaMDaSearch
pip install -r requirements.txt
cp .env.example .env
# fill in ES_URL and ES_API_KEY in .env
```

### Run the pipeline end to end

```bash
# 1. Generate catalog and index into Elasticsearch
python3 catalog.py

# 2. Simulate user click behavior
python3 simulate_clicks.py

# 3. Convert to LibSVM training format
python3 to_libsvm.py

# 4. Train LambdaMART model
python3 train.py

# 5. Run benchmark evaluation
python3 evaluate.py

# 6. Start the search API
uvicorn api:app --reload
```

### Run tests

```bash
bash run_all_tests.sh
```

Integration tests require `ES_URL` and `ES_API_KEY` set in `.env` and skip
cleanly without them.

## API

### BM25 Search

```
GET /search?q=Sony&page=1&page_size=10
```

Returns results ranked by Elasticsearch BM25 scoring.

### Interactive docs

```
http://localhost:8000/docs
```

## Key Engineering Decisions

**Query-level train/holdout split** — the holdout set is split by query ID,
not by row. This prevents the same query appearing in both train and holdout,
which would leak information and inflate evaluation metrics.

**Graded relevance labels** — clicks at position 0-2 get label 3, clicks at
position 3+ get label 2, shown-but-not-clicked gets label 1. Graded labels
give XGBoost more signal than binary click/no-click.

**BM25 score as a feature** — the model uses ES's own BM25 score as one of
its four input features. This means LambdaMART learns when to trust BM25 and
when to override it based on other signals.

**Per-document bulk validation** — the ES bulk API returns HTTP 200 even when
individual documents fail. The indexer parses per-document errors and raises
explicitly rather than silently losing data.

**Pre-fetched query cache** — the click simulator pre-fetches BM25 results
for all queries once before simulation, reducing ES round trips from
n_sessions × n_queries to just n_queries. This makes large simulations
practical without hammering the ES cluster.

## Limitations and Honest Assessment

Click simulation uses BM25 results as the candidate pool, creating a
correlation between BM25 ordering and click labels. In production, LTR's
advantage comes from signals orthogonal to BM25 — user history, item
popularity, session context — which synthetic simulation cannot fully
replicate. The benchmark correctly reflects this: BM25 is essentially
optimal for exact keyword queries against a synthetic catalog, while
LambdaMART's edge emerges on ambiguous, real-world queries.

## TODO

- [ ] Redis feature store for sub-millisecond feature lookup at query time
- [ ] `/search/ltr` endpoint integrating model re-ranking into the API
- [ ] Semantic search via dense vector embeddings as an alternative first stage
- [ ] RAG integration — use LTR re-ranking to improve retrieval quality
      for LLM context, addressing the core weakness of naive RAG pipelines
- [ ] Hyperparameter tuning with cross-validation
- [ ] Real user feedback loop to replace synthetic click simulation
