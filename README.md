# LaMDaSearch Ranking Lab

An evidence-first e-commerce search project for comparing lexical, semantic,
hybrid, and learning-to-rank retrieval. It is a portfolio lab, not a claim
that a model always outperforms BM25.

## Pipeline

```
BM25 + semantic retrieval → Elastic RRF fusion → LambdaMART reranking
```

The catalog and benchmark generator create deterministic product metadata and
graded relevance judgments independently of any retrieval rank. That separation
prevents the position-bias leakage present in click-derived labels.

## Modes

`GET /search?q=audio%20for%20commuting&mode=bm25|semantic|hybrid|ltr`

- `semantic` is the default because it is the strongest first-stage mode in the
  current held-out benchmark.
- `bm25` remains available as the lexical baseline.
- `semantic` and `hybrid` require an Elastic `semantic_text` inference endpoint.
- `ltr` is experimental: it reranks hybrid candidates with
  `models/model.json` and returns 503 when no compatible model is available.

## Setup

Requires Python 3.11 and Elastic Cloud. Configure an explicit semantic endpoint
instead of relying on a deployment default:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Set ES_URL, ES_API_KEY, and ELASTIC_INFERENCE_ID in .env
python catalog.py
uvicorn api:app --reload
```

Run local tests with:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -q
```

Cloud integration tests are opt-in and require reachable credentials:

```bash
RUN_ES_TESTS=1 PYTHONPATH=. .venv/bin/pytest tests/ -q
```

Build and assess the ranking model with the same deterministic catalog used by
the benchmark:

```bash
PYTHONPATH=. .venv/bin/python to_libsvm.py
PYTHONPATH=. .venv/bin/python train.py
PYTHONPATH=. .venv/bin/python evaluate.py
```

The final command writes `results/metrics.json` and `results/benchmark.md`.
`NOT READY` is a valid result: it means LTR did not demonstrate a statistically
reliable NDCG@10 improvement over the strongest of BM25, semantic, and hybrid
retrieval.
NDCG@10 uses every judged product to define the ideal top ten, but cannot
measure relevant products missed beyond that cutoff; read it alongside
Recall@100. Every held-out query currently has at least 19 highest-grade
products, which makes perfect top-ten scores possible despite low recall.

## Evaluation standard

The release gate is a reproducible held-out benchmark where LambdaMART has a
paired-bootstrap 95% confidence interval for NDCG@10 lift entirely above the
strongest first-stage baseline. Until that result exists, this repository is
GitHub-only and must not claim a hosted or production-ready service.

## Known limitations

- Synthetic judgments test pipeline integrity, not real customer behavior.
- Real click logs require propensity-aware collection before they can become
  trustworthy labels.
- A hosted demo is intentionally deferred until the release gate passes.

Licensed under [MIT](LICENSE).
