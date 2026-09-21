# LaMDaSearch Benchmark Results

Held-out queries: 150

Benchmark: ranking-v1 (seed 42)

| Mode | NDCG@10 | NDCG@5 | MRR | Recall@100 |
|---|---:|---:|---:|---:|
| bm25 | 0.9126 | 0.9022 | 0.9617 | 0.3728 |
| semantic | 0.9939 | 0.9934 | 0.9956 | 0.3582 |
| hybrid | 0.9830 | 0.9885 | 0.9933 | 0.3565 |
| ltr | 0.9926 | 0.9956 | 0.9967 | 0.3565 |

## LTR vs Strongest First-Stage Baseline (semantic)

Paired bootstrap NDCG@10 lift (95% CI): -0.0013 [-0.0058, +0.0021]

NDCG uses all judged products for its ideal ranking, but only scores the first ten results. High NDCG@10 can therefore coexist with low Recall@100 when many products are equally relevant.

**Verdict: NOT READY**
