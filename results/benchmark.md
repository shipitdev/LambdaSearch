# LaMDaSearch Benchmark Results

Held-out queries: 150

Benchmark: ranking-v1 (seed 42)

| Mode | NDCG@10 | NDCG@5 | MRR | Recall@100 |
|---|---:|---:|---:|---:|
| bm25 | 0.9084 | 0.9026 | 0.9683 | 0.3737 |
| semantic | 0.9939 | 0.9934 | 0.9956 | 0.3581 |
| hybrid | 0.9881 | 0.9907 | 0.9947 | 0.3565 |
| ltr | 0.9931 | 0.9936 | 0.9967 | 0.3565 |

## LTR vs Strongest First-Stage Baseline (semantic)

Paired bootstrap NDCG@10 lift (95% CI): -0.0008 [-0.0054, +0.0029]

**Verdict: NOT READY**
