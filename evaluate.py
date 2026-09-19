"""Independent, reproducible evaluation for the ranking lab."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb

from catalog import load_catalog
from shared.benchmark_data import build_benchmark
from shared.features import compute_category_stats
from shared.model_loader import load_model
from shared.reranking import rerank_hits
from shared.search import build_search_request


def dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(position + 2) for position, rel in enumerate(relevances[:k]))


def ideal_dcg(relevances: list[int], k: int) -> float:
    return dcg(sorted(relevances, reverse=True), k)


def ndcg_at_k(relevances: list[int], k: int) -> float:
    denominator = ideal_dcg(relevances, k)
    return 0.0 if denominator == 0 else dcg(relevances, k) / denominator


def mrr(relevances: list[int]) -> float:
    for position, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            return 1.0 / position
    return 0.0


def recall_at_k(relevances: list[int], judgments: dict[str, int], k: int) -> float:
    total_relevant = sum(label > 0 for label in judgments.values())
    return 0.0 if total_relevant == 0 else sum(label > 0 for label in relevances[:k]) / total_relevant


def evaluate_ranking(qid_groups: dict) -> dict:
    """Compatibility helper for small unit tests of score ordering."""
    ranked = [sorted(pairs, key=lambda pair: pair[1], reverse=True) for pairs in qid_groups.values()]
    relevances = [[label for label, _ in pairs] for pairs in ranked]
    return {
        "ndcg@10": round(float(np.mean([ndcg_at_k(values, 10) for values in relevances])), 4),
        "ndcg@5": round(float(np.mean([ndcg_at_k(values, 5) for values in relevances])), 4),
        "mrr": round(float(np.mean([mrr(values) for values in relevances])), 4),
        "recall@100": 0.0,
    }


def paired_bootstrap_ci(
    treatment: list[float], baseline: list[float], samples: int = 10_000, seed: int = 42
) -> dict:
    if len(treatment) != len(baseline) or not treatment:
        raise ValueError("treatment and baseline must contain the same non-zero number of queries")
    differences = np.asarray(treatment) - np.asarray(baseline)
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(0, len(differences), size=(samples, len(differences)))].mean(axis=1)
    return {
        "mean": round(float(differences.mean()), 10),
        "lower": round(float(np.quantile(draws, 0.025)), 10),
        "upper": round(float(np.quantile(draws, 0.975)), 10),
    }


def _search(es, query: str, mode: str, size: int = 100) -> list[dict]:
    response = es.options(request_timeout=10).search(
        index="catalog", body=build_search_request(query, mode, size)
    )
    return response["hits"]["hits"]


def _metrics_for_case(case: dict, hits: list[dict]) -> dict:
    labels = [case["judgments"].get(hit["_source"].get("item_id"), 0) for hit in hits]
    return {
        "ndcg@10": ndcg_at_k(labels, 10),
        "ndcg@5": ndcg_at_k(labels, 5),
        "mrr": mrr(labels),
        "recall@100": recall_at_k(labels, case["judgments"], 100),
    }


def _summarize(per_query: list[dict]) -> dict:
    return {
        metric: round(float(np.mean([row[metric] for row in per_query])), 4)
        for metric in ("ndcg@10", "ndcg@5", "mrr", "recall@100")
    }


def run_evaluation(es, model_path: str | Path = "models/model.json") -> dict:
    """Evaluate all modes on the same independent, held-out query set."""
    catalog = load_catalog()
    cases = build_benchmark(catalog)["test"]
    category_stats = compute_category_stats(catalog)
    model = load_model(model_path)
    modes = {"bm25": [], "semantic": [], "hybrid": [], "ltr": []}

    for case in cases:
        hybrid_hits = None
        for mode in ("bm25", "semantic", "hybrid"):
            hits = _search(es, case["query"], mode)
            modes[mode].append({"query": case["query"], "cohort": case["cohort"], **_metrics_for_case(case, hits)})
            if mode == "hybrid":
                hybrid_hits = hits
        ranked_hits = rerank_hits(
            case["query"], hybrid_hits, category_stats,
            lambda rows: model.predict(xgb.DMatrix(rows)),
        )
        modes["ltr"].append({"query": case["query"], "cohort": case["cohort"], **_metrics_for_case(case, ranked_hits)})

    results = {mode: _summarize(rows) for mode, rows in modes.items()}
    results["cohorts"] = {
        cohort: {mode: _summarize([row for row in rows if row["cohort"] == cohort]) for mode, rows in modes.items()}
        for cohort in ("lexical", "attribute", "semantic")
    }
    results["ltr_vs_hybrid_ndcg@10"] = paired_bootstrap_ci(
        [row["ndcg@10"] for row in modes["ltr"]],
        [row["ndcg@10"] for row in modes["hybrid"]],
    )
    results["passed"] = results["ltr_vs_hybrid_ndcg@10"]["lower"] > 0
    results["queries"] = len(cases)
    return results


def write_results(results: dict, output_dir: str | Path = "results") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = "\n".join(
        f"| {mode} | {metrics['ndcg@10']:.4f} | {metrics['ndcg@5']:.4f} | {metrics['mrr']:.4f} | {metrics['recall@100']:.4f} |"
        for mode, metrics in results.items() if mode in {"bm25", "semantic", "hybrid", "ltr"}
    )
    ci = results["ltr_vs_hybrid_ndcg@10"]
    verdict = "PASS" if results["passed"] else "NOT READY"
    (output_dir / "benchmark.md").write_text(
        "# LaMDaSearch Benchmark Results\n\n"
        f"Held-out queries: {results['queries']}\n\n"
        "| Mode | NDCG@10 | NDCG@5 | MRR | Recall@100 |\n|---|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "## LTR vs Hybrid RRF\n\n"
        f"Paired bootstrap NDCG@10 lift (95% CI): {ci['mean']:+.4f} "
        f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}]\n\n"
        f"**Verdict: {verdict}**\n"
    )


if __name__ == "__main__":
    from shared.es_client import get_client

    results = run_evaluation(get_client())
    write_results(results)
    print("PASS" if results["passed"] else "NOT READY")
