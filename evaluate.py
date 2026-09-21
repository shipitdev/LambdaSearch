"""Independent, reproducible evaluation for the ranking lab."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import xgboost as xgb

from catalog import load_catalog
from shared.benchmark_data import build_benchmark
from shared.features import compute_category_stats
from shared.model_loader import load_model, load_model_metadata
from shared.reranking import rerank_hits
from shared.search import build_search_request


def dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(position + 2) for position, rel in enumerate(relevances[:k]))


def ideal_dcg(relevances: list[int], k: int) -> float:
    return dcg(sorted(relevances, reverse=True), k)


def ndcg_at_k(relevances: list[int], k: int, all_relevances: list[int] | None = None) -> float:
    denominator = ideal_dcg(all_relevances if all_relevances is not None else relevances, k)
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


def strongest_baseline(mode_rows: dict[str, list[dict]]) -> str:
    return max(
        ("bm25", "semantic", "hybrid"),
        key=lambda mode: float(np.mean([row["ndcg@10"] for row in mode_rows[mode]])),
    )


def _search_many(es, cases: list[dict], mode: str, size: int = 100) -> list[list[dict]]:
    searches = []
    for case in cases:
        searches.extend(({}, build_search_request(case["query"], mode, size)))
    response = es.options(request_timeout=60).msearch(index="catalog", searches=searches)
    return [item["hits"]["hits"] for item in response["responses"]]


def _metrics_for_case(case: dict, hits: list[dict]) -> dict:
    labels = [case["judgments"].get(hit["_source"].get("item_id"), 0) for hit in hits]
    all_labels = list(case["judgments"].values())
    return {
        "ndcg@10": ndcg_at_k(labels, 10, all_labels),
        "ndcg@5": ndcg_at_k(labels, 5, all_labels),
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
    metadata = load_model_metadata()
    modes = {"bm25": [], "semantic": [], "hybrid": [], "ltr": []}

    retrieved = {mode: _search_many(es, cases, mode) for mode in ("bm25", "semantic", "hybrid")}
    for position, case in enumerate(cases):
        for mode in ("bm25", "semantic", "hybrid"):
            hits = retrieved[mode][position]
            modes[mode].append({"query": case["query"], "cohort": case["cohort"], **_metrics_for_case(case, hits)})
        ranked_hits = rerank_hits(
            case["query"], retrieved["hybrid"][position], category_stats,
            lambda rows: model.predict(xgb.DMatrix(rows)),
        )
        modes["ltr"].append({"query": case["query"], "cohort": case["cohort"], **_metrics_for_case(case, ranked_hits)})

    results = {mode: _summarize(rows) for mode, rows in modes.items()}
    results["cohorts"] = {
        cohort: {mode: _summarize([row for row in rows if row["cohort"] == cohort]) for mode, rows in modes.items()}
        for cohort in ("lexical", "attribute", "semantic")
    }
    baseline = strongest_baseline(modes)
    results["ltr_vs_strongest_baseline_ndcg@10"] = {
        "baseline": baseline,
        **paired_bootstrap_ci(
        [row["ndcg@10"] for row in modes["ltr"]],
        [row["ndcg@10"] for row in modes[baseline]],
        ),
    }
    results["passed"] = results["ltr_vs_strongest_baseline_ndcg@10"]["lower"] > 0
    results["queries"] = len(cases)
    results["provenance"] = {
        "benchmark": "ranking-v1",
        "seed": 42,
        "inference_id": os.getenv("ELASTIC_INFERENCE_ID"),
        "model": metadata,
    }
    return results


def write_results(results: dict, output_dir: str | Path = "results") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = "\n".join(
        f"| {mode} | {metrics['ndcg@10']:.4f} | {metrics['ndcg@5']:.4f} | {metrics['mrr']:.4f} | {metrics['recall@100']:.4f} |"
        for mode, metrics in results.items() if mode in {"bm25", "semantic", "hybrid", "ltr"}
    )
    ci = results["ltr_vs_strongest_baseline_ndcg@10"]
    verdict = "PASS" if results["passed"] else "NOT READY"
    (output_dir / "benchmark.md").write_text(
        "# LaMDaSearch Benchmark Results\n\n"
        f"Held-out queries: {results['queries']}\n\n"
        f"Benchmark: {results['provenance']['benchmark']} (seed {results['provenance']['seed']})\n\n"
        "| Mode | NDCG@10 | NDCG@5 | MRR | Recall@100 |\n|---|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        f"## LTR vs Strongest First-Stage Baseline ({ci['baseline']})\n\n"
        f"Paired bootstrap NDCG@10 lift (95% CI): {ci['mean']:+.4f} "
        f"[{ci['lower']:+.4f}, {ci['upper']:+.4f}]\n\n"
        f"**Verdict: {verdict}**\n"
    )


if __name__ == "__main__":
    from shared.es_client import get_client

    results = run_evaluation(get_client())
    write_results(results)
    print("PASS" if results["passed"] else "NOT READY")
