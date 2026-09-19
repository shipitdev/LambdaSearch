"""Build LambdaMART rows from independent judgments and RRF candidates."""

from __future__ import annotations

import json
from pathlib import Path

from catalog import load_catalog
from shared.benchmark_data import build_benchmark
from shared.features import compute_category_stats, feature_vector
from shared.search import build_search_request


def to_libsvm_row(label: int, qid: int, features: list[float]) -> str:
    return f"{label} qid:{qid} " + " ".join(f"{index}:{value:.6f}" for index, value in enumerate(features))


def build_dataset(cases: list[dict], catalog: list[dict], retrieve) -> list[str]:
    """Use only hybrid candidates; labels are supplied by the benchmark."""
    category_stats = compute_category_stats(catalog)
    rows = []
    for qid, case in enumerate(cases, start=1):
        for hit in retrieve(case["query"]):
            item = hit["_source"]
            label = case["judgments"].get(item.get("item_id"), 0)
            rows.append(to_libsvm_row(label, qid, feature_vector(case["query"], item, hit.get("_score", 0.0), category_stats)))
    return rows


def write_dataset(rows: list[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")


def prepare_datasets(es, output_dir: str | Path = "data/ranking-v1") -> Path:
    """Retrieve candidates once and write query-disjoint train/dev/test files."""
    catalog = load_catalog()
    benchmark = build_benchmark(catalog)
    output_dir = Path(output_dir)

    def retrieve(query: str) -> list[dict]:
        response = es.options(request_timeout=10).search(
            index="catalog", body=build_search_request(query, "hybrid", 100)
        )
        return response["hits"]["hits"]

    for split, cases in benchmark.items():
        write_dataset(build_dataset(cases, catalog, retrieve), output_dir / f"{split}.libsvm")
    (output_dir / "benchmark.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    return output_dir


if __name__ == "__main__":
    from shared.es_client import get_client

    directory = prepare_datasets(get_client())
    print(f"Wrote ranking datasets to {directory}")
