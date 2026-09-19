"""Train LambdaMART from the independent RRF-candidate dataset."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import xgboost as xgb

from shared.features import FEATURE_SCHEMA_VERSION
from shared.model_loader import save_model


def get_groups(path: str) -> list[int]:
    counts: dict[int, int] = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                qid = int(line.split()[1].split(":")[1])
                counts[qid] = counts.get(qid, 0) + 1
    return list(counts.values())


def _matrix(path: str) -> xgb.DMatrix:
    matrix = xgb.DMatrix(f"{path}?format=libsvm")
    matrix.set_group(get_groups(path))
    return matrix


def train(train_path: str = "data/ranking-v1/train.libsvm", dev_path: str = "data/ranking-v1/dev.libsvm") -> xgb.Booster:
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "seed": 42,
    }
    return xgb.train(
        params,
        _matrix(train_path),
        num_boost_round=100,
        evals=[(_matrix(dev_path), "dev")],
        verbose_eval=10,
    )


if __name__ == "__main__":
    booster = train()
    save_model(booster, {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "dataset": "ranking-v1",
        "seed": 42,
        "inference_id": os.getenv("ELASTIC_INFERENCE_ID"),
    })
