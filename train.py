import json
import numpy as np
import xgboost as xgb
from datetime import datetime
from shared.model_loader import save_model


def load_libsvm(path: str) -> xgb.DMatrix:
    return xgb.DMatrix(f"{path}?format=libsvm")


def get_groups(path: str) -> list[int]:
    qid_counts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid = int(line.split()[1].split(":")[1])
            qid_counts[qid] = qid_counts.get(qid, 0) + 1
    return list(qid_counts.values())


def train(train_path: str = "data/train.libsvm") -> xgb.Booster:
    print("Loading training data...")
    dtrain = xgb.DMatrix(f"{train_path}?format=libsvm")
    groups = get_groups(train_path)
    dtrain.set_group(groups)

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "seed": 42,
    }

    print("Training LambdaMART model...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train")],
        verbose_eval=10,
    )
    return booster


def evaluate_on_holdout(booster: xgb.Booster, holdout_path: str = "data/holdout.libsvm") -> float:
    dholdout = xgb.DMatrix(f"{holdout_path}?format=libsvm")
    groups = get_groups(holdout_path)
    dholdout.set_group(groups)
    scores = booster.predict(dholdout)
    print(f"Generated {len(scores)} predictions on holdout set.")
    return float(np.mean(scores))


if __name__ == "__main__":
    booster = train()
    mean_score = evaluate_on_holdout(booster)
    meta = {
        "trained_at": datetime.now().isoformat(),
        "params": {
            "objective": "rank:ndcg",
            "max_depth": 6,
            "learning_rate": 0.1,
            "num_boost_round": 100,
        },
        "mean_holdout_score": mean_score,
    }
    save_model(booster, meta)
    print(f"Done. Mean holdout prediction score: {mean_score:.4f}")