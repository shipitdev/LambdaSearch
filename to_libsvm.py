import json
import os
import random
from catalog import load_catalog
from shared.features import compute_category_stats, feature_vector


def load_click_log(path: str = "data/click_log.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def compute_labels(click_log: list[dict]) -> dict:
    labels = {}
    for event in click_log:
        key = (event["query"], event["item_id"])
        if event["clicked"]:
            labels[key] = 2
        elif key not in labels:
            labels[key] = 1
    return labels


def to_libsvm_row(label: int, qid: int, features: list[float]) -> str:
    feat_str = " ".join(f"{i}:{v:.6f}" for i, v in enumerate(features))
    return f"{label} qid:{qid} {feat_str}"


def build_dataset(click_log: list[dict], catalog: list[dict]) -> list[str]:
    category_stats = compute_category_stats(catalog)
    catalog_by_id = {item["item_id"]: item for item in catalog}
    labels = compute_labels(click_log)

    queries = list(dict.fromkeys(e["query"] for e in click_log))
    query_to_qid = {q: i + 1 for i, q in enumerate(queries)}

    rows = []
    for event in click_log:
        key = (event["query"], event["item_id"])
        label = labels.get(key, 1)
        qid = query_to_qid[event["query"]]
        item = catalog_by_id.get(event["item_id"])
        if item is None:
            continue
        features = feature_vector(
            event["query"], item, event["bm25_score"], category_stats
        )
        rows.append((qid, to_libsvm_row(label, qid, features)))

    rows.sort(key=lambda x: x[0])
    return [row for _, row in rows]


def split_and_write(
    rows: list[str],
    click_log: list[dict],
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    queries = list(dict.fromkeys(e["query"] for e in click_log))
    query_to_qid = {q: i + 1 for i, q in enumerate(queries)}
    qid_to_query = {v: k for k, v in query_to_qid.items()}

    unique_qids = sorted(set(query_to_qid.values()))
    random.seed(seed)
    random.shuffle(unique_qids)

    n_holdout = max(1, int(len(unique_qids) * holdout_ratio))
    holdout_qids = set(unique_qids[:n_holdout])
    train_qids = set(unique_qids[n_holdout:])

    train_rows = [r for r in rows if int(r.split()[1].split(":")[1]) in train_qids]
    holdout_rows = [r for r in rows if int(r.split()[1].split(":")[1]) in holdout_qids]

    os.makedirs("data", exist_ok=True)
    with open("data/train.libsvm", "w") as f:
        f.write("\n".join(train_rows))
    with open("data/holdout.libsvm", "w") as f:
        f.write("\n".join(holdout_rows))

    print(f"Train: {len(train_rows)} rows ({len(train_qids)} queries)")
    print(f"Holdout: {len(holdout_rows)} rows ({len(holdout_qids)} queries)")


if __name__ == "__main__":
    click_log = load_click_log()
    catalog = load_catalog()
    rows = build_dataset(click_log, catalog)
    split_and_write(rows, click_log)
    print("Written data/train.libsvm and data/holdout.libsvm")