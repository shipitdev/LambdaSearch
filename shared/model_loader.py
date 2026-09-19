import json
from pathlib import Path
import xgboost as xgb

DEFAULT_MODEL_PATH = Path("models/model.json")
DEFAULT_META_PATH = Path("models/model_meta.json")

def save_model(
    booster: xgb.Booster,
    meta: dict,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    meta_path: str | Path = DEFAULT_META_PATH,
) -> None:
    model_path, meta_path = Path(model_path), Path(meta_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_path)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Model saved to {model_path}")


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> xgb.Booster:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run train.py first."
        )
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def load_model_metadata(meta_path: str | Path = DEFAULT_META_PATH) -> dict:
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found — run train.py first.")
    with meta_path.open() as f:
        return json.load(f)
