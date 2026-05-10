import json
import os
import xgboost as xgb


def save_model(booster: xgb.Booster, meta: dict) -> None:
    os.makedirs("models", exist_ok=True)
    booster.save_model("models/model.json")
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Model saved to models/model.json")


def load_model() -> xgb.Booster:
    if not os.path.exists("models/model.json"):
        raise FileNotFoundError(
            "models/model.json not found — run train.py first."
        )
    booster = xgb.Booster()
    booster.load_model("models/model.json")
    return booster