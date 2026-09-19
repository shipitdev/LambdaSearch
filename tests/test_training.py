import os
import numpy as np
import pytest
import xgboost as xgb
from train import get_groups
from shared.model_loader import assert_model_compatible, save_model, load_model


MINI_LIBSVM = """2 qid:1 0:2.0 1:299.99 2:1.43 3:4.2
1 qid:1 0:2.0 1:199.99 2:0.95 3:1.1
1 qid:1 0:2.0 1:99.99 2:0.48 3:2.5
2 qid:2 0:1.0 1:79.99 2:0.80 3:5.0
1 qid:2 0:1.0 1:59.99 2:0.60 3:3.8
1 qid:2 0:1.0 1:49.99 2:0.50 3:2.1"""


@pytest.fixture
def mini_libsvm_file(tmp_path):
    p = tmp_path / "mini.libsvm"
    p.write_text(MINI_LIBSVM)
    return str(p)


def test_get_groups(mini_libsvm_file):
    groups = get_groups(mini_libsvm_file)
    assert groups == [3, 3]
    assert sum(groups) == 6


def test_get_groups_real_data():
    if not os.path.exists("data/train.libsvm"):
        pytest.skip("train.libsvm not generated yet")
    groups = get_groups("data/train.libsvm")
    assert len(groups) > 0
    assert all(g > 0 for g in groups)


def test_model_save_load(tmp_path, mini_libsvm_file):
    dmat = xgb.DMatrix(f"{mini_libsvm_file}?format=libsvm")
    dmat.set_group([3, 3])
    params = {"objective": "rank:ndcg", "max_depth": 3, "seed": 42}
    booster = xgb.train(params, dmat, num_boost_round=5, verbose_eval=False)
    original_preds = booster.predict(dmat)

    model_path = tmp_path / "model.json"
    meta_path = tmp_path / "model_meta.json"
    save_model(booster, {"test": True}, model_path=model_path, meta_path=meta_path)
    loaded = load_model(model_path=model_path)
    loaded_preds = loaded.predict(dmat)

    np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)


def test_load_model_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(model_path=tmp_path / "missing.json")


def test_model_metadata_rejects_an_unknown_feature_schema():
    with pytest.raises(ValueError, match="feature schema"):
        assert_model_compatible({"feature_schema": "old"}, "ranking-v1")


def test_predictions_no_nans():
    if not os.path.exists("data/holdout.libsvm"):
        pytest.skip("holdout.libsvm not generated yet")
    if not os.path.exists("models/model.json"):
        pytest.skip("model not trained yet — run train.py first")
    booster = load_model()
    dholdout = xgb.DMatrix("data/holdout.libsvm?format=libsvm")
    preds = booster.predict(dholdout)
    assert not np.any(np.isnan(preds)), "Predictions contain NaN"
    assert not np.any(np.isinf(preds)), "Predictions contain Inf"
    assert len(preds) > 0
