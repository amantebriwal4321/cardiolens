"""Leakage, metric integrity, and live SHAP checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cardiolens_core.features import ENCODED_FEATURES, LEAKAGE_COLUMNS, assert_no_leakage, normalize_record

METRIC_PATH = ROOT / "ml" / "metrics" / "cv_metrics.json"
OOF_PATH = ROOT / "ml" / "metrics" / "oof_predictions.csv"
SCHEMA_PATH = ROOT / "ml" / "models" / "schema.json"


def test_encoder_excludes_targets():
    assert_no_leakage(ENCODED_FEATURES)
    lowered = {name.lower() for name in ENCODED_FEATURES}
    for banned in LEAKAGE_COLUMNS:
        assert banned.lower() not in lowered
    with pytest.raises(ValueError, match="labels"):
        normalize_record({"LAD": "Stenotic"})


def test_oof_file_matches_reported_metrics():
    metrics = json.loads(METRIC_PATH.read_text(encoding="utf-8"))
    frame = pd.read_csv(OOF_PATH)
    selected = metrics["selected_model"]
    chosen = frame[frame["model"] == selected]
    for target, rows in chosen.groupby("target"):
        y_true = rows["y_true"].to_numpy()
        y_prob = rows["y_prob"].to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        reported = metrics["selected"]["targets"][target]["oof_pooled"]
        assert reported["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_prob), abs=1e-12)
        assert reported["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred), abs=1e-12)
        assert reported["precision"] == pytest.approx(precision_score(y_true, y_pred, zero_division=0), abs=1e-12)
        assert reported["recall"] == pytest.approx(recall_score(y_true, y_pred, zero_division=0), abs=1e-12)
        assert reported["f1"] == pytest.approx(f1_score(y_true, y_pred, zero_division=0), abs=1e-12)


def test_api_predicts_sample_and_rejects_leakage():
    from backend.app.main import app

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    sample = schema["samples"][0]
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        leaked = client.post("/predict", json={"features": {**sample["features"], "Cath": "CAD"}})
        assert leaked.status_code == 400
        response = client.post("/predict", json={"features": sample["features"]})
        assert response.status_code == 200
        payload = response.json()
        assert set(payload["targets"]) == {"cad", "lad", "lcx", "rca"}
        for target in payload["targets"].values():
            assert 0.0 <= target["probability"] <= 1.0
            assert len(target["contributions"]) == len(ENCODED_FEATURES)
            assert_no_leakage([item["feature"] for item in target["contributions"]])
            if target["space"] == "probability":
                rebuilt = target["base_value"] + target["shap_sum"]
                assert rebuilt == pytest.approx(target["probability"], abs=0.03)
