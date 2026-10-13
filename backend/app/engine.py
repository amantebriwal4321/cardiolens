"""Load the saved forests and explain one patient at a time."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cardiolens_core.explain import ShapRunner
from cardiolens_core.features import (
    ENCODED_FEATURES,
    ENCODED_META,
    GROUP_COPY,
    GROUP_ORDER,
    TARGET_SPECS,
    assert_no_leakage,
    display_encoded,
    encode_records,
    normalize_record,
)

MODEL_DIR = ROOT / "ml" / "models"
METRIC_PATH = ROOT / "ml" / "metrics" / "cv_metrics.json"
TARGET_IDS = tuple(TARGET_SPECS)


class ModelBundle:
    def __init__(self) -> None:
        self.schema = json.loads((MODEL_DIR / "schema.json").read_text(encoding="utf-8"))
        self.metrics = json.loads(METRIC_PATH.read_text(encoding="utf-8"))
        self.global_shap = json.loads((MODEL_DIR / "global_shap.json").read_text(encoding="utf-8"))
        background = pd.read_csv(MODEL_DIR / "encoded_matrix.csv")
        if list(background.columns) != ENCODED_FEATURES:
            raise RuntimeError("Encoded background columns drifted from the feature schema.")
        assert_no_leakage(list(background.columns))
        self.runners: dict[str, ShapRunner] = {}
        self.estimators: dict[str, Any] = {}
        self.spaces: dict[str, str] = {}
        for target_id in TARGET_IDS:
            payload = joblib.load(MODEL_DIR / f"{target_id}.joblib")
            if payload["encoded_features"] != ENCODED_FEATURES:
                raise RuntimeError(f"{target_id} model features do not match the encoder.")
            assert_no_leakage(payload["encoded_features"])
            estimator = payload["estimator"]
            runner = ShapRunner.restore(
                estimator=estimator,
                feature_names=ENCODED_FEATURES,
                strategy=payload["shap_strategy"],
                space=payload["shap_space"],
                background=background,
                linear_state=payload.get("linear_state"),
            )
            self.estimators[target_id] = estimator
            self.runners[target_id] = runner
            self.spaces[target_id] = payload["shap_space"]

    def predict(self, raw_features: dict[str, Any]) -> dict[str, Any]:
        record = normalize_record(raw_features, compute_bmi="BMI" not in raw_features)
        frame = encode_records([record])
        targets: dict[str, Any] = {}
        for target_id, spec in TARGET_SPECS.items():
            estimator = self.estimators[target_id]
            probability = float(estimator.predict_proba(frame)[0, 1])
            bundle = self.runners[target_id].explain(frame)
            if bundle.space != self.spaces[target_id]:
                raise RuntimeError(f"SHAP space changed for {target_id}.")
            contributions = []
            for feature, shap_value in zip(ENCODED_FEATURES, bundle.values[0], strict=True):
                meta = ENCODED_META[feature]
                contributions.append(
                    {
                        "feature": feature,
                        "label": meta["label"],
                        "group": meta["group"],
                        "display": display_encoded(feature, record),
                        "shap": float(shap_value),
                    }
                )
            contributions.sort(key=lambda item: abs(item["shap"]), reverse=True)
            predicted_positive = probability >= 0.5
            targets[target_id] = {
                "title": spec["title"],
                "short": spec["short"],
                "probability": probability,
                "predicted_label": spec["positive"] if predicted_positive else spec["negative"],
                "positive_label": spec["positive"],
                "negative_label": spec["negative"],
                "threshold": 0.5,
                "base_value": float(bundle.base[0]),
                "shap_sum": float(bundle.values[0].sum()),
                "space": bundle.space,
                "reconstruction_error": float(bundle.max_abs_error),
                "contributions": contributions,
            }
        groups = []
        for group in GROUP_ORDER:
            totals = {}
            for target_id in TARGET_IDS:
                totals[target_id] = float(
                    sum(
                        item["shap"]
                        for item in targets[target_id]["contributions"]
                        if item["group"] == group
                    )
                )
            groups.append(
                {
                    "id": group,
                    "description": GROUP_COPY[group],
                    "shap": totals,
                }
            )
        return {
            "features": record,
            "targets": targets,
            "groups": groups,
            "disclaimer": self.schema["disclaimer"],
        }

    def bootstrap(self) -> dict[str, Any]:
        selected = self.metrics["selected"]["targets"]
        compact = {}
        for target_id, spec in TARGET_SPECS.items():
            pooled = selected[target_id]["oof_pooled"]
            means = selected[target_id]["cv_mean"]
            compact[target_id] = {
                "title": spec["title"],
                "oof_pooled": pooled,
                "cv_mean": {key: means[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc")},
                "positive_rate": self.metrics["class_balance"][target_id]["positive_rate"],
                "majority_class_accuracy": self.metrics["class_balance"][target_id]["majority_class_accuracy"],
                "top_shap": self.global_shap[target_id]["features"][:6],
                "shap_space": self.global_shap[target_id]["space"],
            }
        return {
            "schema": self.schema,
            "model": {
                "name": self.metrics["selected_model"],
                "validation": self.metrics["validation"],
                "selection_rule": self.metrics["selection_rule"],
                "ranking": self.metrics["ranking"],
                "targets": compact,
                "label_audit": self.metrics["label_audit"],
            },
        }


_BUNDLE: ModelBundle | None = None


def get_bundle() -> ModelBundle:
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = ModelBundle()
    return _BUNDLE
