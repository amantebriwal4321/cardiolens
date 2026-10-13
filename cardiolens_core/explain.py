"""SHAP runners that stay consistent between training and the API.

The runner checks that SHAP values reconstruct the model output. Training
refuses a model whose explanations fail that check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _as_2d(values: Any, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        if len(values) == 2:
            values = values[1]
        elif len(values) == 1:
            values = values[0]
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 3:
        if array.shape[-1] == 2 and array.shape[1] == n_features:
            array = array[:, :, 1]
        elif array.shape[1] == 2 and array.shape[-1] == n_features:
            array = array[:, 1, :]
        else:
            raise ValueError(f"Unexpected SHAP shape {array.shape}")
    if array.shape[1] != n_features:
        raise ValueError(f"SHAP width {array.shape[1]} != {n_features} features")
    return array


def _as_base(base: Any, n_rows: int) -> np.ndarray:
    array = np.asarray(base, dtype=float)
    if array.ndim == 0:
        return np.repeat(float(array), n_rows)
    if array.ndim == 2 and array.shape[-1] == 2:
        array = array[:, 1]
    array = array.reshape(-1)
    if array.size == 1:
        return np.repeat(float(array[0]), n_rows)
    if array.size != n_rows:
        raise ValueError(f"Unexpected base shape {np.asarray(base).shape}")
    return array


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def _positive_proba(estimator: Any, frame: pd.DataFrame) -> np.ndarray:
    proba = estimator.predict_proba(frame)
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if classes == [0, 1] or classes == [0.0, 1.0]:
        return np.asarray(proba[:, 1], dtype=float)
    if 1 in classes:
        return np.asarray(proba[:, classes.index(1)], dtype=float)
    raise RuntimeError(f"Estimator classes are not binary 0/1: {classes}")


@dataclass
class ShapBundle:
    strategy: str
    space: str
    values: np.ndarray
    base: np.ndarray
    max_abs_error: float

    def row(self, index: int = 0) -> dict[str, Any]:
        return {
            "values": self.values[index].astype(float),
            "base": float(self.base[index]),
            "space": self.space,
            "strategy": self.strategy,
        }


TREE_STRATEGIES = (
    {
        "strategy": "tree_probability",
        "model_output": "probability",
        "feature_perturbation": "interventional",
        "needs_background": True,
    },
    {
        "strategy": "tree_raw",
        "model_output": "raw",
        "feature_perturbation": "tree_path_dependent",
        "needs_background": False,
    },
)


class ShapRunner:
    def __init__(
        self,
        estimator: Any,
        feature_names: list[str],
        strategy: str,
        space: str,
        explainer: Any = None,
        linear_state: dict[str, Any] | None = None,
    ) -> None:
        self.estimator = estimator
        self.feature_names = list(feature_names)
        self.strategy = strategy
        self.space = space
        self.explainer = explainer
        self.linear_state = linear_state or {}

    @classmethod
    def build(
        cls,
        model_name: str,
        estimator: Any,
        background: pd.DataFrame,
        feature_names: list[str],
    ) -> "ShapRunner":
        probe = background.iloc[: min(24, len(background))].reset_index(drop=True)
        if model_name == "logreg":
            runner = cls._linear_runner(estimator, feature_names, background)
        else:
            runner = cls._first_tree_runner(estimator, feature_names, background, probe)
        bundle = runner.explain(probe)
        runner.space = bundle.space
        if bundle.max_abs_error > 0.03:
            raise RuntimeError(
                f"SHAP reconstruction error {bundle.max_abs_error:.4f} "
                f"for {model_name} via {runner.strategy} ({bundle.space})"
            )
        return runner

    @classmethod
    def restore(
        cls,
        estimator: Any,
        feature_names: list[str],
        strategy: str,
        space: str,
        background: pd.DataFrame | None = None,
        linear_state: dict[str, Any] | None = None,
    ) -> "ShapRunner":
        if strategy == "linear_logit":
            if not linear_state:
                raise RuntimeError("Logistic SHAP state is missing from the saved model.")
            return cls(estimator, feature_names, strategy, space, linear_state=linear_state)
        spec = next(item for item in TREE_STRATEGIES if item["strategy"] == strategy)
        import shap

        kwargs: dict[str, Any] = {
            "model_output": spec["model_output"],
            "feature_perturbation": spec["feature_perturbation"],
        }
        if spec["needs_background"]:
            if background is None:
                raise RuntimeError("SHAP background rows are required for this model.")
            kwargs["data"] = background.iloc[: min(80, len(background))]
        explainer = shap.TreeExplainer(estimator, feature_names=feature_names, **kwargs)
        return cls(estimator, feature_names, strategy, space, explainer=explainer)

    @classmethod
    def _linear_runner(cls, estimator: Any, feature_names: list[str], background: pd.DataFrame) -> "ShapRunner":
        scaler = estimator.named_steps["scaler"]
        clf = estimator.named_steps["clf"]
        scaled = scaler.transform(background)
        mean = scaled.mean(axis=0)
        coef = np.asarray(clf.coef_, dtype=float).ravel()
        if coef.shape[0] != len(feature_names):
            raise RuntimeError("Logistic coefficients do not match the feature list.")
        intercept = float(np.asarray(clf.intercept_).ravel()[0])
        state = {"mean": mean, "coef": coef, "intercept": intercept}
        return cls(estimator, feature_names, "linear_logit", "logit", linear_state=state)

    @classmethod
    def _first_tree_runner(
        cls,
        estimator: Any,
        feature_names: list[str],
        background: pd.DataFrame,
        probe: pd.DataFrame,
    ) -> "ShapRunner":
        import shap

        sample = background.iloc[: min(80, len(background))]
        attempts = []
        for spec in TREE_STRATEGIES:
            kwargs = {
                "model_output": spec["model_output"],
                "feature_perturbation": spec["feature_perturbation"],
            }
            if spec["needs_background"]:
                kwargs["data"] = sample
            attempts.append((spec["strategy"], kwargs))
        errors: list[str] = []
        for strategy, kwargs in attempts:
            try:
                explainer = shap.TreeExplainer(estimator, feature_names=feature_names, **kwargs)
                runner_prob = cls(estimator, feature_names, strategy, "probability", explainer=explainer)
                bundle = runner_prob.explain(probe)
                if bundle.space == "probability" and bundle.max_abs_error <= 0.03:
                    return runner_prob
                runner_logit = cls(estimator, feature_names, strategy, "logit", explainer=explainer)
                bundle = runner_logit.explain(probe)
                if bundle.space == "logit" and bundle.max_abs_error <= 0.03:
                    return runner_logit
                errors.append(f"{strategy}: error {bundle.max_abs_error:.4f} space {bundle.space}")
            except Exception as exc:  # noqa: BLE001 - try the next official SHAP setup
                errors.append(f"{strategy}: {exc}")
        raise RuntimeError("No SHAP strategy reconstructed the model output. " + " | ".join(errors))

    def explain(self, frame: pd.DataFrame) -> ShapBundle:
        proba = _positive_proba(self.estimator, frame)
        if self.strategy == "linear_logit":
            values, base = self._linear_values(frame)
            recon = base + values.sum(axis=1)
            error = float(np.max(np.abs(_sigmoid(recon) - proba)))
            return ShapBundle(self.strategy, "logit", values, base, error)

        import shap

        explained = self.explainer(frame)
        values = _as_2d(explained.values, len(self.feature_names))
        base = _as_base(explained.base_values, len(frame))
        recon = base + values.sum(axis=1)
        prob_error = float(np.max(np.abs(recon - proba)))
        logit_error = float(np.max(np.abs(_sigmoid(recon) - proba)))
        if self.space == "probability" and prob_error <= logit_error:
            return ShapBundle(self.strategy, "probability", values, base, prob_error)
        if logit_error <= 0.03 or logit_error < prob_error:
            return ShapBundle(self.strategy, "logit", values, base, logit_error)
        return ShapBundle(self.strategy, "probability", values, base, prob_error)

    def _linear_values(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        scaler = self.estimator.named_steps["scaler"]
        scaled = scaler.transform(frame)
        mean = self.linear_state["mean"]
        coef = self.linear_state["coef"]
        intercept = self.linear_state["intercept"]
        values = coef * (scaled - mean)
        base_value = intercept + float(np.dot(coef, mean))
        base = np.repeat(base_value, len(frame))
        return values, base
