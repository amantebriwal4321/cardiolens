"""Train leakage-safe CAD and vessel stenosis models.

Run from the repo root:

    python ml/train.py

Metrics are written from this run. Do not hand-edit ml/metrics.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cardiolens_core.explain import ShapRunner
from cardiolens_core.features import (
    DROPPED_CONSTANT_COLUMNS,
    ENCODED_FEATURES,
    ENCODED_META,
    GROUP_COPY,
    GROUP_ORDER,
    INPUTS,
    LEAKAGE_COLUMNS,
    TARGET_SPECS,
    assert_no_leakage,
    display_encoded,
    display_input,
    encode_records,
    load_raw_table,
    records_from_frame,
)

DATA_PATH = ROOT / "ml" / "data" / "z_alizadeh_extension.csv"
MODEL_DIR = ROOT / "ml" / "models"
METRIC_DIR = ROOT / "ml" / "metrics"
PLOT_DIR = ROOT / "ml" / "plots"
RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5
MODEL_NAMES = ("logreg", "random_forest", "hist_gbdt", "xgboost")

SAMPLE_SPECS = [
    {
        "id": "A",
        "pattern": "all_normal",
        "prefer_sex": "Female",
        "title": "No stenotic vessel",
        "mask": lambda frame: (
            (frame["Cath"] == "Normal")
            & (frame["LAD"] == "Normal")
            & (frame["LCX"] == "Normal")
            & (frame["RCA"] == "Normal")
        ),
    },
    {
        "id": "B",
        "pattern": "three_vessel",
        "prefer_sex": "Male",
        "title": "Three-vessel pattern",
        "mask": lambda frame: (
            (frame["Cath"] == "CAD")
            & (frame["LAD"] == "Stenotic")
            & (frame["LCX"] == "Stenotic")
            & (frame["RCA"] == "Stenotic")
        ),
    },
    {
        "id": "C",
        "pattern": "lad_only",
        "prefer_sex": "Female",
        "title": "LAD only",
        "mask": lambda frame: (
            (frame["Cath"] == "CAD")
            & (frame["LAD"] == "Stenotic")
            & (frame["LCX"] == "Normal")
            & (frame["RCA"] == "Normal")
        ),
    },
    {
        "id": "D",
        "pattern": "lcx_only",
        "prefer_sex": "Male",
        "title": "LCX only",
        "mask": lambda frame: (
            (frame["Cath"] == "CAD")
            & (frame["LAD"] == "Normal")
            & (frame["LCX"] == "Stenotic")
            & (frame["RCA"] == "Normal")
        ),
    },
    {
        "id": "E",
        "pattern": "rca_only",
        "prefer_sex": "Female",
        "title": "RCA only",
        "mask": lambda frame: (
            (frame["Cath"] == "CAD")
            & (frame["LAD"] == "Normal")
            & (frame["LCX"] == "Normal")
            & (frame["RCA"] == "Stenotic")
        ),
    },
    {
        "id": "F",
        "pattern": "lad_lcx",
        "prefer_sex": "Male",
        "title": "LAD and LCX",
        "mask": lambda frame: (
            (frame["Cath"] == "CAD")
            & (frame["LAD"] == "Stenotic")
            & (frame["LCX"] == "Stenotic")
            & (frame["RCA"] == "Normal")
        ),
    },
]


def make_model(name: str, y_train: np.ndarray):
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=4000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if name == "hist_gbdt":
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.08,
            max_iter=250,
            min_samples_leaf=15,
            l2_regularization=0.2,
            random_state=RANDOM_STATE,
            early_stopping=False,
            class_weight="balanced",
        )
    if name == "xgboost":
        positive = float(np.sum(y_train == 1))
        negative = float(np.sum(y_train == 0))
        scale = negative / positive if positive else 1.0
        return XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=4,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            scale_pos_weight=scale,
        )
    raise KeyError(name)


def metric_block(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def fold_summary(fold_rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = fold_rows[0].keys()
    summary = {}
    for key in keys:
        values = np.array([row[key] for row in fold_rows], dtype=float)
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return summary


def pick_index(frame: pd.DataFrame, mask: pd.Series, prefer_sex: str, used: set[int]) -> int:
    candidates = [int(idx) for idx in frame.index[mask] if int(idx) not in used]
    if not candidates:
        raise RuntimeError("Sample pattern matched no remaining rows.")
    for idx in candidates:
        if frame.loc[idx, "Sex"] == prefer_sex:
            return idx
    return candidates[0]


def vignette(row: pd.Series, title: str) -> str:
    sex = "woman" if row["Sex"] == "Female" else "man"
    parts = [
        f"{int(row['Age'])} year old {sex}, height {int(row['Length'])} cm, weight {int(row['Weight'])} kg."
    ]
    history = []
    if int(row["DM"]) == 1:
        history.append("diabetes")
    if int(row["HTN"]) == 1:
        history.append("hypertension")
    if int(row["Current Smoker"]) == 1:
        history.append("current smoking")
    if str(row["DLP"]) == "Y":
        history.append("dyslipidemia")
    if history:
        parts.append("History includes " + ", ".join(history) + ".")
    if int(row["Typical Chest Pain"]) == 1:
        parts.append("Typical chest pain is marked yes.")
    elif str(row["Atypical"]) == "Y":
        parts.append("Atypical chest pain is marked yes.")
    elif str(row["Nonanginal"]) == "Y":
        parts.append("Nonanginal pain is marked yes.")
    if int(row["Function Class"]) > 0:
        parts.append(f"Function class {int(row['Function Class'])}.")
    if str(row["Systolic Murmur"]) == "Y":
        parts.append("Systolic murmur is marked yes.")
    if str(row["Diastolic Murmur"]) == "Y":
        parts.append("Diastolic murmur is marked yes.")
    if str(row["VHD"]) != "N":
        parts.append(f"Valve disease is coded {row['VHD']}.")
    parts.append(f"Ejection fraction {int(row['EF-TTE'])}%.")
    if int(row["Region RWMA"]) > 0:
        parts.append(f"Region with RWMA is {int(row['Region RWMA'])}.")
    parts.append(f"Source angiography pattern: {title}.")
    return " ".join(parts)


def style_plots() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_roc(oof: dict[str, np.ndarray], y_map: dict[str, np.ndarray], model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for target_id, spec in TARGET_SPECS.items():
        fpr, tpr, _ = roc_curve(y_map[target_id], oof[target_id])
        auc = roc_auc_score(y_map[target_id], oof[target_id])
        ax.plot(fpr, tpr, lw=2, label=f"{spec['short']}  {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="#888888", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Out-of-fold ROC, {model_name}")
    ax.legend(frameon=False, title="Pooled AUC")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_oof.png", dpi=140)
    plt.close(fig)


def plot_confusion(oof: dict[str, np.ndarray], y_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    for ax, (target_id, spec) in zip(axes.ravel(), TARGET_SPECS.items()):
        matrix = confusion_matrix(y_map[target_id], (oof[target_id] >= THRESHOLD).astype(int))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1], [spec["negative"], spec["positive"]], rotation=20, ha="right")
        ax.set_yticks([0, 1], [spec["negative"], spec["positive"]])
        ax.set_title(spec["short"])
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, str(value), ha="center", va="center", color="#102018")
        ax.set_xlabel("Predicted at 0.50")
        ax.set_ylabel("Source label")
        ax.grid(False)
    fig.suptitle("Out-of-fold confusion matrices", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_oof.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(oof: dict[str, np.ndarray], y_map: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.2))
    for ax, (target_id, spec) in zip(axes.ravel(), TARGET_SPECS.items()):
        fraction, mean_pred = calibration_curve(
            y_map[target_id], oof[target_id], n_bins=8, strategy="quantile"
        )
        ax.plot([0, 1], [0, 1], ls="--", c="#888888", lw=1)
        ax.plot(mean_pred, fraction, marker="o", lw=2)
        ax.set_title(spec["short"])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("Out-of-fold calibration (quantile bins)", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "calibration_oof.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison: dict) -> None:
    targets = list(TARGET_SPECS)
    x = np.arange(len(targets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for index, name in enumerate(MODEL_NAMES):
        means = [comparison[name][target]["cv_mean"]["roc_auc"]["mean"] for target in targets]
        stds = [comparison[name][target]["cv_mean"]["roc_auc"]["std"] for target in targets]
        ax.bar(x + (index - 1.5) * width, means, width, yerr=stds, capsize=3, label=name)
    ax.set_xticks(x, [TARGET_SPECS[target]["short"] for target in targets])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("5-fold stratified ROC-AUC, mean ± sample std")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "model_comparison_auc.png", dpi=140)
    plt.close(fig)


def plot_selected_metrics(selected: dict) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    targets = list(TARGET_SPECS)
    x = np.arange(len(metric_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for index, target in enumerate(targets):
        pooled = selected["targets"][target]["oof_pooled"]
        values = [pooled[name] for name in metric_names]
        ax.bar(x + (index - 1.5) * width, values, width, label=TARGET_SPECS[target]["short"])
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Pooled out-of-fold metrics, {selected['model_name']}, threshold 0.50")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "selected_metrics.png", dpi=140)
    plt.close(fig)


def plot_shap(target_id: str, mean_abs: np.ndarray, space: str) -> None:
    order = np.argsort(mean_abs)[::-1][:12]
    labels = [ENCODED_META[ENCODED_FEATURES[i]]["label"] for i in order][::-1]
    values = mean_abs[order][::-1]
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.barh(labels, values, color="#0f6e6e")
    unit = "probability" if space == "probability" else "log-odds"
    ax.set_xlabel(f"Mean absolute SHAP ({unit})")
    ax.set_title(f"{TARGET_SPECS[target_id]['title']} global SHAP")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"shap_{target_id}.png", dpi=140)
    plt.close(fig)


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write_summary(payload: dict) -> None:
    lines = [
        "# Computed results",
        "",
        "Generated by `ml/train.py`. These numbers are copied from `cv_metrics.json`.",
        "",
        f"Selected model: `{payload['selected_model']}`",
        "",
        "Selection rule: highest mean stratified 5-fold ROC-AUC, averaged across the four targets. Ties break on mean F1, then model name.",
        "",
        "## Pooled out-of-fold metrics for the selected model",
        "",
        "Threshold for accuracy, precision, recall, and F1 is 0.50. Positive class is CAD or Stenotic.",
        "",
        "| Target | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    selected = payload["selected"]
    for target_id, spec in TARGET_SPECS.items():
        row = selected["targets"][target_id]["oof_pooled"]
        lines.append(
            "| "
            + " | ".join(
                [
                    spec["title"],
                    fmt(row["accuracy"]),
                    fmt(row["precision"]),
                    fmt(row["recall"]),
                    fmt(row["f1"]),
                    fmt(row["roc_auc"]),
                    fmt(row["brier"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Mean ± std across the 5 folds (selected model)",
            "",
            "| Target | Accuracy | Precision | Recall | F1 | ROC-AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for target_id, spec in TARGET_SPECS.items():
        summary = selected["targets"][target_id]["cv_mean"]
        cells = [spec["title"]]
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            cells.append(f"{summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Mean ROC-AUC by model", "", "| Model | CAD | LAD | LCX | RCA | Average |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name in MODEL_NAMES:
        cells = [name]
        aucs = []
        for target_id in TARGET_SPECS:
            auc = payload["comparison"][name][target_id]["cv_mean"]["roc_auc"]["mean"]
            aucs.append(auc)
            cells.append(f"{auc:.3f}")
        cells.append(f"{float(np.mean(aucs)):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (METRIC_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for directory in (MODEL_DIR, METRIC_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    style_plots()

    raw = load_raw_table(DATA_PATH)
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert_no_leakage(ENCODED_FEATURES)

    any_vessel = (
        (raw["LAD"] == "Stenotic")
        | (raw["LCX"] == "Stenotic")
        | (raw["RCA"] == "Stenotic")
    )
    cath_cad = raw["Cath"] == "CAD"
    mismatch_index = raw.index[(cath_cad != any_vessel)].tolist()

    records = records_from_frame(raw)
    encoded = encode_records(records)
    assert list(encoded.columns) == ENCODED_FEATURES
    assert_no_leakage(list(encoded.columns))

    y_map = {}
    for target_id, spec in TARGET_SPECS.items():
        y_map[target_id] = (raw[spec["column"]] == spec["positive"]).astype(int).to_numpy()

    used: set[int] = set()
    samples = []
    for spec in SAMPLE_SPECS:
        index = pick_index(raw, spec["mask"](raw), spec["prefer_sex"], used)
        used.add(index)
        row = raw.loc[index]
        record = records[index]
        samples.append(
            {
                "id": spec["id"],
                "source_row": int(index),
                "pattern": spec["pattern"],
                "title": spec["title"],
                "vignette": vignette(row, spec["title"]),
                "features": record,
                "dataset_labels": {
                    "cad": row["Cath"],
                    "lad": row["LAD"],
                    "lcx": row["LCX"],
                    "rca": row["RCA"],
                },
            }
        )

    comparison: dict = {name: {} for name in MODEL_NAMES}
    oof_store: dict[tuple[str, str], np.ndarray] = {}
    oof_rows = []
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for target_id in TARGET_SPECS:
        y = y_map[target_id]
        for model_name in MODEL_NAMES:
            print(f"CV {model_name} {target_id}", flush=True)
            oof = np.zeros(len(y), dtype=float)
            fold_rows = []
            for train_idx, test_idx in splitter.split(encoded, y):
                model = make_model(model_name, y[train_idx])
                model.fit(encoded.iloc[train_idx], y[train_idx])
                prob = model.predict_proba(encoded.iloc[test_idx])[:, 1]
                oof[test_idx] = prob
                fold_rows.append(metric_block(y[test_idx], prob))
            pooled = metric_block(y, oof)
            comparison[model_name][target_id] = {
                "cv_mean": fold_summary(fold_rows),
                "oof_pooled": pooled,
                "folds": fold_rows,
            }
            oof_store[(model_name, target_id)] = oof
            for index, (truth, prob) in enumerate(zip(y, oof)):
                oof_rows.append(
                    {
                        "model": model_name,
                        "target": target_id,
                        "row_index": index,
                        "y_true": int(truth),
                        "y_prob": float(prob),
                    }
                )

    ranking = []
    for name in MODEL_NAMES:
        aucs = [comparison[name][target]["cv_mean"]["roc_auc"]["mean"] for target in TARGET_SPECS]
        f1s = [comparison[name][target]["cv_mean"]["f1"]["mean"] for target in TARGET_SPECS]
        ranking.append((float(np.mean(aucs)), float(np.mean(f1s)), name))
    ranking.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    selected_name = ranking[0][2]
    print("Selected", selected_name, ranking, flush=True)

    selected_oof = {target: oof_store[(selected_name, target)] for target in TARGET_SPECS}
    plot_roc(selected_oof, y_map, selected_name)
    plot_confusion(selected_oof, y_map)
    plot_calibration(selected_oof, y_map)
    plot_model_comparison(comparison)

    global_shap = {}
    shap_sanity = {}
    runners = {}
    for target_id, spec in TARGET_SPECS.items():
        print(f"Fit and explain {selected_name} {target_id}", flush=True)
        y = y_map[target_id]
        model = make_model(selected_name, y)
        model.fit(encoded, y)
        runner = ShapRunner.build(selected_name, model, encoded, ENCODED_FEATURES)
        probe_idx = np.linspace(0, len(encoded) - 1, num=min(24, len(encoded)), dtype=int)
        probe = runner.explain(encoded.iloc[probe_idx])
        bundle = runner.explain(encoded)
        mean_abs = np.abs(bundle.values).mean(axis=0)
        plot_shap(target_id, mean_abs, bundle.space)
        order = np.argsort(mean_abs)[::-1]
        global_shap[target_id] = {
            "space": bundle.space,
            "strategy": bundle.strategy,
            "base_value_mean": float(np.mean(bundle.base)),
            "features": [
                {
                    "feature": ENCODED_FEATURES[int(i)],
                    "label": ENCODED_META[ENCODED_FEATURES[int(i)]]["label"],
                    "group": ENCODED_META[ENCODED_FEATURES[int(i)]]["group"],
                    "mean_abs_shap": float(mean_abs[int(i)]),
                }
                for i in order
            ],
        }
        shap_sanity[target_id] = {
            "strategy": probe.strategy,
            "space": probe.space,
            "max_abs_error": probe.max_abs_error,
            "n_checked": int(len(probe_idx)),
            "full_matrix_max_abs_error": bundle.max_abs_error,
        }
        runners[target_id] = (model, runner)
        linear_state = None
        if runner.linear_state:
            linear_state = {
                "mean": runner.linear_state["mean"],
                "coef": runner.linear_state["coef"],
                "intercept": runner.linear_state["intercept"],
            }
        joblib.dump(
            {
                "estimator": model,
                "model_name": selected_name,
                "encoded_features": ENCODED_FEATURES,
                "target_id": target_id,
                "positive_label": spec["positive"],
                "negative_label": spec["negative"],
                "threshold": THRESHOLD,
                "shap_strategy": runner.strategy,
                "shap_space": runner.space,
                "linear_state": linear_state,
            },
            MODEL_DIR / f"{target_id}.joblib",
        )

    encoded.to_csv(MODEL_DIR / "encoded_matrix.csv", index=False)

    ranges = {}
    for item in INPUTS:
        key = item["key"]
        if item["kind"] == "number":
            series = raw[key].astype(float)
            ranges[key] = {
                "train_min": float(series.min()),
                "train_max": float(series.max()),
                "train_median": float(series.median()),
            }
        elif item["kind"] == "binary":
            true_value = item["true_value"]
            ranges[key] = {"true_count": int((raw[key] == true_value).sum())}

    schema_inputs = []
    for item in INPUTS:
        copied = dict(item)
        copied.update(ranges.get(item["key"], {}))
        schema_inputs.append(copied)

    majority = {}
    for target_id, y in y_map.items():
        prevalence = float(y.mean())
        majority_label = 1 if prevalence >= 0.5 else 0
        majority[target_id] = {
            "positive_count": int(y.sum()),
            "negative_count": int(len(y) - y.sum()),
            "positive_rate": prevalence,
            "majority_class_accuracy": float(max(prevalence, 1 - prevalence)),
            "majority_class": int(majority_label),
        }

    selected_payload = {
        "model_name": selected_name,
        "targets": {
            target_id: {
                "oof_pooled": comparison[selected_name][target_id]["oof_pooled"],
                "cv_mean": comparison[selected_name][target_id]["cv_mean"],
                "folds": comparison[selected_name][target_id]["folds"],
            }
            for target_id in TARGET_SPECS
        },
    }
    metrics = {
        "dataset": {
            "name": "extention of Z-Alizadeh sani dataset",
            "uci_id": 411,
            "doi": "10.24432/C5461K",
            "rows": int(len(raw)),
            "sha256": digest,
            "source_csv": "ml/data/z_alizadeh_extension.csv",
        },
        "leakage_columns_excluded": list(LEAKAGE_COLUMNS),
        "dropped_constant_columns": list(DROPPED_CONSTANT_COLUMNS),
        "encoded_feature_count": len(ENCODED_FEATURES),
        "encoded_features": ENCODED_FEATURES,
        "validation": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,
            "positive_class": "CAD for Cath, Stenotic for LAD, LCX, and RCA",
            "final_fit": "Selected model refit on all 303 rows after cross-validation.",
        },
        "label_audit": {
            "cath_positive_equals_any_vessel_stenotic": bool((cath_cad == any_vessel).all()),
            "mismatch_count": int(len(mismatch_index)),
            "mismatch_rows": [
                {
                    "row_index": int(idx),
                    "Cath": raw.loc[idx, "Cath"],
                    "LAD": raw.loc[idx, "LAD"],
                    "LCX": raw.loc[idx, "LCX"],
                    "RCA": raw.loc[idx, "RCA"],
                }
                for idx in mismatch_index
            ],
            "note": "Models predict the source labels. The discordant row was not relabeled.",
        },
        "class_balance": majority,
        "ranking": [
            {"model": name, "mean_roc_auc": auc, "mean_f1": f1}
            for auc, f1, name in ranking
        ],
        "selected_model": selected_name,
        "selection_rule": "Highest mean ROC-AUC across the four targets. Tie break: higher mean F1, then model name.",
        "comparison": comparison,
        "selected": selected_payload,
        "shap_sanity": shap_sanity,
        "hyperparameters": {
            "logreg": "StandardScaler + LogisticRegression C=1 class_weight=balanced",
            "random_forest": "500 trees, max_depth=6, min_samples_leaf=4, class_weight=balanced_subsample",
            "hist_gbdt": "max_depth=3, learning_rate=0.08, max_iter=250, min_samples_leaf=15, l2=0.2, class_weight=balanced",
            "xgboost": "250 trees, max_depth=3, learning_rate=0.05, min_child_weight=4, reg_lambda=1.5, scale_pos_weight=n_neg/n_pos",
        },
    }
    (METRIC_DIR / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(oof_rows).to_csv(METRIC_DIR / "oof_predictions.csv", index=False)
    (MODEL_DIR / "global_shap.json").write_text(json.dumps(global_shap, indent=2), encoding="utf-8")
    write_summary(metrics)
    plot_selected_metrics(selected_payload)

    schema = {
        "dataset": metrics["dataset"],
        "leakage_columns_excluded": list(LEAKAGE_COLUMNS),
        "dropped_constant_columns": list(DROPPED_CONSTANT_COLUMNS),
        "threshold": THRESHOLD,
        "selected_model": selected_name,
        "shap_space": {target: shap_sanity[target]["space"] for target in TARGET_SPECS},
        "groups": [{"id": group, "description": GROUP_COPY[group]} for group in GROUP_ORDER],
        "inputs": schema_inputs,
        "encoded_features": [
            ENCODED_META[key] for key in ENCODED_FEATURES
        ],
        "targets": {
            target_id: {
                "title": spec["title"],
                "short": spec["short"],
                "question": spec["question"],
                "positive": spec["positive"],
                "negative": spec["negative"],
                "column": spec["column"],
            }
            for target_id, spec in TARGET_SPECS.items()
        },
        "samples": samples,
        "disclaimer": (
            "Decision support and teaching only. CardioLens is not a diagnosis, "
            "not a cath report, and not a substitute for a cardiologist or emergency care."
        ),
        "anatomy_note": (
            "The heart and coronaries are a labeled schematic of typical anatomy, "
            "not a reconstruction from this patient's imaging."
        ),
        "in_sample_note": (
            "Sample patients are real rows from the 303-patient cohort. The live model "
            "was refit on every row after cross-validation, so a sample prediction is an "
            "illustration, not a fair test. Use the cross-validation metrics for performance."
        ),
    }
    (MODEL_DIR / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # Spot-check that a training row's displayed SHAP features cover the matrix.
    sample_record = samples[0]["features"]
    for feature in ENCODED_FEATURES:
        display_encoded(feature, sample_record)
        if feature in sample_record or ENCODED_META[feature]["source"] in sample_record:
            continue
    for item in INPUTS:
        display_input(item["key"], sample_record[item["key"]])

    print("Wrote metrics, plots, and models.", flush=True)
    print((METRIC_DIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
