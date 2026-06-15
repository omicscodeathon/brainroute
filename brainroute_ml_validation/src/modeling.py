from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import load_feature_view
from .preprocessing import CorrelationFilter, LowVarianceFilter, MedianImputer, MissingnessFilter, NonFiniteCleaner
from .utils import LOGGER, project_path, write_csv


def metric_dict(y_true, y_pred, y_score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else np.nan,
        "roc_auc": np.nan,
        "auprc": np.nan,
    }
    if y_score is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["auprc"] = average_precision_score(y_true, y_score)
    return out


def estimator_specs(seed: int, quick: bool = False) -> dict:
    grids = {
        "logistic_regression": (
            LogisticRegression(),
            {"clf__C": [1.0] if quick else [0.1, 1.0, 10.0], "clf__penalty": ["l2"], "clf__class_weight": ["balanced"], "clf__solver": ["lbfgs"], "clf__max_iter": [5000]},
            True,
        ),
        "knn": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5] if quick else [3, 5, 7, 11], "clf__weights": ["distance"] if quick else ["uniform", "distance"], "clf__metric": ["minkowski"]},
            True,
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {"clf__n_estimators": [200] if quick else [200, 300], "clf__max_depth": [None] if quick else [None, 20], "clf__min_samples_leaf": [1, 2] if quick else [1, 2, 5], "clf__max_features": ["sqrt"], "clf__class_weight": ["balanced"]},
            False,
        ),
        "extra_trees": (
            ExtraTreesClassifier(random_state=seed, n_jobs=-1),
            {"clf__n_estimators": [200] if quick else [200, 300], "clf__max_depth": [None] if quick else [None, 20], "clf__min_samples_leaf": [1, 2] if quick else [1, 2, 5], "clf__max_features": ["sqrt"], "clf__class_weight": ["balanced"]},
            False,
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        grids["lightgbm"] = (
            LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1),
            {"clf__n_estimators": [200] if quick else [200, 300], "clf__learning_rate": [0.05] if quick else [0.05, 0.1], "clf__num_leaves": [31] if quick else [31, 50], "clf__class_weight": ["balanced"]},
            False,
        )
    except Exception as exc:
        LOGGER.info("LightGBM unavailable; skipping. Reason: %s", exc)
    try:
        from xgboost import XGBClassifier

        grids["xgboost"] = (
            XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist"),
            {"clf__n_estimators": [200] if quick else [200, 300], "clf__max_depth": [3] if quick else [3, 5], "clf__learning_rate": [0.05] if quick else [0.05, 0.1], "clf__subsample": [0.8] if quick else [0.8, 1.0], "clf__colsample_bytree": [0.8] if quick else [0.8, 1.0], "clf__eval_metric": ["logloss"]},
            False,
        )
    except Exception as exc:
        LOGGER.info("XGBoost unavailable; skipping. Reason: %s", exc)
    return grids


def make_pipeline(estimator, needs_scaling: bool, cfg: dict) -> Pipeline:
    fs = cfg.get("feature_selection", {})
    steps = [
        ("finite", NonFiniteCleaner()),
        ("missingness", MissingnessFilter(float(fs.get("descriptor_missingness_threshold", 0.15)))),
        ("variance", LowVarianceFilter(float(fs.get("variance_threshold", 0.0)))),
        ("imputer", MedianImputer()),
        ("correlation", CorrelationFilter(float(fs.get("correlation_threshold", 0.95)), fs.get("max_features_after_selection"))),
    ]
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def selected_features_from_pipeline(model) -> list[str]:
    names = None
    for step_name in ["missingness", "variance", "correlation"]:
        step = model.named_steps.get(step_name)
        if hasattr(step, "get_feature_names_out"):
            names = step.get_feature_names_out(names)
    return list(names) if names is not None else []


def fitted_clf_params(model) -> dict:
    if not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return {}
    wanted = {
        "C",
        "penalty",
        "class_weight",
        "solver",
        "max_iter",
        "n_neighbors",
        "weights",
        "metric",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "learning_rate",
        "num_leaves",
        "subsample",
        "colsample_bytree",
        "eval_metric",
        "scale_pos_weight",
    }
    params = {}
    for key, value in model.named_steps["clf"].get_params().items():
        if key in wanted:
            if isinstance(value, (str, int, float, bool)) or value is None:
                params[f"clf__{key}"] = value
            else:
                params[f"clf__{key}"] = str(value)
    return params


def read_split_pair(cfg: dict, split_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = project_path(cfg, "data/splits")
    return pd.read_csv(root / f"{split_prefix}_train.csv"), pd.read_csv(root / f"{split_prefix}_test.csv")


def split_data_for_feature_view(
    X: pd.DataFrame, index: pd.DataFrame, split_prefix: str, view: str, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_df, test_df = read_split_pair(cfg, split_prefix)
    idx = index.set_index("molecule_id")
    X_by_id = X.copy()
    X_by_id.index = index["molecule_id"].values
    available_ids = set(X_by_id.index)
    train_df = train_df[train_df["molecule_id"].isin(available_ids)].copy()
    test_df = test_df[test_df["molecule_id"].isin(available_ids)].copy()
    train_ids = train_df["molecule_id"].tolist()
    test_ids = test_df["molecule_id"].tolist()
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError(f"No train/test molecules with feature view {view} in {split_prefix}")
    X_train = X_by_id.loc[train_ids]
    X_test = X_by_id.loc[test_ids]
    y_train = idx.loc[train_ids]["label"].astype(int).to_numpy()
    y_test = idx.loc[test_ids]["label"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError(f"Filtered split lacks both classes for {view} in {split_prefix}")
    return train_df, test_df, X_train, X_test, y_train, y_test


def predictions_and_metrics(model, X_test, y_test, test_df: pd.DataFrame, split_prefix: str, view: str, model_name: str, best_params: dict | str) -> tuple[dict, pd.DataFrame]:
    y_pred = model.predict(X_test)
    y_score = predict_scores(model, X_test)
    metrics = metric_dict(y_test, y_pred, y_score)
    if isinstance(best_params, str):
        params_json = best_params
    else:
        params_json = json.dumps(best_params)
    metrics.update({"split": split_prefix, "feature_view": view, "model": model_name, "best_params": params_json})

    pred = test_df.copy()
    pred["y_true"] = y_test
    pred["y_pred"] = y_pred
    pred["y_score"] = y_score if y_score is not None else np.nan
    pred["feature_view"] = view
    pred["model"] = model_name
    pred["split"] = split_prefix
    return metrics, pred


def fit_evaluate_split(X, index, split_prefix: str, view: str, model_name: str, spec, cfg: dict) -> tuple[dict, pd.DataFrame, object]:
    _, test_df, X_train, X_test, y_train, y_test = split_data_for_feature_view(X, index, split_prefix, view, cfg)
    estimator, grid, needs_scaling = spec

    if model_name == "xgboost":
        neg = max((y_train == 0).sum(), 1)
        pos = max((y_train == 1).sum(), 1)
        grid = dict(grid)
        grid["clf__scale_pos_weight"] = [neg / pos]

    pipeline = make_pipeline(estimator, needs_scaling, cfg)
    cv_n = min(int(cfg.get("modeling", {}).get("cv_folds_for_tuning", 3)), np.bincount(y_train).min())
    if cv_n < 2:
        raise ValueError(f"Not enough class counts for tuning in {split_prefix}")
    cv = StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=int(cfg.get("random_seed", 42)))
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring=cfg.get("modeling", {}).get("scoring_metric", "balanced_accuracy"),
        cv=cv,
        n_jobs=int(cfg.get("modeling", {}).get("n_jobs", -1)),
        refit=True,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    metrics, pred = predictions_and_metrics(best, X_test, y_test, test_df, split_prefix, view, model_name, search.best_params_)
    return metrics, pred, best

def evaluate_saved_split(X, index, split_prefix: str, view: str, model_name: str, model_path: Path, cfg: dict) -> tuple[dict, pd.DataFrame, object]:
    _, test_df, _, X_test, _, y_test = split_data_for_feature_view(X, index, split_prefix, view, cfg)
    model = joblib.load(model_path)
    metrics, pred = predictions_and_metrics(model, X_test, y_test, test_df, split_prefix, view, model_name, fitted_clf_params(model))
    return metrics, pred, model


def available_split_prefixes(cfg: dict) -> list[str]:
    root = project_path(cfg, "data/splits")
    prefixes = []
    for test in root.glob("*_test.csv"):
        prefix = test.name.replace("_test.csv", "")
        if (root / f"{prefix}_train.csv").exists():
            prefixes.append(prefix)
    primary = sorted([p for p in prefixes if p.startswith("scaffold_cv_fold")])
    duplicate = sorted([p for p in prefixes if p.startswith("duplicate_aware_seed")])
    random = sorted([p for p in prefixes if p.startswith("random80")])
    scaffold_holdout = sorted([p for p in prefixes if p.startswith("scaffold_split")])
    return primary + duplicate + random + scaffold_holdout


def train_models(cfg: dict) -> pd.DataFrame:
    seed = int(cfg.get("random_seed", 42))
    quick = bool(cfg.get("quick_mode", False))
    model_cfg = cfg.get("modeling", {})
    requested_models = model_cfg.get("quick_mode_models" if quick else "models_to_train", [])
    specs = estimator_specs(seed, quick)
    views = model_cfg.get("feature_views_to_run", ["padel", "morgan", "padel_morgan"])
    split_prefixes = available_split_prefixes(cfg)
    all_metrics = []
    all_predictions = []
    model_root = project_path(cfg, "models")
    report_root = project_path(cfg, "reports")
    model_root.mkdir(parents=True, exist_ok=True)

    for view in views:
        X, index = load_feature_view(cfg, view)
        if X is None:
            LOGGER.info("Feature view unavailable; skipping %s", view)
            continue
        for model_name in requested_models:
            if model_name not in specs:
                LOGGER.info("Model unavailable; skipping %s", model_name)
                continue
            for split_prefix in split_prefixes:
                safe = f"{view}__{model_name}__{split_prefix}"
                model_path = model_root / f"{safe}.joblib"
                try:
                    if model_path.exists() and not cfg.get("overwrite", False):
                        metrics, pred, best = evaluate_saved_split(X, index, split_prefix, view, model_name, model_path, cfg)
                        LOGGER.info("Resumed existing %s", safe)
                    else:
                        metrics, pred, best = fit_evaluate_split(X, index, split_prefix, view, model_name, specs[model_name], cfg)
                        joblib.dump(best, model_path)
                        LOGGER.info("Trained %s", safe)
                except Exception as exc:
                    LOGGER.warning("Skipping %s/%s/%s: %s", view, model_name, split_prefix, exc)
                    continue
                all_metrics.append(metrics)
                all_predictions.append(pred)
                write_csv(
                    pd.DataFrame({"selected_feature": selected_features_from_pipeline(best)}),
                    report_root / f"selected_features__{safe}.csv",
                    cfg,
                )

    metrics_df = pd.DataFrame(all_metrics)
    write_csv(metrics_df, report_root / "model_performance_all_splits.csv", cfg)
    if all_predictions:
        write_csv(pd.concat(all_predictions, ignore_index=True), report_root / "model_predictions_all_splits.csv", cfg)
    return metrics_df
