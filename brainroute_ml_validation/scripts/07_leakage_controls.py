#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.features import load_feature_view
from brainroute_ml_validation.src.modeling import make_pipeline
from brainroute_ml_validation.src.preprocessing import finite_dataframe
from brainroute_ml_validation.src.utils import LOGGER, ensure_dirs, load_config, project_path, read_table, script_arg_parser, set_global_seed, write_csv


def main() -> None:
    args = script_arg_parser("Run source, permuted-label, and feature leakage controls.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    std = read_table(project_path(cfg, "data/processed/standardized_molecules.csv"))

    if "source_dataset" in std:
        dist = std.groupby(["source_dataset", "label"]).size().reset_index(name="count")
        write_csv(dist, project_path(cfg, "reports/source_label_distribution.csv"), cfg)

    X, idx = load_feature_view(cfg, "padel_morgan")
    if X is None:
        X, idx = load_feature_view(cfg, "morgan")
    if X is None:
        LOGGER.info("No feature matrix available for leakage controls; skipping model-based controls.")
        return
    X_clean = finite_dataframe(X).fillna(0)

    source_rows = []
    if "source_dataset" in idx and idx["source_dataset"].nunique(dropna=True) >= 2:
        y_source = LabelEncoder().fit_transform(idx["source_dataset"].astype(str))
        counts = np.bincount(y_source)
        if len(counts) > 1 and counts.min() >= 5:
            cv = StratifiedKFold(n_splits=min(3, counts.min()), shuffle=True, random_state=int(cfg.get("random_seed", 42)))
            clf = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=2000))])
            pred = cross_val_predict(clf, X_clean, y_source, cv=cv)
            source_rows.append({"control": "source_prediction", "accuracy": accuracy_score(y_source, pred), "balanced_accuracy": balanced_accuracy_score(y_source, pred)})
    write_csv(pd.DataFrame(source_rows), project_path(cfg, "reports/source_prediction_control.csv"), cfg)

    y = idx["label"].astype(int).to_numpy() #permuted label control
    rows = []
    for seed in [11, 22, 33]:
        rng = np.random.default_rng(seed)
        y_perm = rng.permutation(y)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        model = make_pipeline(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1, class_weight="balanced"), False, cfg)
        scores = cross_val_predict(model, X, y_perm, cv=cv, method="predict_proba")[:, 1]
        pred = (scores >= 0.5).astype(int)
        rows.append(
            {
                "seed": seed,
                "balanced_accuracy": balanced_accuracy_score(y_perm, pred),
                "roc_auc": roc_auc_score(y_perm, scores) if len(set(y_perm)) > 1 else np.nan,
            }
        )
    write_csv(pd.DataFrame(rows), project_path(cfg, "reports/permuted_label_control.csv"), cfg)


if __name__ == "__main__":
    main()
