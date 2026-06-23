#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.features import load_feature_view
from brainroute_ml_validation.src.modeling import metric_dict, predict_scores
from brainroute_ml_validation.src.utils import ensure_dirs, load_config, project_path, script_arg_parser, set_global_seed, write_csv


THRESHOLD = 0.5
DEFAULT_EXTERNAL_CACHE_SUBDIR = "results/external_results_new_full_padel_parallel2"
DEFAULT_RESULTS_SUBDIR = "results/external_results_full_refit"

SELECTED_MODELS = [
    {
        "model_configuration": "PaDEL + Morgan LightGBM full-data refit",
        "feature_view": "padel_morgan",
        "model": "lightgbm",
        "source_artifact": "padel_morgan__lightgbm__duplicate_aware_seed5.joblib",
        "refit_artifact": "full_refit__padel_morgan__lightgbm.joblib",
    },
    {
        "model_configuration": "PaDEL + Morgan Extra Trees full-data refit",
        "feature_view": "padel_morgan",
        "model": "extra_trees",
        "source_artifact": "padel_morgan__extra_trees__duplicate_aware_seed5.joblib",
        "refit_artifact": "full_refit__padel_morgan__extra_trees.joblib",
    },
    {
        "model_configuration": "PaDEL + Morgan + ChemBERTa XGBoost full-data refit",
        "feature_view": "padel_morgan_embeddings",
        "model": "xgboost",
        "source_artifact": "padel_morgan_embeddings__xgboost__scaffold_cv_fold1.joblib",
        "refit_artifact": "full_refit__padel_morgan_embeddings__xgboost.joblib",
    },
]


def load_external_validation_module():
    path = Path(__file__).resolve().parent / "09_external_validation.py"
    spec = importlib.util.spec_from_file_location("brainroute_external_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load external validation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_project_relative(cfg: dict, value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return project_path(cfg, str(value))


def refit_selected_models(cfg: dict, overwrite: bool = False) -> pd.DataFrame:
    model_dir = project_path(cfg, "models")
    out_dir = model_dir / "full_refit"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for spec in SELECTED_MODELS:
        source_path = model_dir / spec["source_artifact"]
        output_path = out_dir / spec["refit_artifact"]
        if not source_path.exists():
            raise FileNotFoundError(f"Missing selected source artifact: {source_path}")

        X, index = load_feature_view(cfg, spec["feature_view"])
        if X is None:
            raise FileNotFoundError(f"Feature view unavailable: {spec['feature_view']}")
        y = index["label"].astype(int).to_numpy()

        if output_path.exists() and not overwrite:
            model = joblib.load(output_path)
            action = "resumed_existing_full_refit"
        else:
            source_model = joblib.load(source_path)
            model = clone(source_model)
            if spec["model"] == "xgboost":
                neg = max(int((y == 0).sum()), 1)
                pos = max(int((y == 1).sum()), 1)
                if "clf__scale_pos_weight" in model.get_params():
                    model.set_params(clf__scale_pos_weight=neg / pos)
            model.fit(X, y)
            joblib.dump(model, output_path)
            action = "trained_full_refit"

        rows.append(
            {
                "model_configuration": spec["model_configuration"],
                "feature_view": spec["feature_view"],
                "model": spec["model"],
                "source_artifact": source_path.name,
                "refit_artifact": output_path.name,
                "refit_artifact_path": str(output_path),
                "training_rows": int(len(X)),
                "training_bbb_positive": int((y == 1).sum()),
                "training_bbb_negative": int((y == 0).sum()),
                "feature_count": int(X.shape[1]),
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def load_cached_external_features(cfg: dict, external_cache_dir: Path, ext_helpers) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    cleaned_path = external_cache_dir / "cleaned_non_overlapping_external_validation_dataframe.csv"
    padel_path = external_cache_dir / "external_padel_features_raw.csv"
    failed_path = external_cache_dir / "external_padel_failed_positions.json"
    curation_path = external_cache_dir / "external_dataset_curation_summary.csv"

    for path in [cleaned_path, padel_path, failed_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing cached external validation file: {path}")

    cleaned = pd.read_csv(cleaned_path)
    padel_raw = pd.read_csv(padel_path)
    with failed_path.open("r", encoding="utf-8") as handle:
        failed_positions = set(json.load(handle))

    keep_positions = [idx for idx in range(len(padel_raw)) if idx not in failed_positions]
    padel = padel_raw.iloc[keep_positions].reset_index(drop=True)
    if len(padel) != len(cleaned):
        raise ValueError(
            "Cached PaDEL rows do not match cleaned external dataframe after removing failed positions: "
            f"padel={len(padel)}, cleaned={len(cleaned)}"
        )

    smiles = cleaned["canonical_smiles"].tolist()
    morgan = ext_helpers.calculate_morgan_features(smiles, cfg)
    views = {
        "padel_morgan": pd.concat([padel.reset_index(drop=True), morgan.reset_index(drop=True)], axis=1)
    }
    if any(spec["feature_view"] == "padel_morgan_embeddings" for spec in SELECTED_MODELS):
        embeddings = ext_helpers.calculate_embedding_features(smiles, cfg)
        views["padel_morgan_embeddings"] = pd.concat(
            [padel.reset_index(drop=True), morgan.reset_index(drop=True), embeddings.reset_index(drop=True)],
            axis=1,
        )

    curation = pd.read_csv(curation_path) if curation_path.exists() else pd.DataFrame()
    return cleaned, views, curation


def predict_with_refit_model(model_path: Path, X_view: pd.DataFrame, y_true: np.ndarray, ext_helpers):
    model = joblib.load(model_path)
    expected = ext_helpers.model_expected_features(model)
    X_aligned = X_view.reindex(columns=expected, fill_value=np.nan)
    y_score = predict_scores(model, X_aligned)
    if y_score is None:
        raise ValueError(f"Model does not expose probability scores: {model_path}")
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= THRESHOLD).astype(int)
    metrics = metric_dict(y_true, y_pred, y_score)
    return y_pred, y_score, metrics


def evaluate_refits_on_external(
    cfg: dict,
    refit_summary: pd.DataFrame,
    external_cache_dir: Path,
    output_dir: Path,
    ext_helpers,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    cleaned, feature_views, cached_curation = load_cached_external_features(cfg, external_cache_dir, ext_helpers)
    y_true = cleaned["label"].astype(int).to_numpy()

    metric_rows = []
    prediction_rows = []
    for _, model_row in refit_summary.iterrows():
        model_path = Path(model_row["refit_artifact_path"])
        X_view = feature_views[model_row["feature_view"]]
        y_pred, y_score, metrics = predict_with_refit_model(model_path, X_view, y_true, ext_helpers)
        metrics.update(
            {
                "model_configuration": model_row["model_configuration"],
                "feature_view": model_row["feature_view"],
                "model": model_row["model"],
                "fold_or_seed": "full_refit_all_internal_training_data",
                "model_artifact": model_path.name,
                "threshold": THRESHOLD,
            }
        )
        metric_rows.append(metrics)
        for pos, (pred, score) in enumerate(zip(y_pred, y_score)):
            prediction_rows.append(
                {
                    "external_position": pos,
                    "external_row": cleaned.iloc[pos]["external_row"],
                    "model_configuration": model_row["model_configuration"],
                    "fold_or_seed": "full_refit_all_internal_training_data",
                    "model_artifact": model_path.name,
                    "y_true": int(y_true[pos]),
                    "y_pred": int(pred),
                    "y_score": float(score),
                }
            )

    fold_metrics = pd.DataFrame(metric_rows)
    per_artifact_predictions = pd.DataFrame(prediction_rows)
    summary = ext_helpers.mean_std_summary(fold_metrics)
    probability_metrics, probability_predictions = ext_helpers.probability_averaged_results(per_artifact_predictions, cleaned)
    ensemble_metrics = ext_helpers.brainroute_ensemble_results(probability_predictions, y_true)

    curation = cached_curation.copy()
    if curation.empty:
        curation = pd.DataFrame(
            [
                {
                    "final_external_validation_size": int(len(cleaned)),
                    "final_bbb_positive_count": int((cleaned["label"] == 1).sum()),
                    "final_bbb_negative_count": int((cleaned["label"] == 0).sum()),
                }
            ]
        )
    curation["external_cache_source"] = str(external_cache_dir)
    curation["validation_note"] = "Exact training overlaps removed before evaluation; PaDEL-failed molecules skipped from cached curation."

    write_csv(refit_summary, output_dir / "refit_training_summary.csv", cfg)
    write_csv(curation, output_dir / "external_dataset_curation_summary.csv", cfg)
    write_csv(fold_metrics, output_dir / "external_validation_fold_level_metrics.csv", cfg)
    write_csv(summary, output_dir / "external_validation_summary_mean_std.csv", cfg)
    write_csv(probability_metrics, output_dir / "external_validation_probability_averaged_results.csv", cfg)
    write_csv(ensemble_metrics, output_dir / "external_validation_brainroute_ensemble_results.csv", cfg)
    write_csv(per_artifact_predictions, output_dir / "external_validation_per_artifact_predictions.csv", cfg)
    write_csv(probability_predictions, output_dir / "external_validation_cleaned_predictions.csv", cfg)

    duplicate_prediction_cols = [col for col in probability_predictions.columns if col in cleaned.columns and col != "external_row"]
    cleaned_with_outputs = cleaned.merge(
        probability_predictions.drop(columns=duplicate_prediction_cols),
        on="external_row",
        how="left",
    )
    write_csv(cleaned_with_outputs, output_dir / "cleaned_non_overlapping_external_validation_dataframe.csv", cfg)

    for filename in [
        "external_invalid_or_failed_standardization.csv",
        "external_training_overlaps_removed.csv",
        "external_padel_failed_molecules.csv",
        "external_padel_failed_positions.json",
    ]:
        source = external_cache_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)

    ext_helpers.save_figures(fold_metrics, probability_metrics, ensemble_metrics, probability_predictions, cleaned, output_dir)

    print("\nFull-data refit external validation")
    print("===================================")
    print(curation.to_string(index=False))
    print("\nModel metrics")
    print(
        fold_metrics[
            ["model_configuration", "accuracy", "balanced_accuracy", "roc_auc", "auprc", "mcc", "f1"]
        ].to_string(index=False)
    )
    print("\nBrainRoute ensemble")
    print(
        ensemble_metrics[["accuracy", "balanced_accuracy", "roc_auc", "auprc", "mcc", "f1"]].to_string(index=False)
    )


def main() -> None:
    parser = script_arg_parser("Refit the three BrainRoute prediction-tool models on all internal training data and evaluate externally.")
    parser.add_argument("--external-cache-subdir", default=DEFAULT_EXTERNAL_CACHE_SUBDIR)
    parser.add_argument("--results-subdir", default=DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--overwrite", action="store_true", help="Retrain full-refit artifacts even if they already exist.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)

    ext_helpers = load_external_validation_module()
    external_cache_dir = resolve_project_relative(cfg, args.external_cache_subdir)
    output_dir = resolve_project_relative(cfg, args.results_subdir)

    refit_summary = refit_selected_models(cfg, overwrite=args.overwrite)
    evaluate_refits_on_external(cfg, refit_summary, external_cache_dir, output_dir, ext_helpers)


if __name__ == "__main__":
    main()
