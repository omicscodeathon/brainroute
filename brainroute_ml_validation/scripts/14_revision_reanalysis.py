#!/usr/bin/env python3
"""Supplementary analyses from saved predictions, without model retraining."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.modeling import metric_dict
from brainroute_ml_validation.src.utils import (
    ensure_dirs,
    load_config,
    project_path,
    script_arg_parser,
    set_global_seed,
    write_csv,
)


PRIMARY_PREFIX = "scaffold_cv_fold"
LEGACY_REPEAT_PREFIX = "duplicate_aware_seed"


def ci95_t(values: pd.Series | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan, np.nan
    mean = float(arr.mean())
    half = float(t.ppf(0.975, len(arr) - 1) * arr.std(ddof=1) / np.sqrt(len(arr)))
    return mean - half, mean + half


def fixed_width_reliability(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(y_score, edges[1:-1], right=False), bins - 1)
    rows = []
    for bin_index in range(bins):
        mask = assignments == bin_index
        count = int(mask.sum())
        rows.append(
            {
                "bin": bin_index + 1,
                "lower_bound": edges[bin_index],
                "upper_bound": edges[bin_index + 1],
                "n": count,
                "mean_predicted_probability": float(y_score[mask].mean()) if count else np.nan,
                "observed_positive_fraction": float(y_true[mask].mean()) if count else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    nonempty = frame[frame["n"] > 0].copy()
    gaps = (nonempty["mean_predicted_probability"] - nonempty["observed_positive_fraction"]).abs()
    ece = float((gaps * nonempty["n"] / len(y_true)).sum())
    return frame, {
        "n": len(y_true),
        "positive_fraction": float(np.mean(y_true)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "expected_calibration_error_10_bins": ece,
        "maximum_calibration_error_10_bins": float(gaps.max()),
        "decision_threshold": 0.5,
        "threshold_selection": "fixed in advance; not optimized on held-out data",
    }


def split_audit(cfg: dict, standardized: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    split_root = project_path(cfg, "data/splits")
    for test_path in sorted(split_root.glob("*_test.csv")):
        split = test_path.name.removesuffix("_test.csv")
        if not (
            split.startswith(PRIMARY_PREFIX)
            or split.startswith(LEGACY_REPEAT_PREFIX)
            or split.startswith("scaffold_split")
        ):
            continue
        train = pd.read_csv(split_root / f"{split}_train.csv")
        test = pd.read_csv(test_path)
        train_scaffolds = set(train["murcko_scaffold"].fillna("NO_SCAFFOLD"))
        test_scaffolds = set(test["murcko_scaffold"].fillna("NO_SCAFFOLD"))
        train_keys = set(train["inchikey"])
        test_keys = set(test["inchikey"])
        if split.startswith(PRIMARY_PREFIX):
            display = "Primary five-fold scaffold-grouped cross-validation"
            role = "primary"
        elif split.startswith("scaffold_split"):
            display = "Secondary scaffold-group holdout"
            role = "secondary"
        else:
            display = "Secondary repeated random holdout on the InChIKey-deduplicated cohort"
            role = "secondary"
        rows.append(
            {
                "split_file_identifier": split,
                "manuscript_display_name": display,
                "validation_role": role,
                "train_n": len(train),
                "test_n": len(test),
                "train_bbb_positive": int((train["label"] == 1).sum()),
                "train_bbb_negative": int((train["label"] == 0).sum()),
                "test_bbb_positive": int((test["label"] == 1).sum()),
                "test_bbb_negative": int((test["label"] == 0).sum()),
                "test_positive_fraction": float(test["label"].mean()),
                "inchikey_overlap_count": len(train_keys & test_keys),
                "scaffold_overlap_count": len(train_scaffolds & test_scaffolds),
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["manuscript_display_name", "validation_role"], as_index=False)
        .agg(
            n_splits=("split_file_identifier", "size"),
            mean_test_n=("test_n", "mean"),
            min_test_positive_fraction=("test_positive_fraction", "min"),
            max_test_positive_fraction=("test_positive_fraction", "max"),
            total_inchikey_overlap=("inchikey_overlap_count", "sum"),
            maximum_scaffold_overlap=("scaffold_overlap_count", "max"),
        )
    )
    summary["deduplication_note"] = (
        "The modeling cohort contains one row per InChIKey. Exact-identity control is a curation step, not a validation strategy."
    )
    summary["inner_tuning_note"] = (
        "Hyperparameters were selected with three-fold StratifiedKFold inside each outer training partition; scaffold grouping was applied to the outer folds."
    )
    summary["final_modeling_n"] = len(standardized)
    summary["final_bbb_positive"] = int((standardized["label"] == 1).sum())
    summary["final_bbb_negative"] = int((standardized["label"] == 0).sum())
    return detail, summary


def common_molecule_metrics(cfg: dict, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    views = ["padel", "morgan", "padel_morgan", "embeddings", "padel_morgan_embeddings"]
    id_sets = []
    for view in views:
        index_path = project_path(cfg, f"data/processed/features_{view}_index.csv")
        id_sets.append(set(pd.read_csv(index_path)["molecule_id"]))
    common_ids = set.intersection(*id_sets)
    primary = predictions[
        predictions["split"].str.startswith(PRIMARY_PREFIX, na=False)
        & predictions["molecule_id"].isin(common_ids)
    ].copy()

    rows = []
    for (view, model), frame in primary.groupby(["feature_view", "model"]):
        frame = frame.drop_duplicates("molecule_id")
        metrics = metric_dict(frame["y_true"], frame["y_pred"], frame["y_score"])
        rows.append({"feature_view": view, "model": model, "n_common_oof_molecules": len(frame), **metrics})
    model_metrics = pd.DataFrame(rows)
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "mcc",
        "roc_auc",
        "auprc",
    ]
    feature_summary = model_metrics.groupby("feature_view", as_index=False)[metric_columns].mean()
    n_models = model_metrics.groupby("feature_view").size()
    feature_summary.insert(1, "n_models", feature_summary["feature_view"].map(n_models))
    feature_summary.insert(2, "n_common_oof_molecules_per_model", len(common_ids))
    feature_summary["comparison_note"] = (
        "All test predictions are restricted to the same molecules. Training cohorts for non-PaDEL views contain six additional molecules, so this is a saved-prediction sensitivity analysis rather than identical-cohort retraining."
    )
    return model_metrics, feature_summary, common_ids


def scaffold_fold_metrics(frame: pd.DataFrame, metric: str) -> np.ndarray:
    values = []
    for _, fold in frame.groupby("split"):
        if metric == "balanced_accuracy":
            values.append(balanced_accuracy_score(fold["y_true"], fold["y_pred"]))
        else:
            values.append(average_precision_score(fold["y_true"], fold["y_score"]))
    return np.asarray(values, dtype=float)


def cluster_bootstrap_difference(joined: pd.DataFrame, metric: str, repeats: int, seed: int) -> dict:
    grouped_indices = [group.index.to_numpy() for _, group in joined.groupby("murcko_scaffold", dropna=False)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(grouped_indices), size=len(grouped_indices))
        sample_index = np.concatenate([grouped_indices[i] for i in chosen])
        sample = joined.loc[sample_index]
        if sample["y_true"].nunique() < 2:
            continue
        if metric == "balanced_accuracy":
            first = balanced_accuracy_score(sample["y_true"], sample["y_pred_first"])
            second = balanced_accuracy_score(sample["y_true"], sample["y_pred_second"])
        else:
            first = average_precision_score(sample["y_true"], sample["y_score_first"])
            second = average_precision_score(sample["y_true"], sample["y_score_second"])
        deltas.append(first - second)
    arr = np.asarray(deltas, dtype=float)
    return {
        "bootstrap_repeats_requested": repeats,
        "bootstrap_repeats_valid": len(arr),
        "bootstrap_mean_difference": float(arr.mean()),
        "bootstrap_ci95_low": float(np.quantile(arr, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(arr, 0.975)),
        "bootstrap_two_sided_p_value": float(min(1.0, 2 * min(np.mean(arr <= 0), np.mean(arr >= 0)))),
    }


def paired_model_comparisons(primary: pd.DataFrame, common_ids: set[str], seed: int) -> pd.DataFrame:
    data = primary[primary["molecule_id"].isin(common_ids)].copy()
    configs = []
    for (view, model), frame in data.groupby(["feature_view", "model"]):
        frame = frame.drop_duplicates("molecule_id")
        configs.append(
            {
                "feature_view": view,
                "model": model,
                "balanced_accuracy": balanced_accuracy_score(frame["y_true"], frame["y_pred"]),
                "auprc": average_precision_score(frame["y_true"], frame["y_score"]),
            }
        )
    ranking = pd.DataFrame(configs)
    rows = []
    for metric in ["balanced_accuracy", "auprc"]:
        leaders = ranking.sort_values(metric, ascending=False).head(3).reset_index(drop=True)
        first = leaders.iloc[0]
        first_frame = data[
            (data["feature_view"] == first["feature_view"]) & (data["model"] == first["model"])
        ].drop_duplicates("molecule_id")
        for comparison_rank, second in leaders.iloc[1:].iterrows():
            second_frame = data[
                (data["feature_view"] == second["feature_view"]) & (data["model"] == second["model"])
            ].drop_duplicates("molecule_id")
            joined = first_frame.merge(
                second_frame,
                on="molecule_id",
                suffixes=("_first", "_second"),
                validate="one_to_one",
            )
            joined["y_true"] = joined["y_true_first"]
            joined["murcko_scaffold"] = joined["murcko_scaffold_first"].fillna("NO_SCAFFOLD")
            first_folds = scaffold_fold_metrics(first_frame, metric)
            second_folds = scaffold_fold_metrics(second_frame, metric)
            differences = first_folds - second_folds
            method = "exact" if not np.any(np.isclose(differences, 0.0)) else "auto"
            test = wilcoxon(first_folds, second_folds, alternative="two-sided", method=method)
            low, high = ci95_t(differences)
            row = {
                "metric": metric,
                "first_configuration": f"{first['feature_view']}/{first['model']}",
                "second_configuration": f"{second['feature_view']}/{second['model']}",
                "n_paired_scaffold_folds": len(first_folds),
                "mean_paired_fold_difference": float(differences.mean()),
                "paired_fold_difference_ci95_low": low,
                "paired_fold_difference_ci95_high": high,
                "wilcoxon_statistic": float(test.statistic),
                "wilcoxon_two_sided_p_value": float(test.pvalue),
                "wilcoxon_method": method,
                "n_paired_common_molecules": len(joined),
                "interpretation_note": "The five-fold Wilcoxon test is underpowered; non-significance is not evidence that models are equivalent. The scaffold-cluster bootstrap is a saved-prediction sensitivity analysis and does not replace repeated model refitting.",
            }
            row.update(cluster_bootstrap_difference(joined, metric, repeats=2000, seed=seed + int(comparison_rank)))
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_outputs(cfg: dict, primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = primary[
        (primary["feature_view"] == "padel_morgan_embeddings") & (primary["model"] == "xgboost")
    ].drop_duplicates("molecule_id")
    datasets = [
        (
            "Internal out-of-fold, PaDEL + Morgan + ChemBERTa XGBoost, scaffold-grouped CV",
            selected["y_true"].to_numpy(int),
            selected["y_score"].to_numpy(float),
        )
    ]
    external_path = project_path(cfg, "results/external_results_full_refit/external_validation_cleaned_predictions.csv")
    if external_path.exists():
        external = pd.read_csv(external_path)
        datasets.append(
            (
                "External non-overlapping cohort, three-model full-refit ensemble",
                external["label"].to_numpy(int),
                external["brainroute_ensemble_probability"].to_numpy(float),
            )
        )
    summary_rows = []
    bin_frames = []
    for name, y_true, y_score in datasets:
        bins, summary = fixed_width_reliability(y_true, y_score)
        bins.insert(0, "evaluation_set", name)
        summary_rows.append({"evaluation_set": name, **summary})
        bin_frames.append(bins)
    return pd.DataFrame(summary_rows), pd.concat(bin_frames, ignore_index=True)


def external_audit(cfg: dict, training: pd.DataFrame) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    root = project_path(cfg, "results/external_results_full_refit")
    cohorts = [
        ("Exact training overlaps removed", root / "external_training_overlaps_removed.csv"),
        ("PaDEL descriptor failures excluded", root / "external_padel_failed_molecules.csv"),
        ("Retained external validation cohort", root / "cleaned_non_overlapping_external_validation_dataframe.csv"),
    ]
    training_scaffolds = set(training["murcko_scaffold"].fillna("NO_SCAFFOLD"))
    training_scaffolds.discard("NO_SCAFFOLD")
    rows = []
    for name, path in cohorts:
        frame = pd.read_csv(path)
        scaffold = frame.get("murcko_scaffold", pd.Series(index=frame.index, dtype=object)).fillna("NO_SCAFFOLD")
        nonempty_scaffold = scaffold != "NO_SCAFFOLD"
        similarity = frame.get("max_tanimoto_to_training", pd.Series(index=frame.index, dtype=float))
        properties = []
        for smiles in frame["canonical_smiles"]:
            molecule = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
            properties.append(
                {
                    "molecular_weight": Descriptors.MolWt(molecule) if molecule is not None else np.nan,
                    "logp": Crippen.MolLogP(molecule) if molecule is not None else np.nan,
                    "tpsa": Descriptors.TPSA(molecule) if molecule is not None else np.nan,
                }
            )
        properties = pd.DataFrame(properties)
        rows.append(
            {
                "cohort": name,
                "n": len(frame),
                "bbb_positive": int((frame["label"] == 1).sum()),
                "bbb_negative": int((frame["label"] == 0).sum()),
                "positive_fraction": float(frame["label"].mean()),
                "mean_molecular_weight": float(properties["molecular_weight"].mean()),
                "median_molecular_weight": float(properties["molecular_weight"].median()),
                "mean_logp": float(properties["logp"].mean()),
                "median_logp": float(properties["logp"].median()),
                "mean_tpsa": float(properties["tpsa"].mean()),
                "median_tpsa": float(properties["tpsa"].median()),
                "mean_max_tanimoto_to_training": float(similarity.mean()) if similarity.notna().any() else np.nan,
                "median_max_tanimoto_to_training": float(similarity.median()) if similarity.notna().any() else np.nan,
                "n_max_tanimoto_ge_0_8": int((similarity >= 0.8).sum()) if similarity.notna().any() else np.nan,
                "n_max_tanimoto_ge_0_9": int((similarity >= 0.9).sum()) if similarity.notna().any() else np.nan,
                "n_without_nonempty_scaffold": int((~nonempty_scaffold).sum()),
                "n_with_nonempty_scaffold_seen_in_training": int(
                    scaffold[nonempty_scaffold].isin(training_scaffolds).sum()
                ),
                "unique_nonempty_scaffolds": int(scaffold[nonempty_scaffold].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = script_arg_parser("Run supplementary saved-prediction reanalysis.").parse_args()
    cfg = load_config(args.config)
    output_cfg = dict(cfg)
    output_cfg["overwrite"] = True
    seed = int(cfg.get("random_seed", 42))
    set_global_seed(seed)
    ensure_dirs(cfg)

    standardized = pd.read_csv(project_path(cfg, "data/processed/standardized_molecules.csv"))
    predictions = pd.read_csv(project_path(cfg, "reports/model_predictions_all_splits.csv"))
    primary = predictions[predictions["split"].str.startswith(PRIMARY_PREFIX, na=False)].copy()

    positives = int((standardized["label"] == 1).sum())
    negatives = int((standardized["label"] == 0).sum())
    class_balance = pd.DataFrame(
        [
            {
                "cohort": "Final InChIKey-deduplicated modeling cohort",
                "n": len(standardized),
                "bbb_positive": positives,
                "bbb_negative": negatives,
                "positive_fraction": positives / len(standardized),
                "positive_to_negative_ratio": positives / negatives,
            }
        ]
    )
    model_weight_policy = pd.DataFrame(
        [
            {"model": "Logistic Regression", "imbalance_handling": "class_weight='balanced'"},
            {"model": "Random Forest", "imbalance_handling": "class_weight='balanced'"},
            {"model": "Extra Trees", "imbalance_handling": "class_weight='balanced'"},
            {"model": "LightGBM", "imbalance_handling": "class_weight='balanced'"},
            {"model": "XGBoost", "imbalance_handling": "scale_pos_weight = n_negative / n_positive in each outer training partition"},
            {"model": "K-Nearest Neighbors", "imbalance_handling": "no class-weight parameter; distance weighting was among tuned options"},
        ]
    )

    audit_detail, audit_summary = split_audit(cfg, standardized)
    common_metrics, feature_summary, common_ids = common_molecule_metrics(cfg, predictions)
    comparisons = paired_model_comparisons(primary, common_ids, seed)
    calibration_summary, reliability_bins = calibration_outputs(cfg, primary)
    external = external_audit(cfg, standardized)

    write_csv(class_balance, project_path(cfg, "reports/revision_class_balance.csv"), output_cfg)
    write_csv(model_weight_policy, project_path(cfg, "reports/revision_model_weight_policy.csv"), output_cfg)
    write_csv(audit_detail, project_path(cfg, "reports/revision_validation_split_audit.csv"), output_cfg)
    write_csv(audit_summary, project_path(cfg, "reports/revision_validation_strategy_summary.csv"), output_cfg)
    write_csv(common_metrics, project_path(cfg, "reports/revision_common_molecule_model_metrics.csv"), output_cfg)
    write_csv(feature_summary, project_path(cfg, "reports/revision_common_molecule_feature_summary.csv"), output_cfg)
    write_csv(comparisons, project_path(cfg, "reports/revision_paired_model_comparisons.csv"), output_cfg)
    write_csv(calibration_summary, project_path(cfg, "reports/revision_calibration_summary.csv"), output_cfg)
    write_csv(reliability_bins, project_path(cfg, "reports/revision_reliability_bins.csv"), output_cfg)
    write_csv(external, project_path(cfg, "reports/revision_external_attrition_audit.csv"), output_cfg)


if __name__ == "__main__":
    main()
