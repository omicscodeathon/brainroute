#!/usr/bin/env python3
"""Matched Uni-Mol, GFN2-xTB, and PaDEL/Morgan benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import binomtest, fisher_exact, ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = ROOT / "brainroute_ml_validation"
DEFAULT_CONFIG = VALIDATION_ROOT / "configs/3d_qm_benchmark.yaml"
METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "negative_predictive_value",
    "f1",
    "mcc",
    "roc_auc",
    "auprc",
    "brier_score",
    "expected_calibration_error",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else VALIDATION_ROOT / candidate


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    bins = np.minimum((y_score * 10).astype(int), 9)
    ece = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        if mask.any():
            ece += mask.mean() * abs(y_score[mask].mean() - y_true[mask].mean())
    return {
        "n": len(y_true),
        "bbb_positive": int(y_true.sum()),
        "bbb_negative": int((y_true == 0).sum()),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else math.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "negative_predictive_value": tn / (tn + fn) if tn + fn else math.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score) if np.unique(y_true).size == 2 else math.nan,
        "auprc": average_precision_score(y_true, y_score) if np.unique(y_true).size == 2 else math.nan,
        "brier_score": brier_score_loss(y_true, y_score),
        "expected_calibration_error": ece,
        "threshold": threshold,
    }


def metric_value(y_true: np.ndarray, y_score: np.ndarray, name: str, threshold: float) -> float:
    y_pred = y_score >= threshold
    if name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if name == "auprc":
        return average_precision_score(y_true, y_score)
    if name == "sensitivity":
        return recall_score(y_true, y_pred, zero_division=0)
    if name == "specificity":
        tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fp) if tn + fp else math.nan
    raise ValueError(name)


def cluster_bootstrap_ci(
    frame: pd.DataFrame,
    score_column: str,
    metric_name: str,
    repeats: int,
    seed: int,
    threshold: float,
) -> tuple[float, float]:
    groups = [group.index.to_numpy() for _, group in frame.groupby("murcko_scaffold", dropna=False)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[position] for position in chosen])
        sampled = frame.loc[indices]
        if sampled["label"].nunique() < 2:
            continue
        values.append(
            metric_value(
                sampled["label"].to_numpy(),
                sampled[score_column].to_numpy(),
                metric_name,
                threshold,
            )
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    first: str,
    second: str,
    metric_name: str,
    repeats: int,
    seed: int,
    threshold: float,
) -> dict:
    groups = [group.index.to_numpy() for _, group in frame.groupby("murcko_scaffold", dropna=False)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[position] for position in chosen])
        sampled = frame.loc[indices]
        if sampled["label"].nunique() < 2:
            continue
        first_value = metric_value(sampled["label"].to_numpy(), sampled[first].to_numpy(), metric_name, threshold)
        second_value = metric_value(sampled["label"].to_numpy(), sampled[second].to_numpy(), metric_name, threshold)
        deltas.append(first_value - second_value)
    values = np.asarray(deltas, dtype=float)
    return {
        "mean_bootstrap_difference": float(values.mean()),
        "difference_ci95_low": float(np.quantile(values, 0.025)),
        "difference_ci95_high": float(np.quantile(values, 0.975)),
        "two_sided_bootstrap_p_value": float(min(1.0, 2 * min(np.mean(values <= 0), np.mean(values >= 0)))),
        "valid_bootstrap_repeats": len(values),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def mcnemar_exact(y_true: np.ndarray, first_score: np.ndarray, second_score: np.ndarray, threshold: float) -> dict:
    first_correct = (first_score >= threshold).astype(int) == y_true
    second_correct = (second_score >= threshold).astype(int) == y_true
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(first_only, second_only), discordant, 0.5).pvalue)
    return {
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "discordant_predictions": discordant,
        "mcnemar_exact_p_value": p_value,
    }


def morgan_frame(smiles: list[str], columns: list[str]) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=len(columns), includeChirality=True)
    matrix = np.zeros((len(smiles), len(columns)), dtype=np.uint8)
    for position, value in enumerate(smiles):
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            raise ValueError(f"Morgan generation failed at row {position}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), matrix[position])
    return pd.DataFrame(matrix, columns=columns)


def load_base_features(cfg: dict, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = cfg["paths"]
    internal_index = pd.read_csv(resolve(paths["internal_index"]))
    internal_features = pd.read_csv(resolve(paths["internal_features"])).apply(pd.to_numeric, errors="coerce")
    if len(internal_index) != len(internal_features):
        raise ValueError("Internal index and feature matrix lengths differ")
    internal_features.index = "internal:" + internal_index["molecule_id"].astype(str)

    external_manifest = manifest[manifest["cohort"] == "external"].reset_index(drop=True)
    external_padel_raw = pd.read_csv(resolve(paths["external_padel"])).apply(pd.to_numeric, errors="coerce")
    with resolve(paths["external_padel_failed_positions"]).open("r", encoding="utf-8") as handle:
        failed = set(json.load(handle))
    external_padel = external_padel_raw.loc[
        [position for position in external_padel_raw.index if position not in failed]
    ].reset_index(drop=True)
    if len(external_padel) != len(external_manifest):
        raise ValueError(
            f"External PaDEL alignment failed: {len(external_padel)} retained features for {len(external_manifest)} molecules"
        )
    padel_columns = [column for column in internal_features.columns if column.startswith("padel__")]
    morgan_columns = [column for column in internal_features.columns if column.startswith("morgan__")]
    external_padel = external_padel.reindex(columns=padel_columns)
    external_morgan = morgan_frame(external_manifest["canonical_smiles"].tolist(), morgan_columns)
    external_features = pd.concat([external_padel, external_morgan], axis=1).reindex(columns=internal_features.columns)
    external_features.index = external_manifest["benchmark_id"].astype(str)
    # PaDEL uses the largest IEEE-754 float as a sentinel for some undefined
    # descriptors (notably gmin). XGBoost stores feature values as float32, so
    # treat values outside that range as missing rather than allowing overflow.
    float32_limit = np.finfo(np.float32).max
    internal_features = internal_features.replace([np.inf, -np.inf], np.nan)
    external_features = external_features.replace([np.inf, -np.inf], np.nan)
    internal_features = internal_features.mask(internal_features.abs() > float32_limit)
    external_features = external_features.mask(external_features.abs() > float32_limit)
    return internal_features, external_features


def load_views(cfg: dict, manifest: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    workspace = resolve(cfg["paths"]["external_workspace"])
    xtb = pd.read_csv(workspace / "artifacts/xtb_features.csv")
    xtb = xtb[xtb["status"] == "success"].set_index("benchmark_id")
    xtb_columns = [column for column in xtb.columns if column.startswith("xtb__")]
    unimol_index = pd.read_csv(workspace / "artifacts/unimol_v1_index.csv")
    unimol_array = np.load(workspace / "artifacts/unimol_v1_representations.npy", mmap_mode="r")
    if len(unimol_index) != len(unimol_array):
        raise ValueError("Uni-Mol index and representation lengths differ")
    unimol_columns = [f"unimol__{position}" for position in range(unimol_array.shape[1])]
    unimol = pd.DataFrame(np.asarray(unimol_array, dtype=np.float32), index=unimol_index["benchmark_id"], columns=unimol_columns)

    base_internal, base_external = load_base_features(cfg, manifest)
    if "unimol_3d_conformer" in unimol_index:
        unimol_3d_ids = set(unimol_index.loc[unimol_index["unimol_3d_conformer"].astype(bool), "benchmark_id"])
    else:
        unimol_3d_ids = set(unimol.index)
    common = set(xtb.index) & set(unimol.index) & unimol_3d_ids
    internal_ids = [identifier for identifier in base_internal.index if identifier in common]
    external_ids = [identifier for identifier in base_external.index if identifier in common]
    analysis_manifest = manifest[manifest["benchmark_id"].isin(internal_ids + external_ids)].copy()
    analysis_manifest["feature_complete"] = True

    views_internal = {
        "padel_morgan_control": base_internal.loc[internal_ids].astype(np.float32),
        "padel_morgan_plus_gfn2_xtb": pd.concat(
            [base_internal.loc[internal_ids], xtb.loc[internal_ids, xtb_columns]], axis=1
        ).astype(np.float32),
        "unimol_v1_frozen_representation": unimol.loc[internal_ids].astype(np.float32),
    }
    views_external = {
        "padel_morgan_control": base_external.loc[external_ids].astype(np.float32),
        "padel_morgan_plus_gfn2_xtb": pd.concat(
            [base_external.loc[external_ids], xtb.loc[external_ids, xtb_columns]], axis=1
        ).astype(np.float32),
        "unimol_v1_frozen_representation": unimol.loc[external_ids].astype(np.float32),
    }
    return views_internal, views_external, analysis_manifest


def make_estimator(cfg: dict, y: np.ndarray, seed: int) -> XGBClassifier:
    model_cfg = cfg["models"]
    positives = max(1, int(np.sum(y == 1)))
    negatives = max(1, int(np.sum(y == 0)))
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=int(model_cfg.get("n_jobs", 4)),
        learning_rate=float(model_cfg["learning_rate"]),
        subsample=float(model_cfg["subsample"]),
        colsample_bytree=float(model_cfg["colsample_bytree"]),
        scale_pos_weight=negatives / positives,
    )


def tune_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[XGBClassifier, dict, float]:
    inner = StratifiedGroupKFold(
        n_splits=int(cfg["validation"]["inner_stratified_group_folds"]),
        shuffle=True,
        random_state=seed,
    )
    estimator = make_estimator(cfg, y, seed)
    grid = {
        "n_estimators": [int(value) for value in cfg["models"]["n_estimators"]],
        "max_depth": [int(value) for value in cfg["models"]["max_depth"]],
    }
    search = GridSearchCV(
        estimator,
        grid,
        scoring="balanced_accuracy",
        cv=inner,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def outer_splits(ids: pd.Index) -> list[tuple[str, np.ndarray, np.ndarray]]:
    position = {identifier: index for index, identifier in enumerate(ids)}
    rows = []
    for fold in range(1, 6):
        train = pd.read_csv(VALIDATION_ROOT / f"data/splits/scaffold_cv_fold{fold}_train.csv")
        test = pd.read_csv(VALIDATION_ROOT / f"data/splits/scaffold_cv_fold{fold}_test.csv")
        train_ids = [f"internal:{value}" for value in train["molecule_id"] if f"internal:{value}" in position]
        test_ids = [f"internal:{value}" for value in test["molecule_id"] if f"internal:{value}" in position]
        rows.append(
            (
                f"scaffold_cv_fold{fold}",
                np.asarray([position[value] for value in train_ids]),
                np.asarray([position[value] for value in test_ids]),
            )
        )
    return rows


def run_outer_benchmark(
    views_internal: dict[str, pd.DataFrame],
    views_external: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workspace = resolve(cfg["paths"]["external_workspace"])
    model_dir = workspace / "artifacts/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    internal_meta = manifest[manifest["cohort"] == "internal"].set_index("benchmark_id")
    external_meta = manifest[manifest["cohort"] == "external"].set_index("benchmark_id")
    threshold = float(cfg["validation"]["threshold"])
    internal_prediction_rows = []
    external_prediction_rows = []
    fold_metric_rows = []
    tuning_rows = []

    reference_ids = next(iter(views_internal.values())).index
    splits = outer_splits(reference_ids)
    for view_name, X in views_internal.items():
        external_X = views_external[view_name]
        meta = internal_meta.loc[X.index]
        y = meta["label"].astype(int).to_numpy()
        groups = meta["murcko_scaffold"].fillna("NO_SCAFFOLD").astype(str).to_numpy()
        for fold_number, (fold_name, train_indices, test_indices) in enumerate(splits, start=1):
            started = time.perf_counter()
            model, best_params, inner_score = tune_model(
                X.iloc[train_indices],
                y[train_indices],
                groups[train_indices],
                cfg,
                seed=int(cfg["random_seed"]) + fold_number,
            )
            internal_score = model.predict_proba(X.iloc[test_indices])[:, 1]
            external_score = model.predict_proba(external_X)[:, 1]
            fold_metrics = metrics(y[test_indices], internal_score, threshold)
            fold_metrics.update({"cohort": "internal_outer_fold", "configuration": view_name, "fold": fold_name})
            fold_metric_rows.append(fold_metrics)
            model_path = model_dir / f"{view_name}__{fold_name}.joblib"
            joblib.dump(model, model_path)
            tuning_rows.append(
                {
                    "configuration": view_name,
                    "fold": fold_name,
                    "inner_cv": "StratifiedGroupKFold",
                    "inner_folds": int(cfg["validation"]["inner_stratified_group_folds"]),
                    "inner_group": "Bemis-Murcko scaffold",
                    "inner_best_balanced_accuracy": inner_score,
                    "best_parameters": json.dumps(best_params, sort_keys=True),
                    "train_n": len(train_indices),
                    "test_n": len(test_indices),
                    "wall_seconds": time.perf_counter() - started,
                    "model_artifact": str(model_path),
                }
            )
            test_ids = X.index[test_indices]
            for identifier, score in zip(test_ids, internal_score):
                internal_prediction_rows.append(
                    {
                        "benchmark_id": identifier,
                        "configuration": view_name,
                        "fold": fold_name,
                        "label": int(internal_meta.loc[identifier, "label"]),
                        "murcko_scaffold": internal_meta.loc[identifier, "murcko_scaffold"],
                        "y_score": float(score),
                        "y_pred": int(score >= threshold),
                    }
                )
            for identifier, score in zip(external_X.index, external_score):
                external_prediction_rows.append(
                    {
                        "benchmark_id": identifier,
                        "configuration": view_name,
                        "fold": fold_name,
                        "label": int(external_meta.loc[identifier, "label"]),
                        "murcko_scaffold": external_meta.loc[identifier, "murcko_scaffold"],
                        "y_score": float(score),
                    }
                )
            print(
                f"configuration={view_name} fold={fold_name} inner_BA={inner_score:.4f} test_BA={fold_metrics['balanced_accuracy']:.4f}",
                flush=True,
            )

    internal_predictions = pd.DataFrame(internal_prediction_rows)
    external_fold_predictions = pd.DataFrame(external_prediction_rows)
    external_predictions = (
        external_fold_predictions.groupby(
            ["benchmark_id", "configuration", "label", "murcko_scaffold"], as_index=False
        )["y_score"]
        .mean()
    )
    external_predictions["y_pred"] = (external_predictions["y_score"] >= threshold).astype(int)
    return internal_predictions, external_predictions, pd.DataFrame(fold_metric_rows), pd.DataFrame(tuning_rows)


def summarize_predictions(
    internal_predictions: pd.DataFrame,
    external_predictions: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(cfg["validation"]["threshold"])
    repeats = int(cfg["validation"]["bootstrap_repeats"])
    rows = []
    for cohort, predictions in [("internal_scaffold_oof", internal_predictions), ("external_independent", external_predictions)]:
        for position, (configuration, frame) in enumerate(predictions.groupby("configuration")):
            row = {"cohort": cohort, "configuration": configuration, **metrics(frame["label"], frame["y_score"], threshold)}
            for metric_name in ["balanced_accuracy", "auprc", "sensitivity", "specificity"]:
                low, high = cluster_bootstrap_ci(
                    frame.reset_index(drop=True),
                    "y_score",
                    metric_name,
                    repeats,
                    int(cfg["random_seed"]) + position,
                    threshold,
                )
                row[f"{metric_name}_ci95_low"] = low
                row[f"{metric_name}_ci95_high"] = high
            rows.append(row)

    comparisons = []
    control = "padel_morgan_control"
    for cohort, predictions in [("internal_scaffold_oof", internal_predictions), ("external_independent", external_predictions)]:
        pivot = predictions.pivot(index="benchmark_id", columns="configuration", values="y_score")
        meta = predictions.drop_duplicates("benchmark_id").set_index("benchmark_id")[["label", "murcko_scaffold"]]
        paired = meta.join(pivot, how="inner").dropna().reset_index(drop=True)
        for comparison_number, candidate in enumerate(["padel_morgan_plus_gfn2_xtb", "unimol_v1_frozen_representation"]):
            for metric_name in ["balanced_accuracy", "auprc"]:
                row = {
                    "cohort": cohort,
                    "first_configuration": candidate,
                    "second_configuration": control,
                    "metric": metric_name,
                    "observed_first": metric_value(paired["label"].to_numpy(), paired[candidate].to_numpy(), metric_name, threshold),
                    "observed_second": metric_value(paired["label"].to_numpy(), paired[control].to_numpy(), metric_name, threshold),
                }
                row["observed_difference"] = row["observed_first"] - row["observed_second"]
                row.update(
                    paired_cluster_bootstrap(
                        paired,
                        candidate,
                        control,
                        metric_name,
                        repeats,
                        int(cfg["random_seed"]) + comparison_number,
                        threshold,
                    )
                )
                row.update(mcnemar_exact(paired["label"].to_numpy(), paired[candidate].to_numpy(), paired[control].to_numpy(), threshold))
                comparisons.append(row)
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["holm_adjusted_bootstrap_p_value"] = holm_adjust(
        comparison_frame["two_sided_bootstrap_p_value"].tolist()
    )
    comparison_frame["holm_adjusted_mcnemar_p_value"] = holm_adjust(
        comparison_frame["mcnemar_exact_p_value"].tolist()
    )
    return pd.DataFrame(rows), comparison_frame


def chemical_space_audit(
    manifest: pd.DataFrame,
    external_predictions: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors

    descriptors = []
    for row in manifest.itertuples():
        mol = Chem.MolFromSmiles(row.calculation_smiles)
        descriptors.append(
            {
                "benchmark_id": row.benchmark_id,
                "cohort": row.cohort,
                "label": int(row.label),
                "molecular_weight": float(rdMolDescriptors.CalcExactMolWt(mol)),
                "logp": float(Crippen.MolLogP(mol)),
                "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
                "hydrogen_bond_donors": float(Lipinski.NumHDonors(mol)),
                "hydrogen_bond_acceptors": float(Lipinski.NumHAcceptors(mol)),
                "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
                "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
                "ring_count": float(Lipinski.RingCount(mol)),
                "formal_charge": float(row.calculation_formal_charge),
            }
        )
    frame = pd.DataFrame(descriptors)
    descriptor_columns = [
        "molecular_weight",
        "logp",
        "tpsa",
        "hydrogen_bond_donors",
        "hydrogen_bond_acceptors",
        "rotatable_bonds",
        "fraction_csp3",
        "ring_count",
        "formal_charge",
    ]
    summary_rows = []
    for (cohort, label), group in frame.groupby(["cohort", "label"]):
        for descriptor in descriptor_columns:
            summary_rows.append(
                {
                    "cohort": cohort,
                    "label": label,
                    "descriptor": descriptor,
                    "n": len(group),
                    "mean": group[descriptor].mean(),
                    "standard_deviation": group[descriptor].std(ddof=1),
                    "median": group[descriptor].median(),
                    "interquartile_range": group[descriptor].quantile(0.75) - group[descriptor].quantile(0.25),
                }
            )
    difference_rows = []
    internal = frame[frame["cohort"] == "internal"]
    external = frame[frame["cohort"] == "external"]
    for descriptor in descriptor_columns:
        first = internal[descriptor].to_numpy()
        second = external[descriptor].to_numpy()
        pooled = math.sqrt((np.var(first, ddof=1) + np.var(second, ddof=1)) / 2)
        difference_rows.append(
            {
                "descriptor": descriptor,
                "internal_mean": first.mean(),
                "external_mean": second.mean(),
                "standardized_mean_difference_external_minus_internal": (second.mean() - first.mean()) / pooled if pooled else math.nan,
                "kolmogorov_smirnov_statistic": ks_2samp(first, second).statistic,
                "kolmogorov_smirnov_two_sided_p_value": ks_2samp(first, second).pvalue,
            }
        )
    table = pd.crosstab(frame["cohort"], frame["label"]).reindex(index=["internal", "external"], columns=[0, 1])
    odds_ratio, prevalence_p = fisher_exact(table.to_numpy())
    difference_rows.append(
        {
            "descriptor": "BBB_positive_class_prevalence",
            "internal_mean": internal["label"].mean(),
            "external_mean": external["label"].mean(),
            "standardized_mean_difference_external_minus_internal": math.nan,
            "kolmogorov_smirnov_statistic": odds_ratio,
            "kolmogorov_smirnov_two_sided_p_value": prevalence_p,
        }
    )

    external_source = pd.read_csv(resolve(cfg["paths"]["external_cohort"]))
    similarity = external_source[["inchikey", "max_tanimoto_to_training"]].copy()
    similarity["benchmark_id"] = "external:" + similarity["inchikey"].astype(str)
    predictions = external_predictions.merge(similarity[["benchmark_id", "max_tanimoto_to_training"]], on="benchmark_id", how="left")
    predictions["similarity_bin"] = pd.cut(
        predictions["max_tanimoto_to_training"],
        bins=[0.0, 0.4, 0.6, 0.8, 1.01],
        labels=["[0.0,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.01]"],
        right=False,
        include_lowest=True,
    )
    similarity_rows = []
    threshold = float(cfg["validation"]["threshold"])
    for (configuration, similarity_bin), group in predictions.groupby(
        ["configuration", "similarity_bin"], observed=True
    ):
        similarity_rows.append(
            {
                "configuration": configuration,
                "similarity_bin": str(similarity_bin),
                "mean_max_tanimoto": group["max_tanimoto_to_training"].mean(),
                **metrics(group["label"], group["y_score"], threshold),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(difference_rows), pd.DataFrame(similarity_rows)


def source_validation(
    views: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = manifest[manifest["cohort"] == "internal"].set_index("benchmark_id")
    threshold = float(cfg["validation"]["threshold"])
    definitions = [
        ("B3DB-exclusive_to_MoleculeNet-exclusive", "B3DB", "MoleculeNet BBBP"),
        ("MoleculeNet-exclusive_to_B3DB-exclusive", "MoleculeNet BBBP", "B3DB"),
    ]
    metric_rows = []
    prediction_rows = []
    for view_number, (view_name, X) in enumerate(views.items()):
        available_meta = meta.loc[X.index]
        for direction, train_source, test_source in definitions:
            train_mask = available_meta["provenance_sources"].fillna("").eq(train_source).to_numpy()
            test_mask = available_meta["provenance_sources"].fillna("").eq(test_source).to_numpy()
            X_train = X.iloc[np.flatnonzero(train_mask)]
            X_test = X.iloc[np.flatnonzero(test_mask)]
            train_meta = available_meta.iloc[np.flatnonzero(train_mask)]
            test_meta = available_meta.iloc[np.flatnonzero(test_mask)]
            y_train = train_meta["label"].astype(int).to_numpy()
            y_test = test_meta["label"].astype(int).to_numpy()
            model, best_params, inner_score = tune_model(
                X_train,
                y_train,
                train_meta["murcko_scaffold"].fillna("NO_SCAFFOLD").astype(str).to_numpy(),
                cfg,
                int(cfg["random_seed"]) + 100 + view_number,
            )
            score = model.predict_proba(X_test)[:, 1]
            row = {
                "configuration": view_name,
                "direction": direction,
                "train_source": train_source,
                "test_source": test_source,
                "train_n": len(X_train),
                "inner_best_balanced_accuracy": inner_score,
                "best_parameters": json.dumps(best_params, sort_keys=True),
                **metrics(y_test, score, threshold),
            }
            metric_rows.append(row)
            for identifier, label, scaffold, value in zip(X_test.index, y_test, test_meta["murcko_scaffold"], score):
                prediction_rows.append(
                    {
                        "benchmark_id": identifier,
                        "configuration": view_name,
                        "direction": direction,
                        "label": int(label),
                        "murcko_scaffold": scaffold,
                        "y_score": float(value),
                        "y_pred": int(value >= threshold),
                    }
                )
            print(
                f"source_validation={direction} configuration={view_name} test_BA={row['balanced_accuracy']:.4f}",
                flush=True,
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def feature_and_split_audit(
    views: dict[str, pd.DataFrame], manifest: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = manifest[manifest["cohort"] == "internal"].set_index("benchmark_id")
    feature_rows = []
    for name, frame in views.items():
        feature_rows.append(
            {
                "configuration": name,
                "n_molecules": len(frame),
                "n_features": frame.shape[1],
                "missing_values": int(frame.isna().sum().sum()),
                "molecules_with_any_missing_value": int(frame.isna().any(axis=1).sum()),
            }
        )
    split_rows = []
    for fold_name, train_indices, test_indices in outer_splits(next(iter(views.values())).index):
        ids = next(iter(views.values())).index
        train_meta = meta.loc[ids[train_indices]]
        test_meta = meta.loc[ids[test_indices]]
        split_rows.append(
            {
                "fold": fold_name,
                "train_n": len(train_meta),
                "test_n": len(test_meta),
                "train_positive": int(train_meta["label"].sum()),
                "test_positive": int(test_meta["label"].sum()),
                "inchikey_overlap": len(set(train_meta["inchikey"]) & set(test_meta["inchikey"])),
                "scaffold_overlap": len(
                    set(train_meta["murcko_scaffold"].fillna("NO_SCAFFOLD"))
                    & set(test_meta["murcko_scaffold"].fillna("NO_SCAFFOLD"))
                ),
            }
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(split_rows)


def write_outputs(frames: dict[str, pd.DataFrame], cfg: dict) -> None:
    workspace = resolve(cfg["paths"]["external_workspace"])
    artifact_dir = workspace / "artifacts"
    report_dir = VALIDATION_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for name, frame in frames.items():
        artifact_path = artifact_dir / f"{name}.csv"
        report_path = report_dir / f"{name}.csv"
        frame.to_csv(artifact_path, index=False)
        frame.to_csv(report_path, index=False)
        manifest_rows.append(
            {
                "output": name,
                "artifact_path": f"artifacts/{artifact_path.name}",
                "report_path": str(report_path.relative_to(ROOT)),
                "rows": len(frame),
                "sha256": file_hash(artifact_path),
            }
        )
    output_manifest = pd.DataFrame(manifest_rows)
    output_manifest.to_csv(artifact_dir / "matched_benchmark_output_manifest.csv", index=False)
    output_manifest.to_csv(report_dir / "matched_benchmark_output_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-source-validation", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    workspace = resolve(cfg["paths"]["external_workspace"])
    manifest = pd.read_csv(workspace / "artifacts/benchmark_manifest.csv")
    views_internal, views_external, analysis_manifest = load_views(cfg, manifest)
    feature_audit, split_audit = feature_and_split_audit(views_internal, analysis_manifest, cfg)
    internal_predictions, external_predictions, fold_metrics, tuning = run_outer_benchmark(
        views_internal, views_external, analysis_manifest, cfg
    )
    performance, comparisons = summarize_predictions(internal_predictions, external_predictions, cfg)
    chemical_summary, chemical_differences, external_similarity = chemical_space_audit(
        analysis_manifest, external_predictions, cfg
    )
    frames = {
        "matched_benchmark_feature_audit": feature_audit,
        "matched_benchmark_scaffold_split_audit": split_audit,
        "matched_benchmark_internal_oof_predictions": internal_predictions,
        "matched_benchmark_external_predictions": external_predictions,
        "matched_benchmark_fold_metrics": fold_metrics,
        "matched_benchmark_nested_tuning": tuning,
        "matched_benchmark_performance_summary": performance,
        "matched_benchmark_paired_comparisons": comparisons,
        "matched_benchmark_chemical_space_summary": chemical_summary,
        "matched_benchmark_chemical_space_differences": chemical_differences,
        "matched_benchmark_external_similarity_bin_performance": external_similarity,
    }
    if not args.skip_source_validation:
        source_metrics, source_predictions = source_validation(views_internal, analysis_manifest, cfg)
        frames["source_validation_metrics"] = source_metrics
        frames["source_validation_predictions"] = source_predictions
    analysis_manifest.to_csv(workspace / "artifacts/matched_benchmark_analysis_manifest.csv", index=False)
    frames["matched_benchmark_cohort_summary"] = (
        analysis_manifest.groupby("cohort", as_index=False)
        .agg(n=("benchmark_id", "size"), bbb_positive=("label", "sum"), unique_scaffolds=("murcko_scaffold", "nunique"))
    )
    frames["matched_benchmark_cohort_summary"]["bbb_negative"] = (
        frames["matched_benchmark_cohort_summary"]["n"]
        - frames["matched_benchmark_cohort_summary"]["bbb_positive"]
    )
    write_outputs(frames, cfg)
    print(performance.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
