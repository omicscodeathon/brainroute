#!/usr/bin/env python3
"""Add confidence intervals and paired tests for the existing deployed models.

This script is intentionally read-only with respect to the original validation
artifacts.  It reads the saved, overlap-free external predictions and writes new
supplementary uncertainty tables under ``reports/``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "results"
    / "external_results_full_refit"
    / "external_validation_cleaned_predictions.csv"
)
REPORTS = ROOT / "reports"
N_BOOTSTRAP = 2000
SEED = 20260903

MODELS = {
    "PaDEL + Morgan LightGBM": (
        "padel_morgan_lightgbm_full_data_refit_probability",
        "padel_morgan_lightgbm_full_data_refit_prediction",
    ),
    "PaDEL + Morgan Extra Trees": (
        "padel_morgan_extra_trees_full_data_refit_probability",
        "padel_morgan_extra_trees_full_data_refit_prediction",
    ),
    "PaDEL + Morgan + ChemBERTa XGBoost": (
        "padel_morgan_chemberta_xgboost_full_data_refit_probability",
        "padel_morgan_chemberta_xgboost_full_data_refit_prediction",
    ),
    "BrainRoute probability-averaged ensemble": (
        "brainroute_ensemble_probability",
        "brainroute_ensemble_prediction",
    ),
}


def scaffold_group(smiles: str, row_number: int) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid::{row_number}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold if scaffold else f"acyclic::{row_number}"


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    specificity = recall_score(y, pred, pos_label=0, zero_division=0)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y, pred, zero_division=0),
        "npv": precision_score(y, pred, pos_label=0, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, score),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = roc_auc_score(y, score)
        out["auprc"] = average_precision_score(y, score)
    else:
        out["auroc"] = np.nan
        out["auprc"] = np.nan
    return out


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    frame = pd.read_csv(INPUT)
    frame["scaffold_group"] = [
        scaffold_group(s, i) for i, s in enumerate(frame["canonical_smiles"])
    ]
    groups = frame.groupby("scaffold_group", sort=True).indices
    group_names = np.array(list(groups))
    rng = np.random.default_rng(SEED)
    bootstrap_indices: list[np.ndarray] = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(group_names, size=len(group_names), replace=True)
        bootstrap_indices.append(
            np.concatenate([np.asarray(groups[name], dtype=int) for name in sampled])
        )

    y = frame["label"].astype(int).to_numpy()
    model_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ci_rows: list[dict[str, object]] = []
    for name, (score_column, pred_column) in MODELS.items():
        score = frame[score_column].astype(float).to_numpy()
        pred = frame[pred_column].map({"BBB+": 1, "BBB-": 0, 1: 1, 0: 0}).astype(int).to_numpy()
        model_arrays[name] = (score, pred)
        observed = metrics(y, score, pred)
        sampled_metrics = {metric: [] for metric in observed}
        for index in bootstrap_indices:
            values = metrics(y[index], score[index], pred[index])
            for metric, value in values.items():
                if np.isfinite(value):
                    sampled_metrics[metric].append(value)
        for metric, estimate in observed.items():
            values = np.asarray(sampled_metrics[metric], dtype=float)
            ci_rows.append(
                {
                    "model_configuration": name,
                    "n_external": len(frame),
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": np.percentile(values, 2.5),
                    "ci95_high": np.percentile(values, 97.5),
                    "bootstrap_unit": "Bemis-Murcko scaffold; acyclic compounds sampled individually",
                    "bootstrap_repeats": len(values),
                    "decision_threshold": 0.5,
                }
            )
    pd.DataFrame(ci_rows).to_csv(
        REPORTS / "deployed_external_bootstrap_ci.csv", index=False
    )

    comparisons: list[dict[str, object]] = []
    metric_names = ["balanced_accuracy", "auprc"]
    for first, second in combinations(MODELS, 2):
        first_score, first_pred = model_arrays[first]
        second_score, second_pred = model_arrays[second]
        first_correct = first_pred == y
        second_correct = second_pred == y
        first_only = int(np.sum(first_correct & ~second_correct))
        second_only = int(np.sum(~first_correct & second_correct))
        discordant = first_only + second_only
        mcnemar_p = (
            binomtest(min(first_only, second_only), discordant, 0.5).pvalue
            if discordant
            else 1.0
        )
        for metric_name in metric_names:
            observed_difference = (
                metrics(y, first_score, first_pred)[metric_name]
                - metrics(y, second_score, second_pred)[metric_name]
            )
            differences = []
            for index in bootstrap_indices:
                if len(np.unique(y[index])) < 2:
                    continue
                first_value = metrics(y[index], first_score[index], first_pred[index])[metric_name]
                second_value = metrics(y[index], second_score[index], second_pred[index])[metric_name]
                differences.append(first_value - second_value)
            differences_array = np.asarray(differences, dtype=float)
            raw_p = 2 * min(
                np.mean(differences_array <= 0), np.mean(differences_array >= 0)
            )
            comparisons.append(
                {
                    "first_configuration": first,
                    "second_configuration": second,
                    "metric": metric_name,
                    "observed_difference_first_minus_second": observed_difference,
                    "difference_ci95_low": np.percentile(differences_array, 2.5),
                    "difference_ci95_high": np.percentile(differences_array, 97.5),
                    "cluster_bootstrap_two_sided_p_value_raw": min(1.0, raw_p),
                    "mcnemar_first_only_correct": first_only,
                    "mcnemar_second_only_correct": second_only,
                    "mcnemar_exact_two_sided_p_value": mcnemar_p,
                    "bootstrap_repeats": len(differences_array),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame["cluster_bootstrap_p_value_holm"] = holm_adjust(
        comparison_frame["cluster_bootstrap_two_sided_p_value_raw"].tolist()
    )
    comparison_frame["mcnemar_p_value_holm"] = holm_adjust(
        comparison_frame["mcnemar_exact_two_sided_p_value"].tolist()
    )
    comparison_frame.to_csv(
        REPORTS / "deployed_external_paired_comparisons.csv", index=False
    )

    print(f"External molecules: {len(frame)}")
    print(f"Scaffold bootstrap units: {len(groups)}")
    print("Wrote deployed-model external uncertainty and paired-comparison tables.")


if __name__ == "__main__":
    main()
