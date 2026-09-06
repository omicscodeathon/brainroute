#!/usr/bin/env python3
"""Verify the saved matched benchmark before manuscript generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
WORKSPACE = ROOT / "data/benchmarks/matched_3d_qm_conflict_excluded/artifacts"
CONFIGURATIONS = {
    "padel_morgan_control",
    "padel_morgan_plus_gfn2_xtb",
    "unimol_v1_frozen_representation",
}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    summary = pd.read_csv(REPORTS / "matched_benchmark_cohort_summary.csv")
    internal = pd.read_csv(REPORTS / "matched_benchmark_internal_oof_predictions.csv")
    external = pd.read_csv(REPORTS / "matched_benchmark_external_predictions.csv")
    performance = pd.read_csv(REPORTS / "matched_benchmark_performance_summary.csv")
    tuning = pd.read_csv(REPORTS / "matched_benchmark_nested_tuning.csv")
    split = pd.read_csv(REPORTS / "matched_benchmark_scaffold_split_audit.csv")
    source = pd.read_csv(REPORTS / "source_validation_metrics.csv")
    excluded = pd.read_csv(workspace / "excluded_source_label_conflicts.csv")

    internal_n = int(summary.loc[summary["cohort"] == "internal", "n"].iloc[0])
    external_n = int(summary.loc[summary["cohort"] == "external", "n"].iloc[0])
    check(set(internal["configuration"]) == CONFIGURATIONS, "Unexpected internal configurations")
    check(set(external["configuration"]) == CONFIGURATIONS, "Unexpected external configurations")
    for configuration in CONFIGURATIONS:
        internal_view = internal[internal["configuration"] == configuration]
        external_view = external[external["configuration"] == configuration]
        check(len(internal_view) == internal_n, f"Internal row mismatch for {configuration}")
        check(len(external_view) == external_n, f"External row mismatch for {configuration}")
        check(internal_view["benchmark_id"].nunique() == internal_n, "Internal IDs are not one-to-one")
        check(external_view["benchmark_id"].nunique() == external_n, "External IDs are not one-to-one")

    check((split["inchikey_overlap"] == 0).all(), "InChIKey overlap found in scaffold split")
    check((split["scaffold_overlap"] == 0).all(), "Scaffold overlap found in scaffold split")
    check(len(tuning) == 15, "Expected three configurations by five outer folds")
    check(len(source) == 6, "Expected three configurations by two source directions")
    check(len(excluded) == 24, "Expected 24 retained source-label conflicts to be excluded")
    expected_performance = 3 * 2
    check(len(performance) == expected_performance, "Unexpected performance summary row count")
    required_metrics = [
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "mcc",
        "roc_auc",
        "auprc",
        "brier_score",
        "expected_calibration_error",
    ]
    check(not performance[required_metrics].isna().any().any(), "Missing primary performance metric")

    audit = {
        "status": "passed",
        "internal_matched_molecules": internal_n,
        "external_matched_molecules": external_n,
        "configurations": sorted(CONFIGURATIONS),
        "outer_scaffold_folds": len(split),
        "nested_tuning_rows": len(tuning),
        "source_validation_rows": len(source),
        "source_label_conflicts_excluded": len(excluded),
        "inchikey_overlap_total": int(split["inchikey_overlap"].sum()),
        "scaffold_overlap_total": int(split["scaffold_overlap"].sum()),
    }
    with (REPORTS / "matched_benchmark_verification.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    with (workspace / "matched_benchmark_verification.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
