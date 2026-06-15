#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.utils import ensure_dirs, load_config, project_path, script_arg_parser, set_global_seed, write_csv


def ci95(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    mean = arr.mean()
    half = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan
    return mean - half, mean + half


def main() -> None:
    args = script_arg_parser("Compare scaffold-CV model metrics statistically.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    perf_path = project_path(cfg, "reports/model_performance_all_splits.csv")
    if not perf_path.exists():
        return
    perf = pd.read_csv(perf_path)
    primary = perf[perf["split"].str.startswith("scaffold_cv_fold", na=False)].copy()
    rows = []
    for metric in ["balanced_accuracy", "auprc"]:
        summary = primary.groupby(["feature_view", "model"])[metric].agg(["mean", "std", "count"]).reset_index()
        summary["ci95_low"], summary["ci95_high"] = zip(*summary.apply(lambda r: ci95(primary[(primary.feature_view == r.feature_view) & (primary.model == r.model)][metric]), axis=1))
        for _, r in summary.iterrows():
            rows.append({"comparison": "summary", "metric": metric, **r.to_dict()})
        ranked = summary.sort_values("mean", ascending=False).head(3)
        if len(ranked) >= 2:
            best = ranked.iloc[0]
            for _, other in ranked.iloc[1:].iterrows():
                a = primary[(primary.feature_view == best.feature_view) & (primary.model == best.model)].sort_values("split")[metric].to_numpy()
                b = primary[(primary.feature_view == other.feature_view) & (primary.model == other.model)].sort_values("split")[metric].to_numpy()
                p = wilcoxon(a, b).pvalue if len(a) == len(b) and len(a) > 1 and not np.allclose(a, b) else np.nan
                rows.append({"comparison": f"{best.feature_view}/{best.model} vs {other.feature_view}/{other.model}", "metric": metric, "p_value_wilcoxon": p})
    write_csv(pd.DataFrame(rows), project_path(cfg, "reports/model_statistical_comparison.csv"), cfg)


if __name__ == "__main__":
    main()
