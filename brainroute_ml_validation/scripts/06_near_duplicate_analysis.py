#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.chemistry import max_tanimoto_to_train, morgan_bitvect
from brainroute_ml_validation.src.utils import ensure_dirs, load_config, project_path, read_table, script_arg_parser, set_global_seed, write_csv


def analyze_pair(train: pd.DataFrame, test: pd.DataFrame, cfg: dict, split_name: str) -> pd.DataFrame:
    m = cfg.get("morgan", {})
    train_fps = [morgan_bitvect(s, int(m.get("radius", 2)), int(m.get("nBits", 2048)), bool(m.get("useChirality", True))) for s in train["canonical_smiles"]]
    test_fps = [morgan_bitvect(s, int(m.get("radius", 2)), int(m.get("nBits", 2048)), bool(m.get("useChirality", True))) for s in test["canonical_smiles"]]
    train_fps = [fp for fp in train_fps if fp is not None]
    rows = max_tanimoto_to_train(test_fps, train_fps)
    out = test.reset_index(drop=True).copy()
    out["split"] = split_name
    out["max_tanimoto"] = [r["max_tanimoto"] for r in rows]
    nearest = [r["nearest_train_position"] for r in rows]
    out["nearest_train_inchikey"] = [train.iloc[i]["inchikey"] if i is not None else None for i in nearest]
    out["nearest_train_smiles"] = [train.iloc[i]["canonical_smiles"] if i is not None else None for i in nearest]
    bins = [0, 0.40, 0.60, 0.80, 1.01]
    labels = ["<0.40", "0.40_to_0.60", "0.60_to_0.80", ">0.80"]
    out["similarity_bin"] = pd.cut(out["max_tanimoto"], bins=bins, labels=labels, include_lowest=True, right=False)
    return out


def main() -> None:
    args = script_arg_parser("Analyze max train-test Morgan Tanimoto similarity for saved splits.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    split_root = project_path(cfg, "data/splits")
    detail_frames = []
    summary_rows = []
    for test_path in sorted(split_root.glob("*_test.csv")):
        prefix = test_path.name.replace("_test.csv", "")
        train_path = split_root / f"{prefix}_train.csv"
        if not train_path.exists():
            continue
        out = analyze_pair(pd.read_csv(train_path), pd.read_csv(test_path), cfg, prefix)
        detail_frames.append(out)
        thresholds = cfg.get("validation", {}).get("near_duplicate_tanimoto_thresholds", [0.8, 0.85, 0.9])
        row = {
            "split": prefix,
            "mean_max_tanimoto": out["max_tanimoto"].mean(),
            "median_max_tanimoto": out["max_tanimoto"].median(),
        }
        for t in thresholds:
            row[f"pct_gt_{t}"] = 100 * float((out["max_tanimoto"] > float(t)).mean())
        summary_rows.append(row)
        kind = "scaffold_split" if prefix.startswith("scaffold") else "duplicate_aware" if prefix.startswith("duplicate") else "random80" if prefix.startswith("random") else prefix
        write_csv(out, project_path(cfg, f"reports/near_duplicate_analysis_{kind}.csv"), cfg)
    if detail_frames:
        write_csv(pd.concat(detail_frames, ignore_index=True), project_path(cfg, "reports/near_duplicate_analysis_all_splits.csv"), cfg)
        write_csv(pd.DataFrame(summary_rows), project_path(cfg, "reports/near_duplicate_similarity_summary.csv"), cfg)


if __name__ == "__main__":
    main()
