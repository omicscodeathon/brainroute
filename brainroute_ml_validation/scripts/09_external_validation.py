#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.chemistry import max_tanimoto_to_train, morgan_bitvect, standardize_smiles
from brainroute_ml_validation.src.modeling import metric_dict, predict_scores
from brainroute_ml_validation.src.utils import LOGGER, ensure_dirs, load_config, normalize_binary_label, project_path, read_table, resolve_path, script_arg_parser, set_global_seed, write_csv


def main() -> None:
    args = script_arg_parser("Evaluate selected trained model on an optional external set.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    ext_path = resolve_path(cfg.get("paths", {}).get("external_validation_path"), cfg)
    if ext_path is None or not ext_path.exists():
        LOGGER.info("No external validation file configured; skipping.")
        return

    import joblib
    from brainroute_ml_validation.src.chemistry import calculate_morgan_matrix

    ext = read_table(ext_path)
    cols = cfg.get("columns", {})
    smiles_col, label_col = cols.get("smiles", "smiles"), cols.get("label", "BBB")
    rows = []
    for i, row in ext.iterrows():
        std = standardize_smiles(row.get(smiles_col))
        if std.valid:
            rows.append({"external_row": i, "canonical_smiles": std.canonical_smiles, "inchikey": std.inchikey, "label": normalize_binary_label(row.get(label_col))})
    ext_std = pd.DataFrame(rows).dropna(subset=["label"])
    internal = read_table(project_path(cfg, "data/processed/standardized_molecules.csv"))
    overlaps = ext_std[ext_std["inchikey"].isin(set(internal["inchikey"]))].copy()
    write_csv(overlaps, project_path(cfg, "reports/external_validation_overlap_removed.csv"), cfg)
    ext_std = ext_std[~ext_std["inchikey"].isin(set(internal["inchikey"]))].reset_index(drop=True)

    m = cfg.get("morgan", {})
    train_fps = [morgan_bitvect(s, int(m.get("radius", 2)), int(m.get("nBits", 2048)), bool(m.get("useChirality", True))) for s in internal["canonical_smiles"]]
    test_fps = [morgan_bitvect(s, int(m.get("radius", 2)), int(m.get("nBits", 2048)), bool(m.get("useChirality", True))) for s in ext_std["canonical_smiles"]]
    sim = max_tanimoto_to_train(test_fps, [fp for fp in train_fps if fp is not None])
    ext_std["max_tanimoto_to_training"] = [r["max_tanimoto"] for r in sim]
    write_csv(ext_std, project_path(cfg, "reports/external_near_duplicate_analysis.csv"), cfg)

    perf = pd.read_csv(project_path(cfg, "reports/model_performance_all_splits.csv"))
    primary = perf[perf["split"].str.startswith("scaffold_cv_fold", na=False)]
    if primary.empty:
        LOGGER.info("No scaffold-CV models found; external validation skipped.")
        return
    top = primary.groupby(["feature_view", "model"], as_index=False)["balanced_accuracy"].mean().sort_values("balanced_accuracy", ascending=False).iloc[0]
    if top["feature_view"] != "morgan":
        LOGGER.info("External validation currently auto-runs only Morgan final models; top view was %s. Skipping.", top["feature_view"])
        return
    model_files = sorted(project_path(cfg, "models").glob(f"morgan__{top['model']}__scaffold_cv_fold*.joblib"))
    if not model_files:
        LOGGER.info("No trained Morgan scaffold model artifact found; skipping.")
        return
    model = joblib.load(model_files[0])
    X_ext, valid_positions, _ = calculate_morgan_matrix(ext_std["canonical_smiles"], int(m.get("radius", 2)), int(m.get("nBits", 2048)), bool(m.get("useChirality", True)))
    X_ext = pd.DataFrame(X_ext, columns=[f"morgan_{i}" for i in range(int(m.get("nBits", 2048)))])
    y = ext_std.iloc[valid_positions]["label"].astype(int).to_numpy()
    pred = model.predict(X_ext)
    score = predict_scores(model, X_ext)
    metrics = metric_dict(y, pred, score)
    metrics.update({"feature_view": "morgan", "model": top["model"], "model_artifact": model_files[0].name})
    write_csv(pd.DataFrame([metrics]), project_path(cfg, "reports/external_validation_metrics.csv"), cfg)


if __name__ == "__main__":
    main()
