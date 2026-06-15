#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.chemistry import standardize_smiles
from brainroute_ml_validation.src.utils import (
    LOGGER,
    ensure_dirs,
    load_config,
    normalize_binary_label,
    project_path,
    read_table,
    resolve_path,
    script_arg_parser,
    set_global_seed,
    write_csv,
)


def main() -> None:
    args = script_arg_parser("Standardize SMILES and audit duplicates/conflicting labels.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)

    input_path = resolve_path(cfg["paths"]["input_data"], cfg)
    df = read_table(input_path)
    cols = cfg.get("columns", {})
    smiles_col = cols.get("smiles", "smiles")
    label_col = cols.get("label", "BBB")
    source_col = cols.get("source_dataset")
    name_col = cols.get("molecule_name")
    LOGGER.info("Loaded %s rows from %s", len(df), input_path)

    records, invalid = [], []
    for idx, row in df.iterrows():
        std = standardize_smiles(row.get(smiles_col))
        base = row.to_dict()
        base["input_row"] = idx
        if not std.valid:
            base["standardization_error"] = std.error
            invalid.append(base)
            continue
        records.append(
            {
                "input_row": idx,
                "molecule_id": f"mol_{len(records):06d}",
                "name": row.get(name_col) if name_col in df.columns else None,
                "input_smiles": row.get(smiles_col),
                "canonical_smiles": std.canonical_smiles,
                "inchikey": std.inchikey,
                "murcko_scaffold": std.murcko_scaffold,
                "label": normalize_binary_label(row.get(label_col)),
                "source_dataset": row.get(source_col) if source_col in df.columns else "unspecified",
            }
        )

    valid = pd.DataFrame(records)
    invalid_df = pd.DataFrame(invalid)
    valid = valid.dropna(subset=["label", "inchikey"]).copy()
    valid["label"] = valid["label"].astype(int)

    duplicate_audit = (
        valid.groupby("inchikey")
        .agg(
            n_entries=("inchikey", "size"),
            n_labels=("label", "nunique"),
            labels=("label", lambda x: ";".join(map(str, sorted(set(x))))),
            sources=("source_dataset", lambda x: ";".join(map(str, sorted(set(x))))),
            canonical_smiles=("canonical_smiles", "first"),
        )
        .reset_index()
    )
    duplicate_audit["is_duplicate"] = duplicate_audit["n_entries"] > 1
    conflicting = duplicate_audit[duplicate_audit["n_labels"] > 1].copy()

    handling = cfg.get("duplicate_handling", {}).get("conflicting_labels", "exclude")
    if handling == "majority":
        resolved = (
            valid.groupby("inchikey", as_index=False)
            .agg(
                molecule_id=("molecule_id", "first"),
                name=("name", "first"),
                input_smiles=("input_smiles", "first"),
                canonical_smiles=("canonical_smiles", "first"),
                murcko_scaffold=("murcko_scaffold", "first"),
                label=("label", lambda x: int(x.value_counts().idxmax())),
                source_dataset=("source_dataset", lambda x: ";".join(map(str, sorted(set(x))))),
            )
        )
    else:
        conflict_keys = set(conflicting["inchikey"])
        resolved = valid[~valid["inchikey"].isin(conflict_keys)].drop_duplicates("inchikey", keep="first")

    accounting = pd.DataFrame(
        [
            {"stage": "starting_molecules", "count": len(df)},
            {"stage": "invalid_smiles_removed", "count": len(invalid_df)},
            {"stage": "descriptor_calculation_failures_known", "count": 0},
            {"stage": "duplicate_molecules_found", "count": int(duplicate_audit["is_duplicate"].sum())},
            {"stage": "conflicting_label_molecules_removed", "count": int(conflicting["n_entries"].sum() if handling == "exclude" else 0)},
            {"stage": "final_unique_molecules_available_for_modeling", "count": len(resolved)},
        ]
    )

    write_csv(invalid_df, project_path(cfg, "reports/invalid_smiles.csv"), cfg)
    write_csv(duplicate_audit, project_path(cfg, "reports/duplicate_audit.csv"), cfg)
    write_csv(conflicting, project_path(cfg, "reports/conflicting_labels.csv"), cfg)
    write_csv(accounting, project_path(cfg, "reports/data_accounting.csv"), cfg)
    write_csv(resolved.reset_index(drop=True), project_path(cfg, "data/processed/standardized_molecules.csv"), cfg)


if __name__ == "__main__":
    main()
