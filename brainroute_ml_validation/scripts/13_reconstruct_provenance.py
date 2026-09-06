#!/usr/bin/env python3
"""Reconstruct source provenance from the available B3DB and MoleculeNet files.

This step does not retrain or modify any model. The original retrieval date was
not recorded and is reported as unknown rather than inferred.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.chemistry import standardize_smiles
from brainroute_ml_validation.src.utils import (
    ensure_dirs,
    load_config,
    normalize_binary_label,
    project_path,
    read_table,
    resolve_path,
    script_arg_parser,
    write_csv,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize_source(
    frame: pd.DataFrame,
    *,
    dataset: str,
    smiles_col: str,
    label_col: str,
    name_col: str,
    reference_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    for source_row, record in frame.iterrows():
        result = standardize_smiles(record.get(smiles_col))
        rows.append(
            {
                "source_dataset": dataset,
                "source_row_zero_based": int(source_row),
                "source_name": record.get(name_col),
                "source_smiles": record.get(smiles_col),
                "source_label_raw": record.get(label_col),
                "source_label": normalize_binary_label(record.get(label_col)),
                "source_reference": record.get(reference_col) if reference_col else None,
                "canonical_smiles": result.canonical_smiles if result.valid else None,
                "inchikey": result.inchikey if result.valid else None,
                "standardization_valid": bool(result.valid),
                "standardization_error": result.error,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = script_arg_parser("Reconstruct B3DB and MoleculeNet source provenance.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing provenance reports.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.overwrite:
        cfg["overwrite"] = True
    ensure_dirs(cfg)

    paths = cfg.get("paths", {})
    b3db_path = resolve_path(paths.get("b3db_source_data"), cfg)
    moleculenet_path = resolve_path(paths.get("moleculenet_bbbp_source_data"), cfg)
    archived_padel_path = resolve_path(paths.get("input_data"), cfg)
    if b3db_path is None or moleculenet_path is None:
        raise FileNotFoundError("Both source snapshot paths must be configured.")

    b3db_raw = read_table(b3db_path)
    moleculenet_raw = read_table(moleculenet_path)
    b3db = standardize_source(
        b3db_raw,
        dataset="B3DB",
        smiles_col="SMILES",
        label_col="BBB+/BBB-",
        name_col="compound_name",
        reference_col="reference",
    )
    moleculenet = standardize_source(
        moleculenet_raw,
        dataset="MoleculeNet BBBP",
        smiles_col="smiles",
        label_col="p_np",
        name_col="name",
    )
    source_records = pd.concat([b3db, moleculenet], ignore_index=True)

    standardized = pd.read_csv(project_path(cfg, "data/processed/standardized_molecules.csv"))
    valid_sources = source_records.dropna(subset=["inchikey", "source_label"]).copy()
    valid_sources["source_label"] = valid_sources["source_label"].astype(int)
    provenance_rows = []
    for inchikey, group in valid_sources.groupby("inchikey", sort=False):
        provenance_rows.append(
            {
                "inchikey": inchikey,
                "provenance_sources": ";".join(sorted(set(group["source_dataset"].astype(str)))),
                "provenance_source_rows": ";".join(
                    f"{record.source_dataset}:{int(record.source_row_zero_based)}"
                    for record in group.itertuples()
                ),
                "provenance_labels": ";".join(map(str, sorted(set(group["source_label"])))),
                "provenance_record_count": len(group),
                "provenance_source_count": group["source_dataset"].nunique(),
                "provenance_label_conflict": group["source_label"].nunique() > 1,
            }
        )
    grouped = pd.DataFrame(provenance_rows)

    with_provenance = standardized.merge(grouped, on="inchikey", how="left")
    with_provenance["source_dataset"] = with_provenance["provenance_sources"].fillna("unmatched")
    with_provenance["provenance_reconstructed"] = with_provenance["provenance_sources"].notna()
    unmatched = with_provenance[~with_provenance["provenance_reconstructed"]].copy()

    manifest_rows = [
        {
            "source_dataset": "B3DB",
            "local_source_path": str(b3db_path),
            "sha256": sha256(b3db_path),
            "row_count": len(b3db_raw),
            "bbb_positive_count": int((b3db["source_label"] == 1).sum()),
            "bbb_negative_count": int((b3db["source_label"] == 0).sum()),
            "public_source": "https://github.com/theochem/B3DB",
            "source_publication_doi": "10.1038/s41597-021-01069-5",
            "retrieval_date": "not documented in the original workflow",
        },
        {
            "source_dataset": "MoleculeNet BBBP",
            "local_source_path": str(moleculenet_path),
            "sha256": sha256(moleculenet_path),
            "row_count": len(moleculenet_raw),
            "bbb_positive_count": int((moleculenet["source_label"] == 1).sum()),
            "bbb_negative_count": int((moleculenet["source_label"] == 0).sum()),
            "public_source": "https://moleculenet.org/datasets-1",
            "source_publication_doi": "10.1039/C7SC02664A",
            "retrieval_date": "not documented in the original workflow",
        },
    ]
    if archived_padel_path is not None and archived_padel_path.exists():
        archived_padel = read_table(archived_padel_path)
        manifest_rows.append(
            {
                "source_dataset": "Merged PaDEL modeling input",
                "local_source_path": str(archived_padel_path),
                "sha256": sha256(archived_padel_path),
                "row_count": len(archived_padel),
                "bbb_positive_count": int((archived_padel["BBB"] == 1).sum()),
                "bbb_negative_count": int((archived_padel["BBB"] == 0).sum()),
                "public_source": "derived local artifact",
                "source_publication_doi": "not applicable",
                "retrieval_date": "not documented in the original workflow",
            }
        )

    summary_rows = []
    for dataset, frame in source_records.groupby("source_dataset"):
        keys = set(frame.loc[frame["standardization_valid"], "inchikey"].dropna())
        matched = with_provenance[with_provenance["inchikey"].isin(keys)]
        summary_rows.append(
            {
                "source_dataset": dataset,
                "raw_rows": len(frame),
                "valid_standardized_rows": int(frame["standardization_valid"].sum()),
                "unique_standardized_inchikeys": int(frame["inchikey"].nunique()),
                "final_modeling_molecules_linked": len(matched),
                "final_bbb_positive": int((matched["label"] == 1).sum()),
                "final_bbb_negative": int((matched["label"] == 0).sum()),
            }
        )
    summary_rows.append(
        {
            "source_dataset": "All reconstructed sources",
            "raw_rows": len(source_records),
            "valid_standardized_rows": int(source_records["standardization_valid"].sum()),
            "unique_standardized_inchikeys": int(source_records["inchikey"].nunique()),
            "final_modeling_molecules_linked": int(with_provenance["provenance_reconstructed"].sum()),
            "final_bbb_positive": int((with_provenance["label"] == 1).sum()),
            "final_bbb_negative": int((with_provenance["label"] == 0).sum()),
        }
    )

    conflict_audit = valid_sources.groupby("inchikey").filter(lambda x: x["source_label"].nunique() > 1)
    b3db_quality = (
        b3db_raw.groupby(["group", "BBB+/BBB-"], dropna=False)
        .size()
        .reset_index(name="n")
        .rename(columns={"group": "B3DB_quality_group", "BBB+/BBB-": "source_label_raw"})
    )
    b3db_quality["quality_group_definition"] = b3db_quality["B3DB_quality_group"].map(
        {
            "A": "A numerical logBB value is available; B3DB applies a logBB threshold of -1 for the categorical label.",
            "B": "Contributing sources use a logBB threshold of -1 and agree on the categorical label.",
            "C": "Contributing sources agree on the categorical label but do not report a threshold.",
            "D": "Contributing categorical reports conflict; B3DB retains the most prevalent label after discarding equal-frequency ties.",
        }
    )
    metadata_rows = []
    for dataset, raw, field_map in [
        (
            "B3DB",
            b3db_raw,
            {
                "structure": "SMILES",
                "label": "BBB+/BBB-",
                "compound_name": "compound_name",
                "numerical_logBB": "logBB",
                "classification_threshold": "threshold",
                "source_reference_code": "reference",
                "quality_group": "group",
            },
        ),
        (
            "MoleculeNet BBBP",
            moleculenet_raw,
            {"structure": "smiles", "label": "p_np", "compound_name": "name"},
        ),
    ]:
        for field_role, column in field_map.items():
            metadata_rows.append(
                {
                    "source_dataset": dataset,
                    "field_role": field_role,
                    "source_column": column,
                    "row_count": len(raw),
                    "nonmissing_count": int(raw[column].notna().sum()),
                    "missing_count": int(raw[column].isna().sum()),
                }
            )
    write_csv(source_records, project_path(cfg, "reports/source_provenance_records.csv"), cfg)
    write_csv(pd.DataFrame(manifest_rows), project_path(cfg, "reports/source_provenance_manifest.csv"), cfg)
    write_csv(pd.DataFrame(summary_rows), project_path(cfg, "reports/source_provenance_summary.csv"), cfg)
    write_csv(conflict_audit, project_path(cfg, "reports/source_label_conflict_audit.csv"), cfg)
    write_csv(unmatched, project_path(cfg, "reports/source_provenance_unmatched.csv"), cfg)
    write_csv(with_provenance, project_path(cfg, "data/processed/standardized_molecules_with_provenance.csv"), cfg)
    write_csv(b3db_quality, project_path(cfg, "reports/source_b3db_quality_group_distribution.csv"), cfg)
    write_csv(pd.DataFrame(metadata_rows), project_path(cfg, "reports/source_metadata_completeness.csv"), cfg)


if __name__ == "__main__":
    main()
