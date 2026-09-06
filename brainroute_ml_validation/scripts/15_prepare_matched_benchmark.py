#!/usr/bin/env python3
"""Prepare the locked, molecule-matched cohort for the Uni-Mol/xTB benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "brainroute_ml_validation/configs/3d_qm_benchmark.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / "brainroute_ml_validation" / candidate


def calculation_structure(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("RDKit could not parse the standardized SMILES")
    fragment = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
    Chem.SanitizeMol(fragment)
    canonical = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
    charge = int(sum(atom.GetFormalCharge() for atom in fragment.GetAtoms()))
    electron_count = int(sum(atom.GetAtomicNum() for atom in fragment.GetAtoms()) - charge)
    unpaired = int(electron_count % 2)
    return {
        "calculation_smiles": canonical,
        "calculation_formal_charge": charge,
        "calculation_unpaired_electrons": unpaired,
        "calculation_heavy_atoms": int(fragment.GetNumHeavyAtoms()),
        "calculation_molecular_weight": float(Descriptors.MolWt(fragment)),
        "calculation_fragment_changed": canonical != str(smiles),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    workspace = resolve(paths["external_workspace"])
    artifact_dir = workspace / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    index_path = resolve(paths["internal_index"])
    provenance_path = resolve(paths["internal_provenance"])
    external_path = resolve(paths["external_cohort"])
    internal = pd.read_csv(index_path)
    provenance = pd.read_csv(provenance_path)
    external = pd.read_csv(external_path)

    keep_provenance = [
        "molecule_id",
        "provenance_sources",
        "provenance_source_rows",
        "provenance_record_count",
        "provenance_source_count",
    ]
    internal = internal.merge(
        provenance[[column for column in keep_provenance if column in provenance.columns]],
        on="molecule_id",
        how="left",
        validate="one_to_one",
    )
    internal["cohort"] = "internal"
    internal["benchmark_id"] = "internal:" + internal["molecule_id"].astype(str)
    internal["external_row"] = pd.NA
    internal["compound_name"] = pd.NA

    external["cohort"] = "external"
    external["benchmark_id"] = "external:" + external["inchikey"].astype(str)
    external["molecule_id"] = pd.NA
    external["source_dataset"] = "independent_external_QSAR_set"
    for column in keep_provenance[1:]:
        external[column] = pd.NA

    columns = [
        "benchmark_id",
        "cohort",
        "molecule_id",
        "external_row",
        "compound_name",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "label",
        "source_dataset",
        "provenance_sources",
        "provenance_source_rows",
        "provenance_record_count",
        "provenance_source_count",
    ]
    manifest = pd.concat([internal[columns], external[columns]], ignore_index=True)
    if manifest["benchmark_id"].duplicated().any():
        raise ValueError("benchmark_id is not unique")
    if manifest["inchikey"].duplicated().any():
        duplicates = manifest.loc[manifest["inchikey"].duplicated(False), ["cohort", "inchikey"]]
        raise ValueError(f"Internal/external molecular overlap remains:\n{duplicates.head()}")

    calculated = []
    failures = []
    for position, row in manifest.iterrows():
        try:
            calculated.append({"manifest_position": position, **calculation_structure(row["canonical_smiles"])})
        except Exception as exc:
            failures.append({"manifest_position": position, "benchmark_id": row["benchmark_id"], "error": str(exc)})
    if failures:
        pd.DataFrame(failures).to_csv(artifact_dir / "benchmark_manifest_structure_failures.csv", index=False)
        raise ValueError(f"Calculation-structure preparation failed for {len(failures)} molecules")
    manifest = pd.concat([manifest, pd.DataFrame(calculated).drop(columns="manifest_position")], axis=1)
    manifest["murcko_scaffold"] = manifest["murcko_scaffold"].fillna("NO_SCAFFOLD")
    manifest.to_csv(artifact_dir / "benchmark_manifest.csv", index=False)

    summary = (
        manifest.groupby("cohort", as_index=False)
        .agg(
            n=("benchmark_id", "size"),
            bbb_positive=("label", "sum"),
            unique_inchikeys=("inchikey", "nunique"),
            unique_scaffolds=("murcko_scaffold", "nunique"),
            changed_to_largest_fragment=("calculation_fragment_changed", "sum"),
            median_heavy_atoms=("calculation_heavy_atoms", "median"),
            maximum_heavy_atoms=("calculation_heavy_atoms", "max"),
        )
    )
    summary["bbb_negative"] = summary["n"] - summary["bbb_positive"]
    summary.to_csv(artifact_dir / "benchmark_manifest_summary.csv", index=False)

    source_files = {
        "benchmark_configuration": args.config,
        "internal_feature_index": index_path,
        "internal_provenance_table": provenance_path,
        "internal_feature_matrix": resolve(paths["internal_features"]),
        "external_cohort": external_path,
        "external_padel_matrix": resolve(paths["external_padel"]),
        "external_padel_failed_positions": resolve(paths["external_padel_failed_positions"]),
    }
    records = []
    for role, path in source_files.items():
        records.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest_path = artifact_dir / "benchmark_manifest.csv"
    records.append(
        {
            "role": "locked_benchmark_manifest",
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        }
    )
    pd.DataFrame(records).to_csv(artifact_dir / "benchmark_file_manifest.csv", index=False)
    with (artifact_dir / "benchmark_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "configuration": cfg,
                "internal_rows": int((manifest["cohort"] == "internal").sum()),
                "external_rows": int((manifest["cohort"] == "external").sum()),
                "note": "The external cohort was previously standardized, deduplicated, screened against the internal set, and restricted to molecules with PaDEL descriptors.",
            },
            handle,
            indent=2,
        )
    print(summary.to_string(index=False))
    print(f"Wrote locked benchmark manifest to {manifest_path}")


if __name__ == "__main__":
    main()
