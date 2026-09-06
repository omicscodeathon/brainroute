#!/usr/bin/env python3
"""Create a matched benchmark workspace that excludes source-label conflicts.

The original representation artifacts remain unchanged.  This script copies only
the rows eligible for the final, source-audited benchmark into a separate external
workspace so every compared feature view uses the same molecules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = ROOT / "brainroute_ml_validation"
SOURCE_WORKSPACE = VALIDATION_ROOT / "data/benchmarks/matched_3d_qm"
TARGET_WORKSPACE = VALIDATION_ROOT / "data/benchmarks/matched_3d_qm_conflict_excluded"
PROVENANCE = VALIDATION_ROOT / "data/processed/standardized_molecules_with_provenance.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-workspace", type=Path, default=SOURCE_WORKSPACE)
    parser.add_argument("--target-workspace", type=Path, default=TARGET_WORKSPACE)
    args = parser.parse_args()
    source_workspace = args.source_workspace.expanduser().resolve()
    target_workspace = args.target_workspace.expanduser().resolve()
    source_artifacts = source_workspace / "artifacts"
    target_artifacts = target_workspace / "artifacts"
    target_artifacts.mkdir(parents=True, exist_ok=True)

    provenance = pd.read_csv(PROVENANCE)
    conflict_ids = set(
        "internal:"
        + provenance.loc[
            provenance["provenance_label_conflict"].astype(bool), "molecule_id"
        ].astype(str)
    )

    manifest = pd.read_csv(source_artifacts / "benchmark_manifest.csv")
    keep_manifest = ~manifest["benchmark_id"].isin(conflict_ids)
    filtered_manifest = manifest.loc[keep_manifest].reset_index(drop=True)
    filtered_manifest.to_csv(target_artifacts / "benchmark_manifest.csv", index=False)

    xtb = pd.read_csv(source_artifacts / "xtb_features.csv")
    filtered_xtb = xtb.loc[~xtb["benchmark_id"].isin(conflict_ids)].reset_index(drop=True)
    filtered_xtb.to_csv(target_artifacts / "xtb_features.csv", index=False)

    unimol_index = pd.read_csv(source_artifacts / "unimol_v1_index.csv")
    unimol = np.load(source_artifacts / "unimol_v1_representations.npy", mmap_mode="r")
    if len(unimol_index) != len(unimol):
        raise ValueError("Uni-Mol index and representation array lengths differ")
    keep_unimol = ~unimol_index["benchmark_id"].isin(conflict_ids)
    filtered_index = unimol_index.loc[keep_unimol].reset_index(drop=True)
    filtered_array = np.asarray(unimol[keep_unimol.to_numpy()], dtype=np.float32)
    filtered_index.to_csv(target_artifacts / "unimol_v1_index.csv", index=False)
    np.save(target_artifacts / "unimol_v1_representations.npy", filtered_array)

    audit = {
        "source_workspace": str(source_workspace),
        "target_workspace": str(target_workspace),
        "source_provenance_table": str(PROVENANCE),
        "source_conflicting_internal_molecules_excluded": len(conflict_ids),
        "manifest_rows_before": len(manifest),
        "manifest_rows_after": len(filtered_manifest),
        "xtb_rows_before": len(xtb),
        "xtb_rows_after": len(filtered_xtb),
        "unimol_rows_before": len(unimol_index),
        "unimol_rows_after": len(filtered_index),
        "output_sha256": {
            "benchmark_manifest": sha256(target_artifacts / "benchmark_manifest.csv"),
            "xtb_features": sha256(target_artifacts / "xtb_features.csv"),
            "unimol_index": sha256(target_artifacts / "unimol_v1_index.csv"),
            "unimol_representations": sha256(
                target_artifacts / "unimol_v1_representations.npy"
            ),
        },
    }
    with (target_artifacts / "conflict_exclusion_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(audit, handle, indent=2)
    pd.DataFrame(
        {
            "benchmark_id": sorted(conflict_ids),
            "reason": "conflicting BBB labels across reconstructed source records",
        }
    ).to_csv(target_artifacts / "excluded_source_label_conflicts.csv", index=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
