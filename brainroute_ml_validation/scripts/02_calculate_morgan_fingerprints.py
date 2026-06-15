#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.chemistry import calculate_morgan_matrix
from brainroute_ml_validation.src.utils import ensure_dirs, load_config, project_path, read_table, script_arg_parser, set_global_seed, write_csv


def main() -> None:
    args = script_arg_parser("Calculate Morgan fingerprints.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    df = read_table(project_path(cfg, "data/processed/standardized_molecules.csv"))
    morgan = cfg.get("morgan", {})
    n_bits = int(morgan.get("nBits", 2048))
    matrix, valid_positions, _ = calculate_morgan_matrix(
        df["canonical_smiles"],
        radius=int(morgan.get("radius", 2)),
        n_bits=n_bits,
        use_chirality=bool(morgan.get("useChirality", True)),
    )
    base = project_path(cfg, "data/processed")
    npz_path = base / "morgan_fingerprints.npz"
    if not npz_path.exists() or cfg.get("overwrite", False):
        sparse.save_npz(npz_path, sparse.csr_matrix(matrix))
    write_csv(pd.DataFrame({"feature": [f"morgan_{i}" for i in range(n_bits)]}), base / "morgan_fingerprints_columns.csv", cfg)
    write_csv(df.iloc[valid_positions].reset_index(drop=True), base / "fingerprint_index.csv", cfg)


if __name__ == "__main__":
    main()
