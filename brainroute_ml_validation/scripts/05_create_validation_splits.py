#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.splitting import create_splits
from brainroute_ml_validation.src.utils import ensure_dirs, load_config, project_path, read_table, script_arg_parser, set_global_seed


def main() -> None:
    args = script_arg_parser("Create fixed random, duplicate-aware, and scaffold-aware splits.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    std = read_table(project_path(cfg, "data/processed/standardized_molecules.csv"))
    create_splits(std, cfg)


if __name__ == "__main__":
    main()
