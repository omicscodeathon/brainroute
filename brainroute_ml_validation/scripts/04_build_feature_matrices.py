#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.features import build_feature_matrices
from brainroute_ml_validation.src.utils import ensure_dirs, load_config, script_arg_parser, set_global_seed


def main() -> None:
    args = script_arg_parser("Build PaDEL/Morgan/embedding feature matrices.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    build_feature_matrices(cfg)


if __name__ == "__main__":
    main()
