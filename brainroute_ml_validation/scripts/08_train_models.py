#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.modeling import train_models
from brainroute_ml_validation.src.utils import ensure_dirs, load_config, script_arg_parser, set_global_seed


def main() -> None:
    args = script_arg_parser("Train strict-validation models.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    train_models(cfg)


if __name__ == "__main__":
    main()
