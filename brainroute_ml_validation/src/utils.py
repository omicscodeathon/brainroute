from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def setup_logger(name: str = "brainroute_validation") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


LOGGER = setup_logger()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else PACKAGE_ROOT / "configs" / "validation_config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(path.resolve())
    cfg["_config_dir"] = str(path.resolve().parent)
    return cfg


def resolve_path(path_value: str | Path | None, cfg: dict[str, Any] | None = None) -> Path | None:
    if path_value in (None, "", "null"):
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p
    base = Path(cfg.get("_config_dir", PACKAGE_ROOT / "configs")) if cfg else PACKAGE_ROOT
    return (base / p).resolve()


def output_root(cfg: dict[str, Any]) -> Path:
    root = resolve_path(cfg.get("paths", {}).get("output_dir", ".."), cfg)
    return root if root is not None else PACKAGE_ROOT


def project_path(cfg: dict[str, Any], *parts: str) -> Path:
    return output_root(cfg).joinpath(*parts)


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for sub in [
        "data/processed",
        "data/splits",
        "data/external",
        "reports/figures",
        "models",
    ]:
        project_path(cfg, sub).mkdir(parents=True, exist_ok=True)


def check_overwrite(path: Path, cfg: dict[str, Any]) -> None:
    if path.exists() and not cfg.get("overwrite", False):
        LOGGER.info("Keeping existing file because overwrite=false: %s", path)
        raise FileExistsError(path)


def write_csv(df, path: Path, cfg: dict[str, Any], index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not cfg.get("overwrite", False):
        LOGGER.info("Skipping existing CSV: %s", path)
        return
    df.to_csv(path, index=index)
    LOGGER.info("Wrote %s", path)


def write_json(obj: Any, path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not cfg.get("overwrite", False):
        LOGGER.info("Skipping existing JSON: %s", path)
        return
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
    LOGGER.info("Wrote %s", path)


def read_table(path: str | Path):
    import pandas as pd

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def normalize_binary_label(value: Any) -> int | float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "bbb+", "bbb +", "positive", "pos", "yes", "true", "permeable"}:
            return 1
        if v in {"0", "bbb-", "bbb -", "negative", "neg", "no", "false", "impermeable"}:
            return 0
    try:
        return int(float(value))
    except Exception:
        return np.nan


def existing_file(path: Path | None) -> bool:
    return path is not None and path.exists() and path.is_file()


def script_arg_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "validation_config.yaml"))
    return parser
