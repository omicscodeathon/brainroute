from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit

from .chemistry import standardize_smiles
from .utils import LOGGER, project_path, write_csv
from .utils import resolve_path


SPLIT_COLUMNS = ["molecule_id", "canonical_smiles", "inchikey", "murcko_scaffold", "label", "source_dataset"]


def split_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SPLIT_COLUMNS if c in df.columns]
    return df[cols].copy()


def write_split(df: pd.DataFrame, train_idx, test_idx, prefix: str, cfg: dict) -> None:
    root = project_path(cfg, "data/splits")
    write_csv(split_frame(df.iloc[train_idx]), root / f"{prefix}_train.csv", cfg)
    write_csv(split_frame(df.iloc[test_idx]), root / f"{prefix}_test.csv", cfg)


def create_splits(std: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    validation = cfg.get("validation", {})
    seed = int(validation.get("random_seed", cfg.get("random_seed", 42)))
    test_size = float(validation.get("test_size", 0.2))
    y = std["label"].astype(int).to_numpy()
    rows = []

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(std, y):
        write_split(std, train_idx, test_idx, f"random80_seed{seed}", cfg)
        rows.append({"split": f"random80_seed{seed}", "type": "baseline_random", "train_n": len(train_idx), "test_n": len(test_idx)})

    for split_seed in validation.get("duplicate_aware_repeated_seeds", [1, 2, 3, 4, 5]):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=int(split_seed))
        for train_idx, test_idx in splitter.split(std, y, groups=std["inchikey"]):
            write_split(std, train_idx, test_idx, f"duplicate_aware_seed{split_seed}", cfg)
            rows.append({"split": f"duplicate_aware_seed{split_seed}", "type": "duplicate_aware", "train_n": len(train_idx), "test_n": len(test_idx)})

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_idx, test_idx in splitter.split(std, y, groups=std["murcko_scaffold"]):
        write_split(std, train_idx, test_idx, f"scaffold_split_seed{seed}", cfg)
        rows.append({"split": f"scaffold_split_seed{seed}", "type": "scaffold_holdout", "train_n": len(train_idx), "test_n": len(test_idx)})

    n_folds = int(validation.get("scaffold_cv_folds", 5))
    gkf = GroupKFold(n_splits=n_folds) #makes sure molecules with the same bemis-murcko scaffold are in the same group 
    for fold, (train_idx, test_idx) in enumerate(gkf.split(std, y, groups=std["murcko_scaffold"]), start=1):
        write_split(std, train_idx, test_idx, f"scaffold_cv_fold{fold}", cfg)
        rows.append({"split": f"scaffold_cv_fold{fold}", "type": "primary_scaffold_cv", "train_n": len(train_idx), "test_n": len(test_idx)})

    if "source_dataset" in std and std["source_dataset"].nunique(dropna=True) > 1:
        sources = sorted(std["source_dataset"].dropna().unique())
        for source in sources:
            test_idx = std.index[std["source_dataset"] == source].to_numpy()
            train_idx = std.index[std["source_dataset"] != source].to_numpy()
            if len(test_idx) >= 10 and len(train_idx) >= 10 and std.iloc[test_idx]["label"].nunique() == 2:
                safe_source = str(source).replace("/", "_").replace(" ", "_")
                write_split(std, train_idx, test_idx, f"leave_source_out_{safe_source}", cfg)
                rows.append({"split": f"leave_source_out_{safe_source}", "type": "leave_source_out", "train_n": len(train_idx), "test_n": len(test_idx)})
    else:
        LOGGER.info("No usable source_dataset column; leave-source-out splits skipped.")

    rows.extend(import_legacy_padel_splits(std, cfg))

    summary = pd.DataFrame(rows)
    write_csv(summary, project_path(cfg, "reports/split_summary.csv"), cfg)
    return summary


def import_legacy_padel_splits(std: pd.DataFrame, cfg: dict) -> list[dict]:
    """Import random splits exported by legacy/notebooks/prepare_data_padel.ipynb as baseline-only references."""
    split_dir = resolve_path(cfg.get("paths", {}).get("legacy_padel_split_dir"), cfg)
    if split_dir is None or not split_dir.exists():
        LOGGER.info("No legacy PaDEL split directory configured; notebook split import skipped.")
        return []
    rows = []
    std_by_inchikey = std.drop_duplicates("inchikey").set_index("inchikey")
    for folder in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        x_train, x_test = folder / "x_train.csv", folder / "x_test.csv"
        if not x_train.exists() or not x_test.exists():
            continue

        def map_legacy(path):
            legacy = pd.read_csv(path, index_col=0)
            smiles_col = "Original_SMILES" if "Original_SMILES" in legacy.columns else "smiles"
            if smiles_col not in legacy.columns:
                return pd.DataFrame(columns=std.columns)
            inchikeys = [standardize_smiles(s).inchikey for s in legacy[smiles_col]]
            mapped = std_by_inchikey.reindex(inchikeys).dropna(subset=["molecule_id"]).reset_index()
            return mapped.drop_duplicates("molecule_id")

        train = map_legacy(x_train)
        test = map_legacy(x_test)
        prefix = f"notebook_random_{folder.name}"
        root = project_path(cfg, "data/splits")
        write_csv(split_frame(train), root / f"{prefix}_train.csv", cfg)
        write_csv(split_frame(test), root / f"{prefix}_test.csv", cfg)
        rows.append({"split": prefix, "type": "legacy_notebook_random_baseline", "train_n": len(train), "test_n": len(test)})
    if rows:
        LOGGER.info("Imported %d legacy random split(s) from prepare_data_padel outputs.", len(rows))
    return rows
