from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .chemistry import standardize_smiles
from .preprocessing import numeric_model_columns
from .utils import LOGGER, existing_file, project_path, read_table, resolve_path, write_csv


def load_standardized(cfg: dict) -> pd.DataFrame:
    return read_table(project_path(cfg, "data/processed/standardized_molecules.csv"))


def save_feature_matrix(name: str, X: pd.DataFrame, index: pd.DataFrame, cfg: dict) -> None:
    base = project_path(cfg, "data/processed")
    x_path = base / f"features_{name}.csv"
    i_path = base / f"features_{name}_index.csv"
    wrote_x = not (x_path.exists() and not cfg.get("overwrite", False))
    wrote_i = not (i_path.exists() and not cfg.get("overwrite", False))
    write_csv(X, x_path, cfg)
    write_csv(index, i_path, cfg)
    if not wrote_x or not wrote_i:
        LOGGER.info("Feature view %s already existed; set overwrite=true to regenerate it.", name)
        return
    LOGGER.info("Wrote feature view %s with shape %s", name, X.shape)


def load_morgan_dataframe(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    base = project_path(cfg, "data/processed")
    npz_path = base / "morgan_fingerprints.npz"
    index_path = base / "fingerprint_index.csv"
    if not npz_path.exists() or not index_path.exists():
        return None, None
    data = sparse.load_npz(npz_path).toarray()
    cols = pd.read_csv(base / "morgan_fingerprints_columns.csv")["feature"].tolist()
    return pd.DataFrame(data, columns=cols), pd.read_csv(index_path)


def load_embeddings_dataframe(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    base = project_path(cfg, "data/processed")
    emb_path = base / "pretrained_smiles_embeddings.npy"
    index_path = base / "pretrained_embedding_index.csv"
    if not emb_path.exists() or not index_path.exists():
        return None, None
    arr = np.load(emb_path)
    cols = [f"emb_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols), pd.read_csv(index_path)


def load_padel_dataframe(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    padel_path = resolve_path(cfg.get("paths", {}).get("padel_descriptor_path"), cfg)
    if not existing_file(padel_path):
        LOGGER.warning("PaDEL descriptor file not found: %s", padel_path)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    padel = read_table(padel_path)
    keep, excluded = numeric_model_columns(padel)
    X = padel[keep].apply(pd.to_numeric, errors="coerce")

    std = load_standardized(cfg)
    smiles_col = cfg.get("columns", {}).get("smiles", "smiles")
    candidates = [c for c in ["canonical_smiles", smiles_col, "smiles", "Original_SMILES"] if c in padel.columns]
    if candidates and "canonical_smiles" in std.columns:
        key = candidates[0]
        key_values = padel[key].apply(lambda s: standardize_smiles(s).canonical_smiles if pd.notna(s) else None)
        std_cols = [
            c
            for c in ["molecule_id", "canonical_smiles", "inchikey", "murcko_scaffold", "label", "source_dataset"]
            if c in std.columns
        ]
        idx = pd.DataFrame({"canonical_smiles": key_values}).merge(
            std[std_cols],
            on="canonical_smiles",
            how="left",
        )
    else:
        idx = std[["molecule_id", "canonical_smiles", "inchikey", "label"]].head(len(X)).copy()
    valid_rows = idx["molecule_id"].notna() if "molecule_id" in idx else pd.Series(False, index=idx.index)
    X = X.loc[valid_rows].reset_index(drop=True)
    idx = idx.loc[valid_rows].reset_index(drop=True)

    # PaDEL is loaded from the raw descriptor table, which can still contain
    # duplicate source rows. Collapse to the same unique molecule set used by
    # standardized_molecules.csv so every feature view has one row per molecule.
    keep_rows = ~idx["molecule_id"].duplicated(keep="first")
    return X.loc[keep_rows].reset_index(drop=True), idx.loc[keep_rows].reset_index(drop=True), excluded


def align_by_molecule_id(parts: list[tuple[pd.DataFrame, pd.DataFrame, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_parts = []
    matrices = []
    for X, idx, prefix in parts:
        if X is None or idx is None or X.empty:
            continue
        idx = idx.reset_index(drop=True)
        X = X.reset_index(drop=True).add_prefix(prefix)
        part = pd.concat([idx[["molecule_id"]], X], axis=1)
        part = part.drop_duplicates("molecule_id", keep="first")
        idx_meta = idx.drop_duplicates("molecule_id", keep="first")
        index_parts.append(idx_meta)
        matrices.append(part)
    if not matrices:
        return pd.DataFrame(), pd.DataFrame()
    out = matrices[0]
    for part in matrices[1:]:
        out = out.merge(part, on="molecule_id", how="inner")
    index = load_standardized_from_ids(out["molecule_id"].tolist(), index_parts)
    return out.drop(columns=["molecule_id"]), index


def load_standardized_from_ids(ids: list, fallback_indexes: list[pd.DataFrame] | None) -> pd.DataFrame:
    if fallback_indexes:
        metadata = pd.concat(fallback_indexes, ignore_index=True, sort=False)
        cols = [
            c
            for c in ["molecule_id", "canonical_smiles", "inchikey", "murcko_scaffold", "label", "source_dataset"]
            if c in metadata
        ]
        metadata = metadata[cols].groupby("molecule_id", as_index=False).first()
        return metadata.set_index("molecule_id").loc[ids].reset_index()
    return pd.DataFrame({"molecule_id": ids})


def build_feature_matrices(cfg: dict) -> None:
    padel_X, padel_idx, excluded = load_padel_dataframe(cfg)
    if not excluded.empty:
        write_csv(excluded, project_path(cfg, "reports/excluded_non_model_columns.csv"), cfg)
    morgan_X, morgan_idx = load_morgan_dataframe(cfg)
    emb_X, emb_idx = load_embeddings_dataframe(cfg)

    if not padel_X.empty:
        save_feature_matrix("padel", padel_X, padel_idx, cfg)
    if morgan_X is not None:
        save_feature_matrix("morgan", morgan_X, morgan_idx, cfg)
    if emb_X is not None:
        save_feature_matrix("embeddings", emb_X, emb_idx, cfg)

    combos = {
        "padel_morgan": [(padel_X, padel_idx, "padel__"), (morgan_X, morgan_idx, "morgan__")],
        "padel_embeddings": [(padel_X, padel_idx, "padel__"), (emb_X, emb_idx, "emb__")],
        "padel_morgan_embeddings": [
            (padel_X, padel_idx, "padel__"),
            (morgan_X, morgan_idx, "morgan__"),
            (emb_X, emb_idx, "emb__"),
        ],
    }
    for name, parts in combos.items():
        X, idx = align_by_molecule_id(parts)
        if not X.empty:
            save_feature_matrix(name, X, idx, cfg)


def load_feature_view(cfg: dict, view: str) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    base = project_path(cfg, "data/processed")
    x_path = base / f"features_{view}.csv"
    i_path = base / f"features_{view}_index.csv"
    if not x_path.exists() or not i_path.exists():
        return None, None
    return pd.read_csv(x_path), pd.read_csv(i_path)
