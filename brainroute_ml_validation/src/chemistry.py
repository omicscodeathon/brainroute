from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class StandardizedMol:
    canonical_smiles: str | None
    inchikey: str | None
    murcko_scaffold: str | None
    valid: bool
    error: str | None = None


def _rdkit_imports():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return Chem, AllChem, DataStructs, MurckoScaffold


def standardize_smiles(smiles: str) -> StandardizedMol:
    Chem, _, _, MurckoScaffold = _rdkit_imports()
    if smiles is None or str(smiles).strip() == "":
        return StandardizedMol(None, None, None, False, "empty_smiles")
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return StandardizedMol(None, None, None, False, "rdkit_parse_failed")
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(mol)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return StandardizedMol(canonical, inchikey, scaffold or "NO_SCAFFOLD", True, None)
    except Exception as exc:
        return StandardizedMol(None, None, None, False, str(exc))


def morgan_bitvect(smiles: str, radius: int = 2, n_bits: int = 2048, use_chirality: bool = True):
    Chem, AllChem, _, _ = _rdkit_imports()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        generator = AllChem.GetMorganGenerator(
            radius=radius,
            fpSize=n_bits,
            includeChirality=use_chirality,
        )
        return generator.GetFingerprint(mol)
    except AttributeError:
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, useChirality=use_chirality
        )


def bitvect_to_array(fp, n_bits: int) -> np.ndarray:
    _, _, DataStructs, _ = _rdkit_imports()
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def calculate_morgan_matrix(
    smiles_values: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> tuple[np.ndarray, list[int], list[Any]]:
    rows: list[np.ndarray] = []
    valid_positions: list[int] = []
    fps = []
    for idx, smiles in enumerate(smiles_values):
        fp = morgan_bitvect(smiles, radius, n_bits, use_chirality)
        if fp is None:
            continue
        rows.append(bitvect_to_array(fp, n_bits))
        valid_positions.append(idx)
        fps.append(fp)
    if not rows:
        return np.empty((0, n_bits), dtype=np.uint8), valid_positions, fps
    return np.vstack(rows).astype(np.uint8), valid_positions, fps


def max_tanimoto_to_train(test_fps: list, train_fps: list) -> list[dict]:
    _, _, DataStructs, _ = _rdkit_imports()
    out = []
    for fp in test_fps:
        sims = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) if train_fps else []
        if sims:
            nearest_idx = int(np.argmax(sims))
            out.append({"max_tanimoto": float(sims[nearest_idx]), "nearest_train_position": nearest_idx})
        else:
            out.append({"max_tanimoto": np.nan, "nearest_train_position": None})
    return out
