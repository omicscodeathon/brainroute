#!/usr/bin/env python3
"""Generate resumable GFN2-xTB single-point descriptors for the locked cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "data/benchmarks/matched_3d_qm"
DEFAULT_XTB = Path(shutil.which("xtb") or "xtb")
POLARIZABILITY_RE = re.compile(r"Mol\.\s+[^:]*\(0\)\s+/au\s*:\s*([-+0-9.Ee]+)")


def deterministic_seed(identifier: str) -> int:
    return int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:7], 16)


def cache_path(root: Path, benchmark_id: str) -> Path:
    digest = hashlib.sha256(benchmark_id.encode("utf-8")).hexdigest()
    return root / digest[:2] / f"{digest}.json"


def write_xyz(mol: Chem.Mol, path: Path) -> str:
    with_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = deterministic_seed(Chem.MolToSmiles(mol, isomericSmiles=True))
    params.useRandomCoords = False
    status = AllChem.EmbedMolecule(with_h, params)
    embed_method = "ETKDGv3"
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(with_h, params)
        embed_method = "ETKDGv3_random_coordinates"
    if status != 0:
        raise ValueError("RDKit conformer generation failed")
    if AllChem.MMFFHasAllMoleculeParams(with_h):
        AllChem.MMFFOptimizeMolecule(with_h, mmffVariant="MMFF94s", maxIters=300)
        force_field = "MMFF94s"
    else:
        AllChem.UFFOptimizeMolecule(with_h, maxIters=300)
        force_field = "UFF"
    conf = with_h.GetConformer()
    lines = [str(with_h.GetNumAtoms()), f"{embed_method};{force_field}"]
    for index, atom in enumerate(with_h.GetAtoms()):
        pos = conf.GetAtomPosition(index)
        lines.append(f"{atom.GetSymbol()} {pos.x:.10f} {pos.y:.10f} {pos.z:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"{embed_method};{force_field}"


def frontier_orbitals(payload: dict) -> tuple[float, float]:
    energies = np.asarray(payload.get("orbital energies / eV", []), dtype=float)
    occupations = np.asarray(payload.get("fractional occupation", []), dtype=float)
    if energies.size == 0 or energies.size != occupations.size:
        return math.nan, math.nan
    occupied = energies[occupations > 1e-6]
    virtual = energies[occupations <= 1e-6]
    return (float(occupied.max()) if occupied.size else math.nan, float(virtual.min()) if virtual.size else math.nan)


def summarize(payload: dict, stdout: str, work_dir: Path) -> dict:
    charges = np.asarray(payload.get("partial charges", []), dtype=float)
    dipole = np.asarray(payload.get("dipole / a.u.", []), dtype=float)
    atomic_dipoles = np.asarray(payload.get("atomic dipole moments", []), dtype=float)
    homo, lumo = frontier_orbitals(payload)
    match = POLARIZABILITY_RE.search(stdout)
    wbo_path = work_dir / "wbo"
    wbo = []
    if wbo_path.exists():
        for line in wbo_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    wbo.append(float(parts[2]))
                except ValueError:
                    pass
    wbo_arr = np.asarray(wbo, dtype=float)
    atomic_dipole_norm = (
        np.linalg.norm(atomic_dipoles, axis=1)
        if atomic_dipoles.ndim == 2 and atomic_dipoles.shape[1] == 3
        else np.asarray([], dtype=float)
    )
    return {
        "xtb__total_energy_hartree": payload.get("total energy", math.nan),
        "xtb__electronic_energy_hartree": payload.get("electronic energy", math.nan),
        "xtb__homo_ev": homo,
        "xtb__lumo_ev": lumo,
        "xtb__homo_lumo_gap_ev": payload.get("HOMO-LUMO gap / eV", math.nan),
        "xtb__dipole_debye": float(np.linalg.norm(dipole) * 2.541746) if dipole.size == 3 else math.nan,
        "xtb__polarizability_au": float(match.group(1)) if match else math.nan,
        "xtb__charge_min": float(charges.min()) if charges.size else math.nan,
        "xtb__charge_max": float(charges.max()) if charges.size else math.nan,
        "xtb__charge_std": float(charges.std()) if charges.size else math.nan,
        "xtb__charge_mean_absolute": float(np.abs(charges).mean()) if charges.size else math.nan,
        "xtb__atomic_dipole_mean": float(atomic_dipole_norm.mean()) if atomic_dipole_norm.size else math.nan,
        "xtb__atomic_dipole_max": float(atomic_dipole_norm.max()) if atomic_dipole_norm.size else math.nan,
        "xtb__wbo_mean": float(wbo_arr.mean()) if wbo_arr.size else math.nan,
        "xtb__wbo_std": float(wbo_arr.std()) if wbo_arr.size else math.nan,
        "xtb__wbo_max": float(wbo_arr.max()) if wbo_arr.size else math.nan,
        "xtb__wbo_sum": float(wbo_arr.sum()) if wbo_arr.size else math.nan,
        "xtb__number_electrons": payload.get("number of electrons", math.nan),
        "xtb__number_unpaired_electrons": payload.get("number of unpaired electrons", math.nan),
        "xtb_method": payload.get("method"),
        "xtb_version": payload.get("xtb version"),
    }


def calculate_one(task: dict) -> dict:
    cache = cache_path(Path(task["cache_root"]), task["benchmark_id"])
    if cache.exists():
        with cache.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    started = time.perf_counter()
    record = {
        "benchmark_id": task["benchmark_id"],
        "cohort": task["cohort"],
        "status": "failed",
        "error": None,
    }
    try:
        mol = Chem.MolFromSmiles(task["calculation_smiles"])
        if mol is None:
            raise ValueError("RDKit could not parse calculation_smiles")
        with tempfile.TemporaryDirectory(dir=task["tmp_root"], prefix="xtb_") as tmp:
            work_dir = Path(tmp)
            conformer_method = write_xyz(mol, work_dir / "molecule.xyz")
            command = [
                task["xtb"],
                "molecule.xyz",
                "--sp",
                "--gfn",
                "2",
                "--chrg",
                str(int(task["charge"])),
                "--uhf",
                str(int(task["unpaired"])),
                "--json",
                "--wbo",
                "--alpha",
                "--parallel",
                "1",
                "--norestart",
                "--ceasefiles",
            ]
            environment = os.environ.copy()
            environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            result = subprocess.run(
                command,
                cwd=work_dir,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=int(task["timeout"]),
                check=False,
            )
            json_path = work_dir / "xtbout.json"
            if result.returncode != 0 or not json_path.exists():
                tail = "\n".join(result.stdout.splitlines()[-20:])
                raise RuntimeError(f"xTB exit code {result.returncode}: {tail}")
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            record.update(summarize(payload, result.stdout, work_dir))
            record["conformer_preparation"] = conformer_method
            record["status"] = "success"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_seconds"] = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    temporary.replace(cache)
    return record


def select_pilot(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame
    ranked = frame.sort_values(
        ["cohort", "calculation_heavy_atoms", "calculation_formal_charge", "benchmark_id"]
    ).reset_index(drop=True)
    positions = np.linspace(0, len(ranked) - 1, n, dtype=int)
    return ranked.iloc[np.unique(positions)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--xtb", type=Path, default=DEFAULT_XTB)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.workspace / "artifacts/benchmark_manifest.csv")
    selected = select_pilot(manifest, args.pilot) if args.pilot else manifest
    cache_root = args.workspace / "cache/xtb_gfn2"
    if args.retry_failures:
        for identifier in selected["benchmark_id"]:
            path = cache_path(cache_root, identifier)
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                if cached.get("status") != "success":
                    path.unlink()

    tasks = []
    for row in selected.to_dict("records"):
        tasks.append(
            {
                "benchmark_id": row["benchmark_id"],
                "cohort": row["cohort"],
                "calculation_smiles": row["calculation_smiles"],
                "charge": row["calculation_formal_charge"],
                "unpaired": row["calculation_unpaired_electrons"],
                "cache_root": str(cache_root),
                "tmp_root": str(args.workspace / "tmp"),
                "xtb": str(args.xtb),
                "timeout": args.timeout,
            }
        )
    started = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(calculate_one, task): task["benchmark_id"] for task in tasks}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            rows.append(record)
            if completed % 100 == 0 or completed == len(futures):
                successes = sum(row["status"] == "success" for row in rows)
                elapsed = time.perf_counter() - started
                print(f"completed={completed}/{len(futures)} success={successes} elapsed_seconds={elapsed:.1f}", flush=True)

    output = pd.DataFrame(rows).sort_values("benchmark_id")
    suffix = f"_pilot{args.pilot}" if args.pilot else ""
    output_path = args.workspace / f"artifacts/xtb_features{suffix}.csv"
    output.to_csv(output_path, index=False)
    summary = (
        output.groupby(["cohort", "status"], dropna=False, as_index=False)
        .agg(n=("benchmark_id", "size"), median_seconds=("wall_seconds", "median"), total_seconds=("wall_seconds", "sum"))
    )
    summary["elapsed_wall_seconds"] = time.perf_counter() - started
    summary.to_csv(args.workspace / f"artifacts/xtb_run_summary{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
